import base64
import hashlib
import json
import re
import secrets
import time
import uuid
from dataclasses import replace
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from starlette.testclient import TestClient

from telegram_bridge.auth import permitted_callback
from telegram_bridge.server import create_app
from telegram_bridge.telegram import TelegramAPI
from tests.test_bridge import PASSWORD, DUMMY_TOKEN, config, store, seed, update

CALLBACK = "https://chatgpt.com/connector/oauth/test_callback"


@pytest.fixture
def client(config, store):
    def no_external_requests(request):
        raise AssertionError("No external requests are expected in MCP/auth tests")
    api = TelegramAPI(DUMMY_TOKEN, httpx.AsyncClient(transport=httpx.MockTransport(no_external_requests)))
    app = create_app(config, store, api, start_poller=False)
    with TestClient(app, base_url=config.public_base_url, follow_redirects=False) as client:
        yield client


def start_login(client, config, scopes=None):
    scopes = scopes or config.scopes
    response = client.post("/register", json={"redirect_uris": [CALLBACK],
        "client_name": "ChatGPT test", "token_endpoint_auth_method": "client_secret_post",
        "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"],
        "scope": " ".join(scopes)})
    assert response.status_code == 201, response.text
    oauth_client = response.json()
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    response = client.get("/authorize", params={"response_type": "code", "client_id": oauth_client["client_id"],
        "redirect_uri": CALLBACK, "scope": " ".join(scopes), "state": "test-state",
        "code_challenge": challenge, "code_challenge_method": "S256", "resource": config.resource})
    assert response.status_code in {302, 303, 307}, response.text
    login = client.get(response.headers["location"])
    assert login.status_code == 200
    rid = re.search(r'name="request" value="([^"]+)"', login.text).group(1)
    return oauth_client, verifier, rid


def get_code(client, config, scopes=None):
    oauth_client, verifier, rid = start_login(client, config, scopes)
    response = client.post("/connect", headers={"Origin": config.public_base_url},
                           data={"request": rid, "password": PASSWORD})
    assert response.status_code == 303, response.text
    query = parse_qs(urlsplit(response.headers["location"]).query)
    assert query["state"] == ["test-state"]
    return {"grant_type": "authorization_code", "code": query["code"][0],
            "client_id": oauth_client["client_id"], "client_secret": oauth_client["client_secret"],
            "redirect_uri": CALLBACK, "code_verifier": verifier, "resource": config.resource}


def authorize(client, config, scopes=None):
    form = get_code(client, config, scopes)
    response = client.post("/token", data=form)
    assert response.status_code == 200, response.text
    return response.json(), form


def rpc(client, token, method, params=None, id=1):
    headers = {"Authorization": "Bearer " + token,
               "Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-06-18"}
    request = {"jsonrpc": "2.0", "id": id, "method": method}
    if params is not None:
        request["params"] = params
    return client.post("/mcp", headers=headers, json=request)


def test_anonymous_denied_metadata_available(client, config):
    response = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert response.status_code == 401
    assert "oauth-protected-resource/mcp" in response.headers["www-authenticate"]
    metadata = client.get("/.well-known/oauth-protected-resource/mcp").json()
    assert metadata["resource"] == config.resource
    discovery = client.get("/.well-known/oauth-authorization-server").json()
    assert discovery["code_challenge_methods_supported"] == ["S256"]
    assert "registration_endpoint" in discovery
    assert client.get("/healthz").status_code == 200


def test_complete_oauth_and_mcp_search_fetch(client, config, store):
    seed(store, config, [update()])
    tokens, _ = authorize(client, config)
    token = tokens["access_token"]
    initial = rpc(client, token, "initialize", {"protocolVersion": "2025-06-18",
                  "capabilities": {}, "clientInfo": {"name": "integration-test", "version": "1.0"}})
    assert initial.status_code == 200, initial.text
    assert initial.json()["result"]["serverInfo"]["name"] == "Fahad Telegram"
    response = rpc(client, token, "tools/list")
    assert response.status_code == 200, response.text
    tools = {t["name"]: t for t in response.json()["result"]["tools"]}
    assert {"search", "fetch", "list_chats", "list_messages", "get_status", "send_message",
            "predict_market", "backtest_market"} == set(tools)
    assert tools["search"]["annotations"]["readOnlyHint"]
    assert not tools["send_message"]["annotations"]["readOnlyHint"]
    response = rpc(client, token, "tools/call", {"name": "search", "arguments": {"query": "فهد"}})
    content = response.json()["result"]["content"]
    assert len(content) == 1 and content[0]["type"] == "text"
    matches = json.loads(content[0]["text"])["results"]
    assert len(matches) == 1 and set(matches[0]) == {"id", "title", "url"}
    response = rpc(client, token, "tools/call", {"name": "fetch", "arguments": {"id": matches[0]["id"]}})
    message = json.loads(response.json()["result"]["content"][0]["text"])
    assert "مرحبا" in message["text"]
    assert not message["metadata"]["exact_message_url"]
    assert DUMMY_TOKEN not in initial.text + response.text
    # Even when the poller is stopped, a retained row must not be readable after expiry.
    with store.transaction() as db:
        db.execute("UPDATE messages SET date=?", (int(time.time()) - 31 * 86400,))
    expired = rpc(client, token, "tools/call", {"name": "fetch", "arguments": {"id": matches[0]["id"]}})
    assert expired.json()["result"]["isError"] is True
    hidden = rpc(client, token, "tools/call", {"name": "search", "arguments": {"query": ""}})
    assert json.loads(hidden.json()["result"]["content"][0]["text"])["results"] == []


def test_pkce_audience_and_replay(client, config):
    form = get_code(client, config)
    bad = client.post("/token", data={**form, "code_verifier": "wrong"})
    assert bad.status_code == 400
    wrong_resource = client.post("/token", data={**form, "resource": "https://attacker.invalid/mcp"})
    assert wrong_resource.status_code == 400
    good = client.post("/token", data=form)
    assert good.status_code == 200, good.text
    assert client.post("/token", data=form).status_code == 400


def test_refresh_rotation_and_revocation(client, config):
    first, form = authorize(client, config)
    refresh_form = {"grant_type": "refresh_token", "client_id": form["client_id"],
                    "client_secret": form["client_secret"], "refresh_token": first["refresh_token"],
                    "resource": config.resource}
    response = client.post("/token", data=refresh_form)
    assert response.status_code == 200, response.text
    second = response.json()
    assert second["access_token"] != first["access_token"]
    assert rpc(client, first["access_token"], "tools/list").status_code == 401
    assert rpc(client, second["access_token"], "tools/list").status_code == 200
    assert client.post("/token", data=refresh_form).status_code == 400
    revoked = client.post("/revoke", data={"token": second["refresh_token"],
        "client_id": form["client_id"], "client_secret": form["client_secret"]})
    assert revoked.status_code == 200
    assert rpc(client, second["access_token"], "tools/list").status_code == 401


def test_read_scope_cannot_send(client, config):
    tokens, _ = authorize(client, config, ["telegram:read"])
    response = rpc(client, tokens["access_token"], "tools/call", {"name": "send_message",
        "arguments": {"chat_id": "101", "text": "do not send", "request_id": str(uuid.uuid4())}})
    assert response.status_code == 200
    assert response.json()["result"]["isError"] is True


def test_prediction_and_backtest_over_authenticated_mcp(client, config, monkeypatch):
    from telegram_bridge.demo import demo_series
    series = demo_series()
    monkeypatch.setattr(client.app.state.bridge.predictions.market, "load", lambda symbol: series)
    tokens, _ = authorize(client, config, ["telegram:read"])
    token = tokens["access_token"]
    response = rpc(client, token, "tools/call", {"name": "predict_market",
        "arguments": {"symbol": "DEMO", "horizon_sessions": 5}})
    assert response.status_code == 200 and not response.json()["result"].get("isError")
    result = json.loads(response.json()["result"]["content"][0]["text"])
    assert result["synthetic"] and result["signal"] == "abstain" and 0 < result["probability_up"] < 1
    assert "rows" not in result["backtest"]
    response = rpc(client, token, "tools/call", {"name": "backtest_market",
        "arguments": {"symbol": "DEMO", "horizon_sessions": 5}})
    result = json.loads(response.json()["result"]["content"][0]["text"])
    assert result["backtest"]["n"] == len(result["backtest"]["rows"]) == 40
    invalid = rpc(client, token, "tools/call", {"name": "predict_market",
        "arguments": {"symbol": "DEMO", "horizon_sessions": 2}})
    assert invalid.json()["result"]["isError"] is True


def test_login_csrf_and_bad_password(client, config):
    _, _, rid = start_login(client, config)
    assert client.post("/connect", data={"request": rid, "password": PASSWORD}).status_code == 403
    assert client.post("/connect", headers={"Origin": "https://attacker.invalid"},
                       data={"request": rid, "password": PASSWORD}).status_code == 403
    wrong = client.post("/connect", headers={"Origin": config.public_base_url},
                         data={"request": rid, "password": "wrong"})
    assert wrong.status_code == 403
    assert client.post("/connect", headers={"Origin": config.public_base_url},
                       data={"request": rid, "password": PASSWORD}).status_code == 303


def test_host_origin_redirect_and_body_guards(client, config):
    assert client.get("/healthz", headers={"Host": "attacker.invalid"}).status_code == 400
    assert client.post("/register", content=b"x" * 65537).status_code == 413
    response = client.post("/register", json={"redirect_uris": ["https://attacker.invalid/callback"],
        "token_endpoint_auth_method": "client_secret_post"})
    assert response.status_code == 400
    for uri in ("https://chatgpt.com.attacker.invalid/connector/oauth/id",
                "https://chatgpt.com/connector/oauth/id?next=https://attacker.invalid",
                "https://chatgpt.com@attacker.invalid/connector/oauth/id"):
        assert not permitted_callback(uri)


def test_token_values_not_stored_as_plaintext(client, config, store):
    tokens, form = authorize(client, config)
    rows = str([tuple(row) for row in store.db.execute("SELECT kind,key,value FROM oauth")])
    assert tokens["access_token"] not in rows
    assert tokens["refresh_token"] not in rows
    assert form["code"] not in rows
    assert PASSWORD not in rows
