from __future__ import annotations

import asyncio
import html
import re
import secrets
import time
from urllib.parse import urlencode, urlsplit

from mcp.server.auth.provider import (
    AccessToken, AuthorizationCode, AuthorizationParams, AuthorizeError,
    RefreshToken, RegistrationError, TokenError, construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse, RedirectResponse

from .config import Config, check_password
from .storage import Store, digest


def permitted_callback(uri: str) -> bool:
    """DCR is only for ChatGPT callbacks; no arbitrary redirects or outbound fetches."""
    parsed = urlsplit(uri)
    return (parsed.scheme == "https" and parsed.netloc == "chatgpt.com"
            and not parsed.query and not parsed.fragment
            and bool(re.fullmatch(r"/connector_platform_oauth_redirect|/connector/oauth/[A-Za-z0-9_-]+", parsed.path)))


class OwnerOAuth:
    """Single-owner authorization using the official MCP SDK's PKCE/token endpoints.

    Telegram credentials stay server-side. OAuth bearer/code values are stored by
    hash only. DCR client secrets are in the mode-0600 database because the SDK
    verifies them directly. This is a private connector, not a multi-tenant IdP.
    """

    def __init__(self, config: Config, store: Store):
        self.config, self.store = config, store

    async def get_client(self, client_id: str):
        info = self.store.oauth_get("client", client_id)
        return OAuthClientInformationFull.model_validate(info) if info else None

    async def register_client(self, client_info: OAuthClientInformationFull):
        if not client_info.redirect_uris or any(not permitted_callback(str(uri)) for uri in client_info.redirect_uris):
            raise RegistrationError("invalid_redirect_uri", "Only exact ChatGPT callback URLs are accepted.")
        if client_info.token_endpoint_auth_method not in {"client_secret_post", "client_secret_basic"}:
            raise RegistrationError("invalid_client_metadata", "Use client_secret_post or client_secret_basic.")
        if self.store.oauth_count("client") >= 100:
            raise RegistrationError("invalid_client_metadata", "Registration capacity reached. Contact the owner.")
        self.store.oauth_put("client", client_info.client_id, client_info.model_dump(mode="json"))

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams):
        if params.resource != self.config.resource:
            raise AuthorizeError("invalid_request", "The resource must match this MCP endpoint.")
        if not params.scopes or not set(params.scopes).issubset(self.config.scopes):
            raise AuthorizeError("invalid_scope", "Invalid or missing Telegram scopes.")
        if not permitted_callback(str(params.redirect_uri)):
            raise AuthorizeError("invalid_request", "Callback is not allowed.")
        rid = secrets.token_urlsafe(32)
        expiry = time.time() + 600
        self.store.oauth_put("login", rid, {
            "client_id": client.client_id, "params": params.model_dump(mode="json"),
            "expires": expiry, "attempts": 0}, expiry)
        return self.config.public_base_url + "/connect?" + urlencode({"request": rid})

    async def login(self, request: Request):
        headers = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer",
                   "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'"}
        if request.method == "GET":
            rid = request.query_params.get("request", "")
            flow = self.store.oauth_get("login", rid)
            if not flow:
                return PlainTextResponse("Connection request expired. Start the connection again in ChatGPT.", 400, headers)
            browser_nonce = secrets.token_urlsafe(32)
            flow["browser_nonce"] = digest(browser_nonce)
            self.store.oauth_put("login", rid, flow, flow["expires"])
            sending = "وإرسال الرسائل عند طلبك" if "telegram:send" in flow["params"]["scopes"] else "للقراءة فقط"
            body = """<!doctype html><html lang="ar" dir="rtl"><meta charset="utf-8">
            <meta name="viewport" content="width=device-width,initial-scale=1"><title>ربط Telegram</title>
            <style>body{font:18px system-ui;background:#f4f6fa;color:#172b43;margin:0;padding:24px}
            main{max-width:430px;margin:8vh auto;background:white;padding:28px;border-radius:20px}
            input,button{font:inherit;box-sizing:border-box;width:100%;padding:14px;margin-top:16px;border-radius:9px}
            input{border:1px solid #bac5d4}button{border:0;background:#176cbd;color:white}p{line-height:1.8}</style>
            <main><h1>ربط Telegram بـChatGPT</h1><p>السماح بالوصول إلى رسائل البوت @BOT_USER MODE.</p>
            <p>أدخل كلمة مرور الربط التي أنشأها برنامج الإعداد. هذه ليست كلمة مرور Telegram.</p>
            <form method="post" action="/connect"><input type="hidden" name="request" value="RID">
            <label for="password">كلمة مرور الربط</label>
            <input id="password" name="password" type="password" autocomplete="current-password" required maxlength="256">
            <button type="submit">السماح بالربط</button></form></main></html>"""
            body = body.replace("BOT_USER", html.escape(self.config.expected_bot_username)).replace("MODE", sending).replace("RID", html.escape(rid, quote=True))
            response = HTMLResponse(body, headers=headers)
            response.set_cookie("bridge_connect", browser_nonce, max_age=600, httponly=True,
                                secure=self.config.public_base_url.startswith("https:"), samesite="strict", path="/connect")
            return response

        if request.headers.get("origin") != self.config.public_base_url:
            return PlainTextResponse("Invalid form origin.", 403, headers)
        form = await request.form()
        rid, password = str(form.get("request", "")), str(form.get("password", ""))
        flow = self.store.oauth_get("login", rid)
        nonce = request.cookies.get("bridge_connect", "")
        if not flow or not nonce or not secrets.compare_digest(flow.get("browser_nonce", ""), digest(nonce)):
            return PlainTextResponse("Connection request expired. Restart from ChatGPT.", 400, headers)
        if flow["attempts"] >= 5 or len(password) > 256:
            self.store.oauth_consume("login", rid)
            return PlainTextResponse("Too many attempts. Restart from ChatGPT.", 429, headers)
        if not await asyncio.to_thread(check_password, password, self.config.password_hash):
            flow["attempts"] += 1
            self.store.oauth_put("login", rid, flow, flow["expires"])
            return PlainTextResponse("Incorrect connection password. Go back and try again.", 403, headers)
        # Consuming the login request prevents two successful POSTs from issuing two codes.
        flow = self.store.oauth_consume("login", rid)
        if not flow:
            return PlainTextResponse("This request was already used.", 400, headers)
        params = AuthorizationParams.model_validate(flow["params"])
        code = secrets.token_urlsafe(32)
        auth_code = AuthorizationCode(code=code, client_id=flow["client_id"], scopes=params.scopes,
                                      expires_at=time.time() + 120, code_challenge=params.code_challenge,
                                      redirect_uri=params.redirect_uri,
                                      redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
                                      resource=params.resource, subject="owner")
        self.store.oauth_put("code", code, auth_code.model_dump(mode="json", exclude={"code"}), auth_code.expires_at)
        response = RedirectResponse(construct_redirect_uri(str(params.redirect_uri), code=code, state=params.state),
                                    status_code=303, headers=headers)
        response.delete_cookie("bridge_connect", path="/connect")
        return response

    async def load_authorization_code(self, client, authorization_code):
        value = self.store.oauth_get("code", authorization_code)
        if not value or value["client_id"] != client.client_id:
            return None
        return AuthorizationCode(code=authorization_code, **value)

    def issue(self, client_id: str, scopes: list[str], family: str | None = None):
        family = family or secrets.token_urlsafe(24)
        access, refresh = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        now = int(time.time())
        self.store.oauth_put("access", access, {"client_id": client_id, "scopes": scopes,
                             "expires_at": now + 3600, "resource": self.config.resource,
                             "subject": "owner", "family": family}, now + 3600, family)
        self.store.oauth_put("refresh", refresh, {"client_id": client_id, "scopes": scopes,
                             "expires_at": now + 30 * 86400, "subject": "owner",
                             "family": family}, now + 30 * 86400, family)
        return OAuthToken(access_token=access, token_type="Bearer", expires_in=3600,
                          refresh_token=refresh, scope=" ".join(scopes))

    async def exchange_authorization_code(self, client, authorization_code):
        value = self.store.oauth_consume("code", authorization_code.code)
        if not value or value["client_id"] != client.client_id:
            raise TokenError("invalid_grant", "Authorization code expired or was already used.")
        return self.issue(client.client_id, value["scopes"])

    async def load_refresh_token(self, client, refresh_token):
        value = self.store.oauth_get("refresh", refresh_token)
        if not value or value["client_id"] != client.client_id:
            return None
        return RefreshToken(token=refresh_token, **{k: v for k, v in value.items() if k != "family"})

    async def exchange_refresh_token(self, client, refresh_token, scopes):
        value = self.store.oauth_consume("refresh", refresh_token.token)
        if not value or value["client_id"] != client.client_id:
            raise TokenError("invalid_grant", "Refresh token expired or was already used.")
        if not set(scopes).issubset(value["scopes"]) or not set(scopes).issubset(self.config.scopes):
            raise TokenError("invalid_scope", "Requested scopes are not allowed.")
        self.store.revoke_family(value["family"])
        return self.issue(client.client_id, scopes, value["family"])

    async def load_access_token(self, token):
        value = self.store.oauth_get("access", token)
        if not value or value.get("resource") != self.config.resource:
            return None
        if not set(value["scopes"]).issubset(self.config.scopes):
            return None
        return AccessToken(token=token, **{k: v for k, v in value.items() if k != "family"})

    async def revoke_token(self, token):
        kind = "refresh" if isinstance(token, RefreshToken) else "access"
        value = self.store.oauth_get(kind, token.token)
        if value:
            self.store.revoke_family(value["family"])
