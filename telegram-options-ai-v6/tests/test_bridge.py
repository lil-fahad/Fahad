import asyncio
import json
import time
import uuid
from dataclasses import replace

import httpx
import pytest

from telegram_bridge.config import Config, hash_password
from telegram_bridge.storage import Store
from telegram_bridge.telegram import Bridge, TelegramAPI, TelegramError

PASSWORD = "test-connection-password-only"
DUMMY_TOKEN = "123456789:" + "x" * 35


@pytest.fixture
def config(tmp_path):
    return Config(bot_token=DUMMY_TOKEN, allowed_chat_ids=(101,), password_hash=hash_password(PASSWORD),
                  data_dir=tmp_path / "state", allow_send=True)


@pytest.fixture
def store(config):
    db = Store(config.data_dir / "bridge.sqlite3")
    yield db
    db.close()


def message(chat_id=101, mid=1, text="مرحبا يا فهد", date=None):
    return {"message_id": mid, "date": date or int(time.time()), "text": text,
            "chat": {"id": chat_id, "type": "private", "first_name": "Fahad"},
            "from": {"id": chat_id}}


def update(uid=10, **kwargs):
    return {"update_id": uid, "message": message(**kwargs)}


def seed(store, config, updates):
    store.ingest(updates, config.allowed_chat_ids, config.expected_bot_username,
                 config.retention_days, config.max_messages)


def test_allowlist_and_atomic_cursor(config, store):
    seed(store, config, [update(10), update(11, chat_id=666, text="private other chat")])
    assert store.get("offset") == "12"
    assert len(store.messages((101,))) == 1
    assert not store.fetch("666:1", (666,))
    with pytest.raises(KeyError):
        seed(store, config, [{"update_id": 12, "message": {"chat": {"id": 101}}}])
    assert store.get("offset") == "12"


def test_duplicate_and_edit(config, store):
    seed(store, config, [update()])
    seed(store, config, [update()])
    edited = message(text="تم التعديل")
    edited["edit_date"] = int(time.time())
    seed(store, config, [{"update_id": 11, "edited_message": edited}])
    assert len(store.messages((101,))) == 1
    assert store.fetch("101:1", (101,))["text"] == "تم التعديل"
    assert store.fetch("101:1", (101,))["metadata"]["edited"]


def test_search_pagination_retention_and_removal(config, store):
    seed(store, config, [update(20 + i, mid=i + 1, text=f"كرسي {i}") for i in range(5)]
                        + [update(30, mid=100, text="old", date=int(time.time()) - 31 * 86400)])
    assert len(store.messages((101,), query="كرسي")) == 5
    assert not store.messages((101,), query="old")
    assert not store.messages((101,), query="%' OR 1=1 --")
    page1 = store.messages((101,), limit=2)
    page2 = store.messages((101,), limit=2, before=page1[-1]["seq"])
    assert not {x["id"] for x in page1} & {x["id"] for x in page2}
    assert store.fetch("101:1", (202,)) is None
    seed(store, replace(config, allowed_chat_ids=(202,)), [])
    assert not store.messages((101,))


def test_send_idempotency_and_recipient_guard(config, store):
    calls = []
    def handle(req):
        calls.append(json.loads(req.content))
        return httpx.Response(200, json={"ok": True, "result": message(text="test", mid=44)})
    async def run():
        api = TelegramAPI(DUMMY_TOKEN, httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        bridge = Bridge(config, store, api)
        key = str(uuid.uuid4())
        first = await bridge.send(101, "test", key)
        second = await bridge.send(101, "test", key)
        assert first == second and first["status"] == "sent"
        with pytest.raises(ValueError):
            await bridge.send(101, "changed", key)
        with pytest.raises(ValueError):
            await bridge.send(666, "test", str(uuid.uuid4()))
        assert len(calls) == 1
        assert "parse_mode" not in calls[0]
        await api.close()
    asyncio.run(run())


def test_uncertain_send_never_retried(config, store):
    calls = []
    def handle(req):
        calls.append(1)
        raise httpx.ReadTimeout("simulated secret URL " + DUMMY_TOKEN, request=req)
    async def run():
        api = TelegramAPI(DUMMY_TOKEN, httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        bridge = Bridge(config, store, api)
        key = str(uuid.uuid4())
        first = await bridge.send(101, "test", key)
        assert first["status"] == "unknown" and DUMMY_TOKEN not in str(first)
        assert await bridge.send(101, "test", key) == first
        assert len(calls) == 1
        await api.close()
    asyncio.run(run())


def test_disabled_sends_and_utf16_limit(config, store):
    async def run():
        api = TelegramAPI(DUMMY_TOKEN)
        with pytest.raises(ValueError):
            await Bridge(replace(config, allow_send=False), store, api).send(101, "test", str(uuid.uuid4()))
        with pytest.raises(ValueError):
            await Bridge(config, store, api).send(101, "🙂" * 2049, str(uuid.uuid4()))
        await api.close()
    asyncio.run(run())


def test_poll_resumes_committed_offset(config, store):
    offsets = []
    def handle(req):
        params = json.loads(req.content)
        offsets.append(params["offset"])
        return httpx.Response(200, json={"ok": True, "result": [update(10)] if len(offsets) == 1 else []})
    async def run():
        api = TelegramAPI(DUMMY_TOKEN, httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        bridge = Bridge(config, store, api)
        await bridge.poll_once()
        await bridge.poll_once()
        assert offsets == [0, 11]
        assert not store.acquire_lease("another-instance")
        await api.close()
    asyncio.run(run())


@pytest.mark.parametrize("code", [401, 403, 409, 429, 500])
def test_upstream_errors_are_sanitized(code):
    def handle(req):
        return httpx.Response(code, json={"ok": False, "error_code": code,
            "description": "sensitive " + DUMMY_TOKEN, "parameters": {"retry_after": 3}})
    async def run():
        api = TelegramAPI(DUMMY_TOKEN, httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        with pytest.raises(TelegramError) as error:
            await api.call("sendMessage", chat_id=101, text="test")
        assert DUMMY_TOKEN not in str(error.value)
        assert error.value.retry_after == 3
        assert error.value.ambiguous == (code >= 500)
        await api.close()
    asyncio.run(run())


def test_config_rejects_unsafe_values(config):
    for changes in ({"public_base_url": "http://public.example.com"},
                    {"allowed_chat_ids": ()}, {"allowed_chat_ids": (True,)},
                    {"allow_send": "false"}, {"public_base_url": "https://example.com/path"}):
        with pytest.raises(ValueError):
            replace(config, **changes)
    assert DUMMY_TOKEN not in repr(config)
