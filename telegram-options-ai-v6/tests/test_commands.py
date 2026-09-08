import asyncio
import json
import time
from dataclasses import replace

import httpx
import pytest

from telegram_bridge.commands import parse_update
from telegram_bridge.demo import demo_series
from telegram_bridge.prediction import evaluate
from telegram_bridge.telegram import Bridge, TelegramAPI
from tests.test_bridge import DUMMY_TOKEN, config, store, update, message


def command_update(uid=10, text="/predict AAPL 5", **kwargs):
    item = update(uid=uid, text=text, **kwargs)
    item["message"]["entities"] = [{"type": "bot_command", "offset": 0, "length": len(text.split()[0])}]
    return item


def enqueue(store, config, updates):
    commands = {u["update_id"]: c for u in updates if (c := parse_update(u, config))}
    store.ingest(updates, config.allowed_chat_ids, config.expected_bot_username,
                 config.retention_days, config.max_messages, commands)


def test_only_fresh_explicit_private_owner_commands(config):
    config = replace(config, prediction_enabled=True)
    good = command_update()
    assert parse_update(good, config)["symbol"] == "AAPL"
    assert parse_update(command_update(text="/predict@Lil_fahad_bot MSFT 20"), config)["horizon"] == 20
    assert "error" in parse_update(command_update(text="/predict AAPL 2"), config)
    assert parse_update(command_update(text="/predict@SomeoneElseBot AAPL 5"), config) is None
    assert parse_update(command_update(chat_id=666), config) is None
    assert parse_update(command_update(date=int(time.time())-601), config) is None
    assert parse_update(update(text="/predict AAPL 5"), config) is None
    assert parse_update(good, replace(config, prediction_enabled=False)) is None
    for patch in ({"forward_origin": {}}, {"via_bot": {}}, {"from": {"id": 999}},
                  {"chat": {"id": 101, "type": "group"}}, {"from": {"id": 101, "is_bot": True}}):
        assert parse_update({**good, "message": {**good["message"], **patch}}, config) is None
    assert parse_update({"update_id": 10, "edited_message": good["message"]}, config) is None


def test_commands_cursor_atomic_and_rate_bounded(config, store):
    config = replace(config, prediction_enabled=True)
    enqueue(store, config, [command_update()])
    enqueue(store, config, [command_update()])
    assert store.db.execute("SELECT count(*) FROM prediction_jobs").fetchone()[0] == 1
    with pytest.raises(KeyError):
        enqueue(store, config, [command_update(uid=11), {"update_id": 12, "message": {"chat": {"id": 101}}}])
    assert store.get("offset") == "11"
    assert store.db.execute("SELECT count(*) FROM prediction_jobs").fetchone()[0] == 1
    enqueue(store, config, [command_update(uid=i, mid=i) for i in range(11, 21)])
    assert store.db.execute("SELECT count(*) FROM prediction_jobs").fetchone()[0] == 5
    assert store.get("offset") == "21"


def test_real_command_worker_to_mocked_telegram_prediction(config, store):
    config = replace(config, prediction_enabled=True, allow_send=False)
    report, calls = evaluate(demo_series(), 5), []
    class TestService:
        async def predict(self, symbol, horizon):
            assert (symbol, horizon) == ("AAPL", 5)
            return {**report, "symbol": symbol}
    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True, "result": message(mid=50, text=calls[-1]["text"])})
    async def run():
        api = TelegramAPI(DUMMY_TOKEN, httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        bridge = Bridge(config, store, api)
        bridge.commands.service = TestService()
        enqueue(store, config, [command_update()])
        assert await bridge.commands.once()
        assert not await bridge.commands.once()
        assert len(calls) == 1 and "احتمال الصعود" in calls[0]["text"]
        assert calls[0]["reply_parameters"]["message_id"] == 1
        assert calls[0]["reply_parameters"]["allow_sending_without_reply"] is False
        assert not config.allow_send and "telegram:send" not in config.scopes
        # Simulate a crash after send, before the job completion transaction: the frozen response is reused.
        with store.transaction() as db:
            db.execute("UPDATE prediction_jobs SET status='pending'")
        assert await bridge.commands.once()
        assert len(calls) == 1
        await api.close()
    asyncio.run(run())


def test_command_uncertain_delivery_and_expiry(config, store):
    config = replace(config, prediction_enabled=True, allow_send=False)
    calls = []
    def handle(request):
        calls.append(1)
        raise httpx.ReadTimeout("simulated failure", request=request)
    async def run():
        api = TelegramAPI(DUMMY_TOKEN, httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        bridge = Bridge(config, store, api)
        enqueue(store, config, [command_update(text="/help")])
        assert await bridge.commands.once()
        delivery = json.loads(store.db.execute("SELECT delivery FROM prediction_jobs").fetchone()[0])
        assert delivery["status"] == "unknown"
        with store.transaction() as db:
            db.execute("UPDATE prediction_jobs SET status='pending'")
        await bridge.commands.once()
        assert len(calls) == 1
        enqueue(store, config, [command_update(uid=11, text="/status")])
        with store.transaction() as db:
            db.execute("UPDATE prediction_jobs SET created=? WHERE update_id=11", (int(time.time())-601,))
        assert not await bridge.commands.once()
        assert len(calls) == 1
        await api.close()
    asyncio.run(run())


def test_poll_enqueues_command_before_next_telegram_offset(config, store):
    config = replace(config, prediction_enabled=True)
    requested_offsets = []
    def handle(request):
        payload = json.loads(request.content)
        requested_offsets.append(payload["offset"])
        return httpx.Response(200, json={"ok": True, "result": [command_update()] if payload["offset"] == 0 else []})
    async def run():
        api = TelegramAPI(DUMMY_TOKEN, httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        bridge = Bridge(config, store, api)
        await bridge.poll_once()
        assert store.next_prediction_job()["request"]["symbol"] == "AAPL"
        await bridge.poll_once()
        assert requested_offsets == [0, 11]
        await api.close()
    asyncio.run(run())
