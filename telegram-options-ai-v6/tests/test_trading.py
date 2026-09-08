import asyncio
import copy
import json
import time
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from telegram_bridge.commands import parse_update
from telegram_bridge.market import MarketData
from telegram_bridge.trading import AlpacaAPI, BrokerError, TradeError, TradingService, order_values, render_order
from tests.test_bridge import config, store
from tests.test_commands import command_update, enqueue

ACCOUNT_ID = "11111111-1111-4111-8111-111111111111"


@pytest.fixture
def trading_config(config):
    return replace(config, trading_enabled=True, trading_chat_id=101, trading_account_id=ACCOUNT_ID,
                   alpaca_key_id="test-key-only", alpaca_secret_key="test-secret-only", allow_send=False)


class PaperBroker:
    """Stateful broker contract simulator; never reaches a network endpoint."""
    def __init__(self):
        self.account = {"id": ACCOUNT_ID, "status": "ACTIVE", "currency": "USD", "trading_blocked": False,
                        "account_blocked": False, "trade_suspended_by_user": False,
                        "cash": "10000", "buying_power": "20000", "equity": "10000"}
        self.positions = [{"symbol": "AAPL", "side": "long", "qty": "10", "qty_available": "10",
                           "market_value": "2000", "unrealized_pl": "0"}]
        self.asset = {"symbol": "AAPL", "class": "us_equity", "status": "active", "tradable": True}
        self.open = True
        self.quote_age = 0
        self.orders = {}
        self.requests = []
        self.submissions = []
        self.cancellations = []
        self.post_error = None
        self.lookup_missing = False
        self.cancel_error = None

    def handle(self, request):
        self.requests.append(request)
        path = request.url.path
        now = datetime.now(timezone.utc).isoformat()
        if request.method == "POST":
            assert path == "/v2/orders"
            body = json.loads(request.content)
            self.submissions.append(body)
            if isinstance(self.post_error, int):
                return httpx.Response(self.post_error, json={"message": "test-secret-only"})
            if self.post_error == "timeout_before_accept":
                raise httpx.ReadTimeout("simulated unknown delivery", request=request)
            order = {**body, "id": str(uuid.uuid4()), "status": "new", "filled_qty": "0",
                     "filled_avg_price": None, "submitted_at": now}
            self.orders[body["client_order_id"]] = order
            if self.post_error == "timeout_after_accept":
                raise httpx.ReadTimeout("simulated lost receipt", request=request)
            if self.post_error == "wrong_receipt":
                return httpx.Response(200, json={**order, "symbol": "WRONG"})
            return httpx.Response(200, json=order)
        if request.method == "DELETE":
            self.cancellations.append(path)
            if self.cancel_error:
                return httpx.Response(self.cancel_error, json={"message": "test-secret-only"})
            for order in self.orders.values():
                if path.endswith(order["id"]):
                    order["status"] = "pending_cancel"
                    return httpx.Response(204)
            return httpx.Response(404)
        assert request.method == "GET"
        if path == "/v2/account":
            data = self.account
        elif path == "/v2/clock":
            data = {"is_open": self.open, "timestamp": now}
        elif path.startswith("/v2/assets/"):
            data = self.asset
        elif path == "/v2/positions":
            data = self.positions
        elif path == "/v2/orders":
            data = [o for o in self.orders.values() if o["status"] not in {"filled", "canceled", "rejected", "expired"}]
        elif path == "/v2/orders:by_client_order_id":
            data = self.orders.get(request.url.params["client_order_id"])
            if self.lookup_missing or data is None:
                return httpx.Response(404, json={"message": "not found"})
        elif path.endswith("/quotes/latest"):
            assert request.url.host == "data.alpaca.markets" and request.url.params["feed"] == "iex"
            data = {"symbol": "AAPL", "quote": {"bp": 199, "ap": 200,
                "t": (datetime.now(timezone.utc) - timedelta(seconds=self.quote_age)).isoformat()}}
        else:
            raise AssertionError("Unexpected broker endpoint")
        return httpx.Response(200, json=data)


def service(config, store, broker):
    api = AlpacaAPI(config.trading_mode, config.alpaca_key_id, config.alpaca_secret_key,
                    httpx.AsyncClient(transport=httpx.MockTransport(broker.handle)))
    return TradingService(config, store, api)


def test_preview_buy_confirm_fill_and_sell(trading_config, store):
    broker = PaperBroker()
    async def run():
        trader = service(trading_config, store, broker)
        draft = await trader.prepare(101, "buy", "AAPL", "1", "200", str(uuid.uuid4()))
        assert draft["state"] == "draft" and not broker.submissions
        assert "لم يُرسل إلى الوسيط" in render_order(draft)
        placed = await trader.confirm(101, draft["ticket"])
        assert placed["state"] == "submitted" and placed["result"]["status"] == "new"
        assert "منفذ بالكامل" not in render_order(placed)
        assert len(broker.submissions) == 1
        assert broker.submissions[0]["time_in_force"] == "day" and broker.submissions[0]["extended_hours"] is False
        remote = next(iter(broker.orders.values()))
        remote.update(status="partially_filled", filled_qty="0.5", filled_avg_price="199.5")
        partial = await trader.reconcile(draft["ticket"], 101)
        assert "منفذ جزئيًا" in render_order(partial)
        remote.update(status="filled", filled_qty="1")
        filled = await trader.reconcile(draft["ticket"], 101)
        assert "منفذ بالكامل" in render_order(filled)
        sell = await trader.prepare(101, "sell", "AAPL", "1", "199", str(uuid.uuid4()))
        await trader.confirm(101, sell["ticket"])
        assert broker.submissions[-1]["side"] == "sell"
        await trader.api.close()
    asyncio.run(run())


def test_replayed_and_concurrent_confirmation_only_posts_once(trading_config, store):
    broker = PaperBroker()
    async def run():
        trader = service(trading_config, store, broker)
        rid = str(uuid.uuid4())
        draft = await trader.prepare(101, "buy", "AAPL", "1", "200", rid)
        assert (await trader.prepare(101, "buy", "AAPL", "1", "200", rid))["ticket"] == draft["ticket"]
        with pytest.raises(TradeError):
            await trader.prepare(101, "buy", "AAPL", "2", "200", rid)
        await asyncio.gather(trader.confirm(101, draft["ticket"]), trader.confirm(101, draft["ticket"]))
        # A new service instance reuses the durable ledger after a process restart.
        restarted = service(trading_config, store, broker)
        await restarted.confirm(101, draft["ticket"])
        assert len(broker.submissions) == 1
        await trader.api.close()
        await restarted.api.close()
    asyncio.run(run())


@pytest.mark.parametrize("failure,expected", [("timeout_after_accept", "submitted"), ("timeout_before_accept", "unknown"), ("wrong_receipt", "submitted"), (500, "unknown"), (422, "rejected")])
def test_submission_failures_reconcile_never_repost(trading_config, store, failure, expected):
    broker = PaperBroker()
    broker.post_error = failure
    async def run():
        trader = service(trading_config, store, broker)
        draft = await trader.prepare(101, "buy", "AAPL", "1", "200", str(uuid.uuid4()))
        result = await trader.confirm(101, draft["ticket"])
        assert result["state"] == expected
        again = await trader.confirm(101, draft["ticket"])
        assert again["state"] == expected and len(broker.submissions) == 1
        assert "test-secret-only" not in json.dumps(result)
        if expected == "unknown":
            another = await trader.prepare(101, "buy", "AAPL", "1", "200", str(uuid.uuid4()))
            with pytest.raises(TradeError, match="غير محسوم"):
                await trader.confirm(101, another["ticket"])
        await trader.api.close()
    asyncio.run(run())


@pytest.mark.parametrize("guard", ["cash", "buying_power", "blocked", "account", "closed", "stale_quote", "asset", "price", "cap", "short"])
def test_preflight_guards_prevent_submission(trading_config, store, guard):
    broker = PaperBroker()
    side, price, qty = "buy", "200", "1"
    if guard == "cash": broker.account["cash"] = "10"
    elif guard == "buying_power": broker.account["buying_power"] = "10"
    elif guard == "blocked": broker.account["account_blocked"] = True
    elif guard == "account": broker.account["id"] = str(uuid.uuid4())
    elif guard == "closed": broker.open = False
    elif guard == "stale_quote": broker.quote_age = 61
    elif guard == "asset": broker.asset["class"] = "crypto"
    elif guard == "price": price = "250"
    elif guard == "cap": qty = "3"
    elif guard == "short":
        side = "sell"
        broker.positions[0]["qty_available"] = "0"
    async def run():
        trader = service(trading_config, store, broker)
        with pytest.raises(TradeError):
            await trader.prepare(101, side, "AAPL", qty, price, str(uuid.uuid4()))
        assert not broker.submissions
        await trader.api.close()
    asyncio.run(run())


def test_confirm_rechecks_cash_expiry_and_pause(trading_config, store):
    broker = PaperBroker()
    async def run():
        trader = service(trading_config, store, broker)
        draft = await trader.prepare(101, "buy", "AAPL", "1", "200", str(uuid.uuid4()))
        broker.account["cash"] = "0"
        with pytest.raises(TradeError, match="النقد"):
            await trader.confirm(101, draft["ticket"])
        broker.account["cash"] = "10000"
        trader.halt(101, True)
        with pytest.raises(TradeError, match="متوقف"):
            await trader.confirm(101, draft["ticket"])
        assert service(trading_config, store, broker).paused()
        trader.halt(101, False)
        with store.transaction() as db:
            db.execute("UPDATE trades SET expires=?", (time.time()-1,))
        with pytest.raises(TradeError, match="مهلة"):
            await trader.confirm(101, draft["ticket"])
        assert not broker.submissions
        await trader.api.close()
    asyncio.run(run())


def test_daily_limit_reservations_and_open_order_limit(trading_config, store):
    config = replace(trading_config, daily_buy_limit_usd="600")
    broker = PaperBroker()
    async def run():
        trader = service(config, store, broker)
        first = await trader.prepare(101, "buy", "AAPL", "2", "200", str(uuid.uuid4()))
        await trader.confirm(101, first["ticket"])
        second = await trader.prepare(101, "buy", "AAPL", "2", "200", str(uuid.uuid4()))
        with pytest.raises(TradeError, match="سقف"):
            await trader.confirm(101, second["ticket"])
        # Even with leveraged buying power, open orders reserve the unlevered cash budget.
        broker.account["cash"] = "500"
        with pytest.raises(TradeError, match="النقد"):
            await trader.prepare(101, "buy", "AAPL", "1", "200", str(uuid.uuid4()))
        broker.account["cash"] = "10000"
        trader.config = replace(config, max_open_orders=1)
        with pytest.raises(TradeError, match="الأوامر المفتوحة"):
            await trader.prepare(101, "buy", "AAPL", "1", "200", str(uuid.uuid4()))
        assert len(broker.submissions) == 1
        await trader.api.close()
    asyncio.run(run())


def test_cancel_is_requested_then_confirmed_and_never_reissued(trading_config, store):
    broker = PaperBroker()
    async def run():
        trader = service(trading_config, store, broker)
        draft = await trader.prepare(101, "buy", "AAPL", "1", "200", str(uuid.uuid4()))
        assert (await trader.cancel(101, draft["ticket"]))["state"] == "canceled"
        await trader.confirm(101, draft["ticket"])
        assert not broker.submissions
        live = await trader.prepare(101, "buy", "AAPL", "1", "200", str(uuid.uuid4()))
        await trader.confirm(101, live["ticket"])
        pending = await trader.cancel(101, live["ticket"])
        assert pending["cancel_state"] == "requested" and pending["result"]["status"] == "pending_cancel"
        assert "الإلغاء قيد المعالجة" in render_order(pending)
        await trader.cancel(101, live["ticket"])
        assert len(broker.cancellations) == 1
        next(iter(broker.orders.values()))["status"] = "canceled"
        final = await trader.reconcile(live["ticket"], 101)
        assert "ملغى" in render_order(final)
        await trader.api.close()
    asyncio.run(run())


def test_live_and_paper_use_explicit_different_hosts(trading_config, store):
    async def run():
        for mode, host in (("paper", "paper-api.alpaca.markets"), ("live", "api.alpaca.markets")):
            broker = PaperBroker()
            trader = service(replace(trading_config, trading_mode=mode), store, broker)
            draft = await trader.prepare(101, "buy", "AAPL", "1", "200", str(uuid.uuid4()))
            await trader.confirm(101, draft["ticket"])
            assert {r.url.host for r in broker.requests if r.url.path == "/v2/orders"} == {host}
            assert draft["mode"] == mode
            await trader.api.close()
    asyncio.run(run())


def test_trade_owner_and_command_parsing(trading_config, store):
    cfg = replace(trading_config, allowed_chat_ids=(101, 202))
    request = command_update(text="/buy AAPL 1 200")
    assert parse_update(request, cfg)["limit_price"] == "200"
    assert parse_update(command_update(text="/buy AAPL 1 200", chat_id=202), cfg) is None
    assert parse_update(command_update(text="/confirm abcdef012345", date=int(time.time())-121), cfg) is None
    assert "error" in parse_update(command_update(text="/buy AAPL 0.5 200"), cfg)
    for bad in ("NaN", "Infinity", "-1", "1e3", "200.001"):
        with pytest.raises(TradeError):
            order_values("buy", "AAPL", "1", bad)
    broker = PaperBroker()
    async def run():
        trader = service(cfg, store, broker)
        with pytest.raises(TradeError):
            await trader.prepare(202, "buy", "AAPL", "1", "200", str(uuid.uuid4()))
        assert not broker.requests
        await trader.api.close()
    asyncio.run(run())


def test_pause_admitted_and_prioritized_despite_normal_command_limit(trading_config, store):
    cfg = replace(trading_config, prediction_enabled=True)
    enqueue(store, cfg, [command_update(uid=i, mid=i, text="/help") for i in range(10, 15)])
    enqueue(store, cfg, [command_update(uid=15, mid=15, text="/pause")])
    assert store.next_prediction_job()["request"]["verb"] == "pause"


def test_alpaca_free_history_paginates_and_requests_split_only(tmp_path):
    calls = []
    def handle(request):
        calls.append(request)
        assert request.url.host == "data.alpaca.markets"
        assert request.url.params["feed"] == "iex" and request.url.params["adjustment"] == "split"
        assert request.headers["APCA-API-SECRET-KEY"] == "test-secret-only"
        if len(calls) == 1:
            return httpx.Response(200, json={"bars": {"AAPL": [{"t": "2026-08-03T04:00:00Z", "c": 200}]}, "next_page_token": "page2"})
        assert request.url.params["page_token"] == "page2"
        return httpx.Response(200, json={"bars": {"AAPL": [{"t": "2026-08-04T04:00:00Z", "c": 201}]}, "next_page_token": None})
    market = MarketData("alpaca", "", tmp_path, alpaca_key_id="test-key-only", alpaca_secret_key="test-secret-only", transport=httpx.MockTransport(handle))
    series = market.load("AAPL")
    assert len(series.bars) == 2 and series.basis == "split_adjusted" and "IEX" in series.source
    assert series.bars[0].date == "2026-08-03"
    assert "test-secret-only" not in repr(series)


def test_telegram_worker_trade_preview_confirmation_and_crash_replay(trading_config, store):
    from telegram_bridge.telegram import Bridge, TelegramAPI
    from tests.test_bridge import DUMMY_TOKEN, message
    broker, replies = PaperBroker(), []
    def telegram_response(request):
        payload = json.loads(request.content)
        replies.append(payload)
        return httpx.Response(200, json={"ok": True, "result": message(mid=100 + len(replies), text=payload["text"])})
    async def run():
        api = TelegramAPI(DUMMY_TOKEN, httpx.AsyncClient(transport=httpx.MockTransport(telegram_response)))
        bridge = Bridge(trading_config, store, api)
        await bridge.trading.api.close()
        bridge.trading = service(trading_config, store, broker)
        enqueue(store, trading_config, [command_update(uid=10, text="/buy AAPL 1 200")])
        assert await bridge.commands.once()
        ticket = store.db.execute("SELECT ticket FROM trades").fetchone()[0]
        assert not broker.submissions and f"/confirm {ticket}" in replies[0]["text"]
        # Rejected sender cannot authorize a broker request.
        enqueue(store, trading_config, [command_update(uid=11, mid=2, chat_id=999, text=f"/confirm {ticket}")])
        assert not await bridge.commands.once() and not broker.submissions
        enqueue(store, trading_config, [command_update(uid=12, mid=3, text=f"/confirm {ticket}")])
        assert await bridge.commands.once()
        assert len(broker.submissions) == 1 and "حالة الوسيط: مفتوح" in replies[-1]["text"]
        assert replies[-1]["reply_parameters"]["message_id"] == 3
        # Crash after broker POST and Telegram reply, before command completion.
        with store.transaction() as db:
            db.execute("UPDATE prediction_jobs SET status='pending',response=NULL WHERE update_id=12")
        assert await bridge.commands.once()
        assert len(broker.submissions) == 1 and len(replies) == 2
        # A lost read must show unknown, never the cached confirmed status.
        broker.lookup_missing = True
        enqueue(store, trading_config, [command_update(uid=13, mid=4, text="/orders")])
        assert await bridge.commands.once()
        assert "unknown" in replies[-1]["text"] and "| new" not in replies[-1]["text"]
        assert len(broker.submissions) == 1
        await api.close()
        await bridge.trading.api.close()
    asyncio.run(run())


def test_mcp_trade_scopes_only_read_or_prepare_never_execute(trading_config, store):
    from starlette.testclient import TestClient
    from telegram_bridge.server import create_app
    from telegram_bridge.telegram import TelegramAPI
    from tests.test_auth_mcp import authorize, rpc
    from tests.test_bridge import DUMMY_TOKEN
    broker = PaperBroker()
    def no_telegram(request):
        raise AssertionError("MCP trading tools must not send Telegram messages")
    api = TelegramAPI(DUMMY_TOKEN, httpx.AsyncClient(transport=httpx.MockTransport(no_telegram)))
    app = create_app(trading_config, store, api, start_poller=False)
    asyncio.run(app.state.bridge.trading.api.close())
    app.state.bridge.trading = service(trading_config, store, broker)
    with TestClient(app, base_url=trading_config.public_base_url, follow_redirects=False) as client:
        read, _ = authorize(client, trading_config, ["telegram:read", "trading:read"])
        token = read["access_token"]
        tool_list = rpc(client, token, "tools/list").json()["result"]["tools"]
        names = {t["name"] for t in tool_list}
        assert names == {"get_status", "list_chats", "search", "fetch", "list_messages", "predict_market",
                         "backtest_market", "get_trading_account", "list_trade_orders", "prepare_trade"}
        account = rpc(client, token, "tools/call", {"name": "get_trading_account", "arguments": {}})
        assert not account.json()["result"].get("isError") and "paper" in account.text
        args = {"side": "buy", "symbol": "AAPL", "quantity": "1", "limit_price": "200", "request_id": str(uuid.uuid4())}
        denied = rpc(client, token, "tools/call", {"name": "prepare_trade", "arguments": args})
        assert denied.json()["result"]["isError"] is True
        assert not store.db.execute("SELECT ticket FROM trades").fetchone()
        authorized, _ = authorize(client, trading_config, ["telegram:read", "trading:prepare"])
        prepared = rpc(client, authorized["access_token"], "tools/call", {"name": "prepare_trade", "arguments": args})
        assert not prepared.json()["result"].get("isError")
        draft = json.loads(prepared.json()["result"]["content"][0]["text"])
        assert draft["state"] == "draft" and draft["mode"] == "paper"
        assert not broker.submissions and not broker.cancellations
        assert trading_config.alpaca_secret_key not in account.text + prepared.text
    asyncio.run(api.close())
    asyncio.run(app.state.bridge.trading.api.close())


def test_inconsistent_fill_receipts_are_unknown(trading_config, store):
    broker = PaperBroker()
    async def run():
        trader = service(trading_config, store, broker)
        draft = await trader.prepare(101, "buy", "AAPL", "1", "200", str(uuid.uuid4()))
        await trader.confirm(101, draft["ticket"])
        remote = next(iter(broker.orders.values()))
        for changes in ({"status": "filled", "filled_qty": "0", "filled_avg_price": None},
                        {"status": "filled", "filled_qty": "1", "filled_avg_price": "NaN"},
                        {"status": "filled", "filled_qty": "1", "filled_avg_price": "-1"}):
            remote.update(changes)
            result = await trader.reconcile(draft["ticket"])
            assert result["state"] == "unknown" and "منفذ بالكامل" not in render_order(result)
        assert len(broker.submissions) == 1
        await trader.api.close()
    asyncio.run(run())


def test_local_trading_setup_verifies_account_without_submitting(config, tmp_path, monkeypatch, capsys):
    from telegram_bridge import cli
    from telegram_bridge.config import load_config, write_private_json
    from tests.test_bridge import DUMMY_TOKEN
    path = tmp_path / "private" / "config.json"
    write_private_json(path, {"bot_token": DUMMY_TOKEN, "allowed_chat_ids": [101],
                             "password_hash": config.password_hash, "allow_send": False, "data_dir": "state"})
    broker = PaperBroker()
    api = AlpacaAPI("paper", "local-key-only", "local-secret-only",
                    httpx.AsyncClient(transport=httpx.MockTransport(broker.handle)))
    entries = iter(["local-key-only", "local-secret-only"])
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: next(entries))
    monkeypatch.setattr(cli, "AlpacaAPI", lambda mode, key, secret: api)
    asyncio.run(cli.configure_trading(path, "paper", None, "500", "1000"))
    result = load_config(path)
    assert result.trading_enabled and result.trading_mode == "paper" and result.trading_chat_id == 101
    assert result.trading_account_id == ACCOUNT_ID and result.market_provider == "alpaca"
    assert result.prediction_enabled and not result.allow_send and result.password_hash == config.password_hash
    assert len(broker.requests) == 1 and broker.requests[0].method == "GET"
    assert not broker.submissions and path.stat().st_mode & 0o777 == 0o600
    assert "local-secret-only" not in capsys.readouterr().out + repr(result)
