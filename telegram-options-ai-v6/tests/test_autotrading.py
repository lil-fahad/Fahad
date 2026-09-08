import asyncio
import json
import uuid
from dataclasses import replace
from datetime import datetime, timezone

import httpx
import pytest

from telegram_bridge.config import Config
from telegram_bridge.trading import AlpacaAPI, TradeError, TradingService
from tests.test_bridge import config, store
from tests.test_trading import ACCOUNT_ID, PaperBroker


def auto_config(config):
    return replace(
        config,
        prediction_enabled=True,
        market_provider="alpaca",
        trading_enabled=True,
        trading_mode="paper",
        trading_chat_id=101,
        trading_account_id=ACCOUNT_ID,
        alpaca_key_id="test-key-only",
        alpaca_secret_key="test-secret-only",
        allow_send=False,
        auto_enabled=True,
        auto_symbols=("AAPL",),
        auto_order_usd="200",
        auto_max_position_usd="500",
    )


def approved_report(symbol="AAPL", as_of="2026-09-04", direction="up"):
    up = direction == "up"
    return {
        "symbol": symbol,
        "horizon_sessions": 5,
        "source": "Alpaca IEX daily bars",
        "basis": "split_adjusted",
        "as_of": as_of,
        "synthetic": False,
        "stale": False,
        "signal": direction,
        "probability_up": 0.72 if up else 0.28,
        "forecast_return_pct": 3.5 if up else -3.5,
        "backtest": {
            "n": 40,
            "brier": 0.18,
            "baseline_brier": 0.25,
            "mae_pct": 2.0,
            "baseline_mae_pct": 3.0,
            "interval_coverage": 0.8,
        },
    }


def test_automatic_config_is_strict_and_paper_only(config):
    cfg = auto_config(config)
    assert cfg.auto_symbols == ("AAPL",)
    assert cfg.auto_enabled and cfg.trading_mode == "paper"
    with pytest.raises(ValueError, match="paper"):
        replace(cfg, trading_mode="live")
    with pytest.raises(ValueError):
        replace(cfg, auto_buy_probability="NaN")
    with pytest.raises(ValueError):
        replace(cfg, auto_sell_probability="0.8")
    with pytest.raises(ValueError):
        replace(cfg, auto_symbols=())


def test_forecast_policy_requires_verified_current_evidence(config):
    from telegram_bridge.autotrading import decision_from_report

    cfg = auto_config(config)
    assert decision_from_report(cfg, approved_report(), "AAPL", "2026-09-04")[0] == "buy"
    assert decision_from_report(cfg, approved_report(direction="down"), "AAPL", "2026-09-04")[0] == "sell"
    for change in (
        {"stale": True},
        {"synthetic": True},
        {"basis": "unverified"},
        {"as_of": "2026-09-03"},
        {"source": "other"},
        {"signal": "abstain"},
    ):
        report = {**approved_report(), **change}
        action, reason = decision_from_report(cfg, report, "AAPL", "2026-09-04")
        assert action == "hold" and reason


class AutoPaperBroker(PaperBroker):
    def __init__(self):
        super().__init__()
        self.positions = []
        self.calendar_date = "2026-09-04"
        self.quote_symbol = "AAPL"
        self.clock_timestamp = None
        self.clock_open = False

    def handle(self, request):
        if request.method == "GET" and request.url.path == "/v2/clock" and self.clock_timestamp:
            return httpx.Response(200, json={"is_open": self.clock_open, "timestamp": self.clock_timestamp})
        if request.method == "GET" and request.url.path == "/v2/calendar":
            return httpx.Response(200, json=[{"date": self.calendar_date, "open": "09:30", "close": "16:00"}])
        if request.method == "GET" and request.url.path.endswith("/quotes/latest"):
            return httpx.Response(200, json={"symbol": self.quote_symbol, "quote": {
                "bp": 199, "ap": 200, "t": datetime.now(timezone.utc).isoformat()}})
        if request.method == "GET" and request.url.path.startswith("/v2/assets/"):
            symbol = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json={"symbol": symbol, "class": "us_equity", "status": "active", "tradable": True})
        return super().handle(request)


class FakePredictions:
    def __init__(self, broker):
        self.broker = broker
        self.direction = "up"

    async def predict(self, symbol, horizon):
        return approved_report(symbol, self.broker.calendar_date, self.direction)


class FakeBridge:
    def __init__(self, cfg, store, trading, predictions):
        self.config, self.store, self.trading, self.predictions = cfg, store, trading, predictions
        self.owner = "test-owner"
        self.last_poll_ok = __import__("time").time()
        self.poll_error = None
        self.stop = asyncio.Event()
        self.notices = []
        self.store.acquire_lease(self.owner)

    async def send(self, chat_id, text, request_id, **kwargs):
        self.notices.append((chat_id, text, request_id, kwargs))
        return {"status": "sent", "request_id": request_id}


def service(cfg, store, broker):
    api = AlpacaAPI(cfg.trading_mode, cfg.alpaca_key_id, cfg.alpaca_secret_key,
                    httpx.AsyncClient(transport=httpx.MockTransport(broker.handle)))
    return TradingService(cfg, store, api)


def test_automatic_buy_fill_then_sell_owned_shares(config, store):
    from telegram_bridge.autotrading import AutoTrader

    cfg = auto_config(config)
    broker = AutoPaperBroker()

    async def run():
        trading = service(cfg, store, broker)
        bridge = FakeBridge(cfg, store, trading, FakePredictions(broker))
        auto = AutoTrader(bridge)
        auto.set_running(101, True)

        broker.calendar_date = "2026-09-03"
        first = await auto.once()
        assert first["action"] == "buy"
        assert len(broker.submissions) == 1 and broker.submissions[0]["side"] == "buy"
        ticket = store.db.execute("SELECT ticket FROM trades").fetchone()[0]
        remote = next(iter(broker.orders.values()))
        remote.update(status="filled", filled_qty="1", filled_avg_price="200")
        broker.positions = [{"symbol": "AAPL", "side": "long", "qty": "1", "qty_available": "1",
                             "market_value": "200", "unrealized_pl": "0"}]
        await trading.reconcile(ticket)

        broker.calendar_date = "2026-09-04"
        bridge.predictions.direction = "down"
        second = await auto.once()
        assert second["action"] == "sell"
        assert len(broker.submissions) == 2 and broker.submissions[-1]["side"] == "sell"
        assert broker.submissions[-1]["qty"] == "1"

        # Restart does not submit the same session/symbol twice.
        restarted = AutoTrader(bridge)
        restarted.set_running(101, True)
        replay = await restarted.once()
        assert replay["action"] == "already_decided"
        assert len(broker.submissions) == 2
        await trading.api.close()

    asyncio.run(run())


def test_manual_confirm_cannot_submit_automatic_draft(config, store):
    from telegram_bridge.autotrading import auto_state_key, policy_fingerprint

    cfg = auto_config(config)
    broker = AutoPaperBroker()

    async def run():
        trading = service(cfg, store, broker)
        rid = str(uuid.uuid4())
        with store.transaction() as db:
            db.execute("""INSERT INTO auto_decisions
                (account_id,session_date,symbol,request_id,decision,state,reason,report,created)
                VALUES (?,?,?,?,?,'prepared','test','{}',?)""",
                (ACCOUNT_ID, "2026-09-04", "AAPL", rid, "buy", __import__('time').time()))
        draft = await trading.prepare(101, "buy", "AAPL", "1", "200", rid)
        with pytest.raises(TradeError, match="automatic"):
            await trading.confirm(101, draft["ticket"])
        store.set(auto_state_key(cfg), json.dumps({"running": True, "fingerprint": policy_fingerprint(cfg)}))
        placed = await trading.submit_automatic(draft["ticket"])
        assert placed["state"] == "submitted" and len(broker.submissions) == 1
        await trading.api.close()

    asyncio.run(run())


def test_auto_commands_are_owner_only_and_autooff_is_urgent(config, store):
    from telegram_bridge.commands import parse_update
    from tests.test_commands import command_update, enqueue

    cfg = auto_config(config)
    for verb in ("autoon", "autooff", "autostatus"):
        parsed = parse_update(command_update(text=f"/{verb}"), cfg)
        assert parsed and parsed["verb"] == verb
        assert parse_update(command_update(chat_id=999, text=f"/{verb}"), cfg) is None
    enqueue(store, cfg, [command_update(uid=i, mid=i, text="/help") for i in range(20, 25)])
    enqueue(store, cfg, [command_update(uid=25, mid=25, text="/autooff")])
    assert store.next_prediction_job()["request"]["verb"] == "autooff"


def test_automatic_notice_is_narrowly_allowed_when_generic_send_is_disabled(config, store):
    from telegram_bridge.telegram import Bridge, TelegramAPI
    from tests.test_bridge import DUMMY_TOKEN, message

    cfg = auto_config(config)
    calls = []
    def handle(req):
        calls.append(json.loads(req.content))
        return httpx.Response(200, json={"ok": True, "result": message(mid=99, text=calls[-1]["text"])})

    async def run():
        api = TelegramAPI(DUMMY_TOKEN, httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        bridge = Bridge(cfg, store, api)
        key = str(uuid.uuid4())
        with pytest.raises(ValueError):
            await bridge.send(101, "not allowed", key)
        sent = await bridge.send(101, "auto paper notice", key, _auto_notice=True)
        assert sent["status"] == "sent" and len(calls) == 1
        with pytest.raises(ValueError):
            await bridge.send(999, "auto paper notice", str(uuid.uuid4()), _auto_notice=True)
        await api.close()
        await bridge.trading.api.close()

    asyncio.run(run())


def test_configure_autotrading_persists_policy_without_network(config, tmp_path):
    from telegram_bridge.config import load_config, write_private_json
    from tests.test_bridge import DUMMY_TOKEN
    from telegram_bridge import cli

    path = tmp_path / "private" / "config.json"
    write_private_json(path, {
        "bot_token": DUMMY_TOKEN,
        "allowed_chat_ids": [101],
        "password_hash": config.password_hash,
        "data_dir": "state",
        "prediction_enabled": True,
        "market_provider": "alpaca",
        "trading_enabled": True,
        "trading_mode": "paper",
        "trading_chat_id": 101,
        "trading_account_id": ACCOUNT_ID,
        "alpaca_key_id": "test-key-only",
        "alpaca_secret_key": "test-secret-only",
    })
    cli.configure_autotrading(path, "AAPL,MSFT", "200", "500", 5, 300, "0.60", "0.40", True)
    cfg = load_config(path)
    assert cfg.auto_enabled and cfg.auto_symbols == ("AAPL", "MSFT")
    assert cfg.auto_order_usd == "200" and cfg.auto_notify is True
    from telegram_bridge.autotrading import auto_state_key
    from telegram_bridge.storage import Store
    state_store = Store(cfg.data_dir / "bridge.sqlite3")
    try:
        state = json.loads(state_store.get(auto_state_key(cfg), "{}"))
    finally:
        state_store.close()
    assert state["running"] is False


def test_second_symbol_is_processed_after_first_was_already_decided(config, store):
    from telegram_bridge.autotrading import AutoTrader, policy_fingerprint

    cfg = replace(auto_config(config), auto_symbols=("AAPL", "MSFT"))
    broker = AutoPaperBroker()
    class TwoPredictions(FakePredictions):
        async def predict(self, symbol, horizon):
            report = approved_report(symbol, self.broker.calendar_date, "up")
            if symbol == "MSFT":
                report["signal"] = "abstain"
            return report

    async def run():
        trading = service(cfg, store, broker)
        bridge = FakeBridge(cfg, store, trading, TwoPredictions(broker))
        auto = AutoTrader(bridge)
        auto.set_running(101, True)
        rid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"auto:{ACCOUNT_ID}:{policy_fingerprint(cfg)}:{broker.calendar_date}:AAPL"))
        with store.transaction() as db:
            db.execute("""INSERT INTO auto_decisions
                (account_id,session_date,symbol,request_id,decision,state,reason,report,created)
                VALUES (?,?,?,?,?,'hold','already','{}',?)""",
                (ACCOUNT_ID, broker.calendar_date, "AAPL", rid, "hold", __import__('time').time()))
        result = await auto.once()
        assert result["symbol"] == "MSFT" and result["action"] == "hold"
        await trading.api.close()
    asyncio.run(run())


def test_automatic_worker_requires_current_poller_lease(config, store):
    from telegram_bridge.autotrading import AutoTrader

    cfg = auto_config(config)
    broker = AutoPaperBroker()
    async def run():
        trading = service(cfg, store, broker)
        bridge = FakeBridge(cfg, store, trading, FakePredictions(broker))
        store.release_lease(bridge.owner)
        auto = AutoTrader(bridge)
        auto.set_running(101, True)
        result = await auto.once()
        assert result["action"] == "unhealthy_receiver"
        assert not broker.submissions
        await trading.api.close()
    asyncio.run(run())


def test_market_calendar_never_treats_open_current_session_as_completed(config, store):
    from telegram_bridge.autotrading import AutoTrader

    cfg = auto_config(config)
    broker = AutoPaperBroker()
    broker.calendar_date = "2026-09-08"
    def handle(request):
        if request.method == "GET" and request.url.path == "/v2/calendar":
            return httpx.Response(200, json=[
                {"date": "2026-09-04", "open": "09:30", "close": "16:00"},
                {"date": "2026-09-08", "open": "09:30", "close": "16:00"},
            ])
        if request.method == "GET" and request.url.path == "/v2/clock":
            return httpx.Response(200, json={"is_open": True, "timestamp": "2026-09-08T10:00:00-04:00"})
        return broker.handle(request)

    async def run():
        api = AlpacaAPI(cfg.trading_mode, cfg.alpaca_key_id, cfg.alpaca_secret_key,
                        httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        trading = TradingService(cfg, store, api)
        bridge = FakeBridge(cfg, store, trading, FakePredictions(broker))
        auto = AutoTrader(bridge)
        assert await auto._completed_session() == "2026-09-04"
        await trading.api.close()
    asyncio.run(run())


def test_unresolved_automatic_order_blocks_new_session_without_consuming_decision(config, store):
    from telegram_bridge.autotrading import AutoTrader, policy_fingerprint

    cfg = auto_config(config)
    broker = AutoPaperBroker()
    broker.post_error = "timeout_before_accept"

    async def run():
        trading = service(cfg, store, broker)
        bridge = FakeBridge(cfg, store, trading, FakePredictions(broker))
        auto = AutoTrader(bridge)
        auto.set_running(101, True)
        previous = "2026-09-04"
        rid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"auto:{ACCOUNT_ID}:{policy_fingerprint(cfg)}:{previous}:AAPL"))
        with store.transaction() as db:
            db.execute("""INSERT INTO auto_decisions
                (account_id,session_date,symbol,request_id,decision,state,reason,report,created)
                VALUES (?,?,?,?,?,'prepared','test','{}',?)""",
                (ACCOUNT_ID, previous, "AAPL", rid, "buy", __import__('time').time()))
        draft = await trading.prepare(101, "buy", "AAPL", "1", "200", rid)
        result = await trading.submit_automatic(draft["ticket"])
        assert result["state"] == "unknown"
        broker.post_error = None
        broker.lookup_missing = True
        broker.calendar_date = "2026-09-08"
        broker.clock_timestamp = "2026-09-08T17:00:00-04:00"
        outcome = await auto.once()
        assert outcome["action"] == "blocked_unresolved"
        assert store.db.execute("SELECT 1 FROM auto_decisions WHERE session_date='2026-09-08'").fetchone() is None
        assert len(broker.submissions) == 1
        await trading.api.close()
    asyncio.run(run())


def test_automatic_submission_rechecks_runtime_guard_after_preflight(config, store):
    from telegram_bridge.autotrading import auto_state_key, policy_fingerprint

    cfg = auto_config(config)
    broker = AutoPaperBroker()

    async def run():
        trading = service(cfg, store, broker)
        rid = str(uuid.uuid4())
        with store.transaction() as db:
            db.execute("""INSERT INTO auto_decisions
                (account_id,session_date,symbol,request_id,decision,state,reason,report,created)
                VALUES (?,?,?,?,?,'prepared','test','{}',?)""",
                (ACCOUNT_ID, "2026-09-04", "AAPL", rid, "buy", __import__('time').time()))
        draft = await trading.prepare(101, "buy", "AAPL", "1", "200", rid)
        bridge = FakeBridge(cfg, store, trading, FakePredictions(broker))
        from telegram_bridge.autotrading import AutoTrader
        auto = AutoTrader(bridge)
        auto.set_running(101, True)
        original = trading.preflight

        async def stop_during_preflight(payload):
            checks = await original(payload)
            store.release_lease(bridge.owner)
            return checks

        trading.preflight = stop_during_preflight
        with pytest.raises(TradeError, match="stopped|unhealthy"):
            await trading.submit_automatic(draft["ticket"])
        assert broker.submissions == []
        await trading.api.close()

    asyncio.run(run())


def test_worker_advances_to_next_symbol_without_full_interval(config, store):
    from telegram_bridge.autotrading import AutoTrader

    cfg = replace(auto_config(config), auto_symbols=("AAPL", "MSFT"), auto_interval_seconds=60)
    broker = AutoPaperBroker()

    async def run():
        trading = service(cfg, store, broker)
        bridge = FakeBridge(cfg, store, trading, FakePredictions(broker))
        auto = AutoTrader(bridge)
        calls = []

        async def fake_once():
            calls.append(len(calls) + 1)
            if len(calls) >= 2:
                bridge.stop.set()
            return {"action": "hold" if len(calls) == 1 else "already_decided"}

        auto.once = fake_once
        await asyncio.wait_for(auto.run(), timeout=0.5)
        assert calls == [1, 2]
        await trading.api.close()

    asyncio.run(run())


def test_autooff_race_is_rechecked_atomically_when_order_is_claimed(config, store):
    from telegram_bridge.autotrading import auto_state_key, policy_fingerprint

    cfg = auto_config(config)
    broker = AutoPaperBroker()

    async def run():
        trading = service(cfg, store, broker)
        rid = str(uuid.uuid4())
        with store.transaction() as db:
            db.execute("""INSERT INTO auto_decisions
                (account_id,session_date,symbol,request_id,decision,state,reason,report,created)
                VALUES (?,?,?,?,?,'prepared','test','{}',?)""",
                (ACCOUNT_ID, "2026-09-04", "AAPL", rid, "buy", __import__('time').time()))
        draft = await trading.prepare(101, "buy", "AAPL", "1", "200", rid)
        fingerprint = policy_fingerprint(cfg)
        store.set(auto_state_key(cfg), json.dumps({"running": True, "fingerprint": fingerprint}))
        calls = {"n": 0}

        def guard():
            calls["n"] += 1
            if calls["n"] == 2:
                store.set(auto_state_key(cfg), json.dumps({"running": False, "fingerprint": fingerprint}))
            return True

        trading.set_automatic_guard(guard)
        with pytest.raises(TradeError, match="stopped"):
            await trading.submit_automatic(draft["ticket"])
        assert broker.submissions == []
        await trading.api.close()

    asyncio.run(run())


def test_poller_lease_race_is_rechecked_when_automatic_order_is_claimed(config, store):
    from telegram_bridge.autotrading import auto_state_key, policy_fingerprint

    cfg = auto_config(config)
    broker = AutoPaperBroker()

    async def run():
        trading = service(cfg, store, broker)
        rid = str(uuid.uuid4())
        with store.transaction() as db:
            db.execute("""INSERT INTO auto_decisions
                (account_id,session_date,symbol,request_id,decision,state,reason,report,created)
                VALUES (?,?,?,?,?,'prepared','test','{}',?)""",
                (ACCOUNT_ID, "2026-09-04", "AAPL", rid, "buy", __import__('time').time()))
        draft = await trading.prepare(101, "buy", "AAPL", "1", "200", rid)
        fingerprint = policy_fingerprint(cfg)
        store.set(auto_state_key(cfg), json.dumps({"running": True, "fingerprint": fingerprint}))
        owner = "atomic-owner"
        assert store.acquire_lease(owner)
        calls = {"n": 0}

        def guard():
            calls["n"] += 1
            if calls["n"] == 2:
                store.release_lease(owner)
            return True

        trading.set_automatic_guard(guard, owner=owner)
        with pytest.raises(TradeError, match="lease|receiver|stopped"):
            await trading.submit_automatic(draft["ticket"])
        assert broker.submissions == []
        await trading.api.close()

    asyncio.run(run())
