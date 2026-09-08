"""Opt-in forecast-driven automatic trading for Alpaca paper accounts only."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
import uuid
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from zoneinfo import ZoneInfo

from .market import symbol_checked
from .trading import BrokerError, TradeError, number, timestamp


def policy_fingerprint(config) -> str:
    fields = {
        "account": config.trading_account_id,
        "chat": config.trading_chat_id,
        "symbols": list(config.auto_symbols),
        "horizon": config.auto_horizon,
        "interval": config.auto_interval_seconds,
        "order_usd": config.auto_order_usd,
        "max_position_usd": config.auto_max_position_usd,
        "buy_probability": config.auto_buy_probability,
        "sell_probability": config.auto_sell_probability,
        "notify": config.auto_notify,
    }
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def auto_state_key(config) -> str:
    return "autotrading:" + config.trading_account_id


def _finite(value) -> Decimal:
    result = number(value)
    if not result.is_finite():
        raise TradeError("Automatic forecast contains a non-finite value.")
    return result


def decision_from_report(config, report: dict, symbol: str, session_date: str) -> tuple[str, str]:
    """Return buy/sell/hold only for a verified report for the completed session."""
    try:
        if not isinstance(report, dict) or report.get("symbol") != symbol_checked(symbol):
            return "hold", "forecast symbol mismatch"
        if report.get("horizon_sessions") != config.auto_horizon:
            return "hold", "forecast horizon mismatch"
        if report.get("source") != "Alpaca IEX daily bars":
            return "hold", "forecast source is not Alpaca IEX daily bars"
        if report.get("basis") != "split_adjusted" or report.get("synthetic") is not False:
            return "hold", "forecast data is not verified split-adjusted market data"
        if report.get("stale") is not False or report.get("as_of") != session_date:
            return "hold", "forecast is not current for the latest completed session"
        if report.get("signal") not in {"up", "down"}:
            return "hold", "forecast abstained"
        bt = report.get("backtest")
        if not isinstance(bt, dict):
            return "hold", "forecast backtest is missing"
        n = int(bt.get("n", 0))
        brier, base_brier = _finite(bt.get("brier")), _finite(bt.get("baseline_brier"))
        mae, base_mae = _finite(bt.get("mae_pct")), _finite(bt.get("baseline_mae_pct"))
        coverage = _finite(bt.get("interval_coverage"))
        probability = _finite(report.get("probability_up"))
        forecast_return = _finite(report.get("forecast_return_pct"))
        if n < 40 or brier >= base_brier or mae >= base_mae or coverage < Decimal("0.70"):
            return "hold", "forecast validation thresholds were not met"
        if report["signal"] == "up" and probability >= Decimal(config.auto_buy_probability) and forecast_return > 0:
            return "buy", "validated upward forecast"
        if report["signal"] == "down" and probability <= Decimal(config.auto_sell_probability) and forecast_return < 0:
            return "sell", "validated downward forecast"
        return "hold", "forecast did not reach the configured action threshold"
    except (ValueError, TypeError, KeyError, ArithmeticError, TradeError):
        return "hold", "forecast report is malformed"


def _limit_price(value: Decimal, side: str) -> str:
    tick = Decimal("0.01") if value >= 1 else Decimal("0.0001")
    rounding = ROUND_CEILING if side == "buy" else ROUND_FLOOR
    return format(value.quantize(tick, rounding=rounding), "f")


class AutoTrader:
    def __init__(self, bridge):
        self.bridge = bridge
        self.config = bridge.config
        self.store = bridge.store
        self.lock = asyncio.Lock()
        self.last_error: str | None = None
        if hasattr(bridge.trading, "set_automatic_guard"):
            bridge.trading.set_automatic_guard(self.submission_allowed, owner=bridge.owner)

    def submission_allowed(self) -> bool:
        return (self.running() and not self.bridge.trading.paused()
                and self.bridge.poll_error is None and self.bridge.last_poll_ok is not None
                and time.time() - self.bridge.last_poll_ok <= 120
                and self.store.owns_lease(self.bridge.owner))

    def _enabled(self):
        cfg = self.config
        if not cfg.auto_enabled:
            raise TradeError("Automatic trading is not configured.")
        if cfg.trading_mode != "paper":
            raise TradeError("Automatic trading is paper-only.")
        self.bridge.trading.enabled(cfg.trading_chat_id)

    def _state(self) -> dict:
        try:
            state = json.loads(self.store.get(auto_state_key(self.config), "{}"))
            return state if isinstance(state, dict) else {}
        except json.JSONDecodeError:
            return {}

    def running(self) -> bool:
        state = self._state()
        return state.get("running") is True and state.get("fingerprint") == policy_fingerprint(self.config)

    def set_running(self, chat_id: int, running: bool) -> dict:
        self._enabled()
        if chat_id != self.config.trading_chat_id:
            raise TradeError("Only the configured trading owner can control automatic trading.")
        state = {"running": bool(running), "fingerprint": policy_fingerprint(self.config), "changed_at": time.time()}
        self.store.set(auto_state_key(self.config), json.dumps(state, sort_keys=True))
        return self.status()

    def status(self) -> dict:
        with self.store.lock:
            row = self.store.db.execute(
                "SELECT session_date,symbol,decision,state,reason,ticket,updated FROM auto_decisions "
                "WHERE account_id=? ORDER BY created DESC LIMIT 1", (self.config.trading_account_id,)).fetchone()
        latest = dict(row) if row else None
        return {
            "configured": self.config.auto_enabled,
            "paper_only": True,
            "running": self.running() if self.config.auto_enabled else False,
            "symbols": list(self.config.auto_symbols),
            "horizon_sessions": self.config.auto_horizon,
            "interval_seconds": self.config.auto_interval_seconds,
            "latest": latest,
            "last_error": self.last_error,
        }

    async def _completed_session(self) -> str:
        end = date.today()
        start = end - timedelta(days=14)
        rows, clock = await asyncio.gather(
            self.bridge.trading.api.call("GET", "/v2/calendar", params={"start": start.isoformat(), "end": end.isoformat()}),
            self.bridge.trading.api.call("GET", "/v2/clock"),
        )
        if not isinstance(rows, list) or not rows or not isinstance(clock, dict):
            raise TradeError("Broker calendar did not return a completed market session.")
        broker_now = timestamp(clock.get("timestamp")).astimezone(ZoneInfo("America/New_York"))
        valid = []
        for row in rows:
            try:
                day = date.fromisoformat(row.get("date", ""))
                close_text = row.get("close", "16:00")
                close_time = datetime.strptime(close_text, "%H:%M").time()
                close_at = datetime.combine(day, close_time, ZoneInfo("America/New_York"))
            except (ValueError, TypeError, AttributeError):
                continue
            if close_at <= broker_now:
                valid.append(day)
        if not valid:
            raise TradeError("Broker calendar did not return a valid completed market session.")
        return max(valid).isoformat()

    def _decision(self, session: str, symbol: str):
        with self.store.lock:
            row = self.store.db.execute("SELECT * FROM auto_decisions WHERE account_id=? AND session_date=? AND symbol=?",
                                        (self.config.trading_account_id, session, symbol)).fetchone()
        return dict(row) if row else None

    def _save_decision(self, session: str, symbol: str, request_id: str, action: str, state: str,
                       reason: str, report: dict, ticket: str | None = None, result: dict | None = None):
        now = time.time()
        with self.store.transaction() as db:
            db.execute("""INSERT INTO auto_decisions
                (account_id,session_date,symbol,request_id,decision,state,reason,report,created,ticket,result,updated)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(account_id,session_date,symbol) DO UPDATE SET
                state=excluded.state,reason=excluded.reason,ticket=COALESCE(excluded.ticket,auto_decisions.ticket),
                result=COALESCE(excluded.result,auto_decisions.result),updated=excluded.updated""",
                (self.config.trading_account_id, session, symbol, request_id, action, state, reason,
                 json.dumps(report, sort_keys=True, allow_nan=False), now, ticket,
                 json.dumps(result, sort_keys=True) if result is not None else None, now))

    def _owned_filled(self, symbol: str) -> Decimal:
        with self.store.lock:
            rows = self.store.db.execute("""SELECT t.payload,t.result FROM trades t
                JOIN auto_decisions d ON d.request_id=t.request_id
                WHERE d.account_id=? AND d.symbol=? AND t.result IS NOT NULL""",
                (self.config.trading_account_id, symbol)).fetchall()
        owned = Decimal(0)
        for row in rows:
            try:
                payload, result = json.loads(row["payload"]), json.loads(row["result"])
                filled = number(result.get("filled_qty", "0"))
                owned += filled if payload.get("side") == "buy" else -filled
            except (ValueError, TypeError, json.JSONDecodeError, TradeError):
                raise TradeError("Automatic ownership ledger is inconsistent.") from None
        return max(owned, Decimal(0))

    async def _reconcile_unresolved(self) -> bool:
        with self.store.lock:
            rows = self.store.db.execute("""SELECT t.ticket,d.request_id FROM trades t
                JOIN auto_decisions d ON d.request_id=t.request_id
                WHERE d.account_id=? AND t.state IN ('submitting','unknown')
                ORDER BY t.created LIMIT 10""", (self.config.trading_account_id,)).fetchall()
        blocked = False
        for row in rows:
            result = await self.bridge.trading.reconcile(row["ticket"], self.config.trading_chat_id)
            with self.store.transaction() as db:
                db.execute("UPDATE auto_decisions SET state=?,result=?,updated=? WHERE request_id=?",
                           (result.get("state", "unknown"), json.dumps(result, sort_keys=True), time.time(), row["request_id"]))
            if result.get("state") in {"submitting", "unknown"}:
                blocked = True
        return blocked

    async def _size(self, symbol: str, action: str) -> tuple[str, str] | tuple[None, str]:
        account, positions, orders, quote_data = await asyncio.gather(
            self.bridge.trading.api.call("GET", "/v2/account"),
            self.bridge.trading.api.call("GET", "/v2/positions"),
            self.bridge.trading.api.call("GET", "/v2/orders", params={"status": "open", "limit": 100, "nested": "true"}),
            self.bridge.trading.api.call("GET", f"/v2/stocks/{symbol}/quotes/latest", data=True, params={"feed": "iex"}),
        )
        self.bridge.trading._check_account(account)
        if not isinstance(positions, list) or not isinstance(orders, list):
            return None, "broker positions or orders are unavailable"
        if any(o.get("symbol") == symbol for o in orders):
            return None, "an open broker order already exists for this symbol"
        quote = quote_data.get("quote", {}) if isinstance(quote_data, dict) and quote_data.get("symbol") == symbol else {}
        bid, ask = number(quote.get("bp")), number(quote.get("ap"))
        if not 0 < bid <= ask or ask / bid - 1 > Decimal("0.01"):
            return None, "quote is invalid or spread exceeds 1%"
        position = next((p for p in positions if p.get("symbol") == symbol and p.get("side") == "long"), None)
        owned = self._owned_filled(symbol)
        if action == "buy":
            if position or owned != 0:
                return None, "automatic strategy does not pyramid into an existing position"
            budget = min(Decimal(self.config.auto_order_usd), Decimal(self.config.auto_max_position_usd),
                         Decimal(self.config.max_order_usd), number(account.get("cash")), number(account.get("buying_power")))
            qty = int(budget // ask)
            if qty < 1:
                return None, "automatic budget cannot buy one whole share"
            return str(qty), _limit_price(ask, "buy")
        if owned < 1 or owned != owned.to_integral_value():
            return None, "no whole strategy-owned shares are available to sell"
        if not position:
            return None, "broker position no longer matches strategy ownership"
        current = number(position.get("qty"))
        available = number(position.get("qty_available"))
        if current != owned or available < owned:
            return None, "broker position changed outside the automatic strategy"
        max_qty = int(Decimal(self.config.max_order_usd) // bid)
        qty = min(int(owned), max_qty)
        if qty < 1:
            return None, "per-order limit cannot sell one whole share"
        return str(qty), _limit_price(bid, "sell")

    async def _notify(self, action: str, symbol: str, result: dict, request_id: str):
        if not self.config.auto_notify:
            return
        text = ("Automatic paper order: " + action.upper() + " " + symbol + "\n"
                + "State: " + result.get("state", "unknown") + "\n"
                + "Ticket: " + (result.get("ticket") or "—"))
        notice_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "auto-notice:" + request_id + ":" + result.get("state", "unknown")))
        await self.bridge.send(self.config.trading_chat_id, text, notice_id, _auto_notice=True)

    async def once(self) -> dict:
        async with self.lock:
            self._enabled()
            if not self.running():
                return {"action": "stopped"}
            if self.bridge.trading.paused():
                return {"action": "paused"}
            if (self.bridge.poll_error or self.bridge.last_poll_ok is None
                    or time.time() - self.bridge.last_poll_ok > 120
                    or not self.store.owns_lease(self.bridge.owner)):
                return {"action": "unhealthy_receiver"}
            if await self._reconcile_unresolved():
                return {"action": "blocked_unresolved"}
            session = await self._completed_session()
            for symbol in self.config.auto_symbols:
                existing = self._decision(session, symbol)
                if existing:
                    continue
                request_id = str(uuid.uuid5(uuid.NAMESPACE_URL,
                    f"auto:{self.config.trading_account_id}:{policy_fingerprint(self.config)}:{session}:{symbol}"))
                try:
                    report = await self.bridge.predictions.predict(symbol, self.config.auto_horizon)
                    action, reason = decision_from_report(self.config, report, symbol, session)
                    if action == "hold":
                        self._save_decision(session, symbol, request_id, action, "hold", reason, report)
                        return {"action": "hold", "symbol": symbol, "reason": reason, "session_date": session}
                    sized = await self._size(symbol, action)
                    if sized[0] is None:
                        self._save_decision(session, symbol, request_id, "hold", "hold", sized[1], report)
                        return {"action": "hold", "symbol": symbol, "reason": sized[1], "session_date": session}
                    qty, price = sized
                    self._save_decision(session, symbol, request_id, action, "prepared", reason, report)
                    draft = await self.bridge.trading.prepare(self.config.trading_chat_id, action, symbol, qty, price, request_id)
                    result = await self.bridge.trading.submit_automatic(draft["ticket"])
                    state = result.get("state", "unknown")
                    self._save_decision(session, symbol, request_id, action, state, reason, report, draft["ticket"], result)
                    await self._notify(action, symbol, result, request_id)
                    return {"action": action, "symbol": symbol, "session_date": session, "order": result}
                except Exception as exc:
                    self.last_error = str(exc) if isinstance(exc, TradeError) else "Automatic decision failed."
                    # Preserve the single-attempt invariant once a report/order attempt begins.
                    if not self._decision(session, symbol):
                        self._save_decision(session, symbol, request_id, "hold", "error", self.last_error, {})
                    return {"action": "error", "symbol": symbol, "session_date": session, "error": self.last_error}
            return {"action": "already_decided", "session_date": session}

    async def run(self):
        while not self.bridge.stop.is_set():
            try:
                result = await self.once()
                advance = (len(self.config.auto_symbols) > 1
                           and result.get("action") in {"buy", "sell", "hold", "error"})
                delay = 0.05 if advance else self.config.auto_interval_seconds
            except Exception:
                self.last_error = "Automatic trading loop failed safely."
                delay = max(self.config.auto_interval_seconds, 60)
            try:
                await asyncio.wait_for(self.bridge.stop.wait(), timeout=delay)
            except TimeoutError:
                pass
