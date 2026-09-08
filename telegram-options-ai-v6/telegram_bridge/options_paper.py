from __future__ import annotations

import asyncio
import json
import math
import statistics
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx

SYMBOLS = ("SPY", "SPX")
INTERVAL_SECONDS = 300
STARTING_CASH = 100000.0
RISK_FRACTION = 0.05
DAILY_LOSS_FRACTION = 0.03
TARGET_PCT = 0.35
STOP_PCT = 0.25
MULTIPLIER = 100
NY = ZoneInfo("America/New_York")


class OptionsPaperError(ValueError):
    pass


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs(kind: str, spot: float, strike: float, t: float, vol: float, r: float = 0.04, q: float = 0.013):
    t = max(t, 1 / (365 * 24 * 12))
    vol = max(vol, 0.01)
    root = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * vol * vol) * t) / (vol * root)
    d2 = d1 - vol * root
    if kind == "call":
        premium = spot * math.exp(-q * t) * _cdf(d1) - strike * math.exp(-r * t) * _cdf(d2)
        delta = math.exp(-q * t) * _cdf(d1)
    else:
        premium = strike * math.exp(-r * t) * _cdf(-d2) - spot * math.exp(-q * t) * _cdf(-d1)
        delta = -math.exp(-q * t) * _cdf(-d1)
    return max(premium, 0.01), delta


def _american(kind: str, spot: float, strike: float, t: float, vol: float, r: float = 0.04, q: float = 0.013):
    t = max(t, 1 / (365 * 24 * 12))
    steps = 80
    dt = t / steps
    u = math.exp(vol * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp((r - q) * dt)
    p = min(max((growth - d) / (u - d), 0.0), 1.0)
    disc = math.exp(-r * dt)
    values = []
    for j in range(steps + 1):
        s = spot * (u ** j) * (d ** (steps - j))
        values.append(max(s - strike, 0.0) if kind == "call" else max(strike - s, 0.0))
    for step in range(steps - 1, -1, -1):
        nxt = []
        for j in range(step + 1):
            s = spot * (u ** j) * (d ** (step - j))
            hold = disc * (p * values[j + 1] + (1 - p) * values[j])
            exercise = max(s - strike, 0.0) if kind == "call" else max(strike - s, 0.0)
            nxt.append(max(hold, exercise))
        values = nxt
    return max(values[0], 0.01)


def theoretical_option(symbol: str, kind: str, spot: float, strike: float, t: float, vol: float) -> dict:
    if symbol not in SYMBOLS or kind not in {"call", "put"} or min(spot, strike, t, vol) <= 0:
        raise OptionsPaperError("Invalid theoretical option inputs.")
    if symbol == "SPX":
        premium, delta = _bs(kind, spot, strike, t, vol)
        return {"premium": premium, "delta": delta, "style": "european", "settlement": "cash"}
    premium = _american(kind, spot, strike, t, vol)
    bump = max(spot * 0.001, 0.05)
    up = _american(kind, spot + bump, strike, t, vol)
    down = _american(kind, max(spot - bump, 0.01), strike, t, vol)
    delta = (up - down) / (2 * bump)
    return {"premium": premium, "delta": max(-1.0, min(1.0, delta)), "style": "american", "settlement": "shares"}


def _extract_headlines(payload: dict) -> list[str]:
    items = payload.get("news", []) if isinstance(payload, dict) else []
    seen: set[str] = set()
    headlines: list[str] = []
    for item in items if isinstance(items, list) else []:
        title = str(item.get("title", "")).strip() if isinstance(item, dict) else ""
        if title and title not in seen:
            seen.add(title)
            headlines.append(title)
        if len(headlines) >= 8:
            break
    return headlines


class KeylessUnderlyingFeed:
    """Underlying-price feed only. It never fetches or represents a real option chain."""
    def __init__(self, client: httpx.AsyncClient | None = None):
        self.client = client or httpx.AsyncClient(timeout=12, follow_redirects=False, trust_env=False,
                                                 headers={"User-Agent": "FahadPaperOptions/5"})

    async def _chart(self, symbol: str, interval: str, range_: str):
        ticker = "^SPX" if symbol == "SPX" else "SPY"
        url = "https://query1.finance.yahoo.com/v8/finance/chart/" + quote(ticker, safe="")
        response = await self.client.get(url, params={"interval": interval, "range": range_})
        response.raise_for_status()
        data = response.json()
        result = data.get("chart", {}).get("result")
        if not isinstance(result, list) or not result:
            raise OptionsPaperError("Underlying feed returned no chart data.")
        return result[0]

    @staticmethod
    def _closes(chart: dict) -> list[float]:
        values = ((chart.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
        return [float(v) for v in values if isinstance(v, (int, float)) and v > 0 and math.isfinite(v)]

    async def _headlines(self, symbol: str) -> list[str]:
        query = "SPY ETF" if symbol == "SPY" else "S&P 500 index"
        try:
            response = await self.client.get("https://query1.finance.yahoo.com/v1/finance/search",
                                             params={"q": query, "newsCount": 8, "quotesCount": 0})
            response.raise_for_status()
            return _extract_headlines(response.json())
        except Exception:
            return []

    async def snapshot(self, symbol: str) -> dict:
        if symbol not in SYMBOLS:
            raise OptionsPaperError("Only SPY and SPX are supported.")
        intraday, daily, headlines = await asyncio.gather(
            self._chart(symbol, "5m", "1d"), self._chart(symbol, "1d", "3mo"), self._headlines(symbol))
        closes = self._closes(intraday)
        daily_closes = self._closes(daily)
        timestamps = intraday.get("timestamp") or []
        if len(closes) < 14 or len(daily_closes) < 20 or not timestamps:
            raise OptionsPaperError("Underlying feed has insufficient bars.")
        ts = int(timestamps[-1])
        now = time.time()
        et = datetime.fromtimestamp(now, NY)
        market_open = et.weekday() < 5 and (et.hour, et.minute) >= (9, 30) and (et.hour, et.minute) < (16, 0)
        if now - ts > 20 * 60:
            market_open = False
        return {"symbol": symbol, "spot": closes[-1], "closes_5m": closes, "daily_closes": daily_closes,
                "timestamp": ts, "market_open": market_open, "headlines": headlines}

    async def close(self):
        await self.client.aclose()


def _ema(values: list[float], span: int) -> float:
    alpha = 2.0 / (span + 1)
    value = values[0]
    for x in values[1:]:
        value = alpha * x + (1 - alpha) * value
    return value


def _signal(closes: list[float]) -> tuple[str, float, str]:
    if len(closes) < 14:
        return "hold", 0.0, "insufficient intraday bars"
    fast, slow = _ema(closes[-14:], 5), _ema(closes[-20:] if len(closes) >= 20 else closes, 13)
    ret = closes[-1] / closes[-7] - 1 if len(closes) >= 7 else 0.0
    gap = fast / slow - 1
    strength = min(1.0, abs(ret) * 80 + abs(gap) * 150)
    if fast > slow and ret > 0.001:
        return "call", strength, "upward EMA and 30-minute momentum agree"
    if fast < slow and ret < -0.001:
        return "put", strength, "downward EMA and 30-minute momentum agree"
    return "hold", strength, "momentum is mixed or weak"


def _volatility(daily: list[float]) -> float:
    returns = [math.log(b / a) for a, b in zip(daily, daily[1:]) if a > 0 and b > 0]
    if len(returns) < 10:
        return 0.25
    return min(1.20, max(0.12, statistics.stdev(returns) * math.sqrt(252)))


def _next_weekday(day):
    nxt = day + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def _expiry(strength: float, now: datetime) -> str:
    if now.weekday() < 5 and (now.hour, now.minute) < (14, 0) and strength >= 0.75:
        return now.date().isoformat()
    return _next_weekday(now.date()).isoformat()


def _time_to_expiry(expiry: str, now: datetime) -> float:
    exp = datetime.fromisoformat(expiry + "T16:00:00").replace(tzinfo=NY)
    seconds = max((exp - now).total_seconds(), 300)
    return seconds / (365.0 * 86400.0)


def _select_contract(symbol: str, kind: str, spot: float, expiry: str, iv: float, now: datetime) -> dict:
    step = 1.0 if symbol == "SPY" else 5.0
    center = round(spot / step) * step
    t = _time_to_expiry(expiry, now)
    candidates = []
    for offset in range(-10, 11):
        strike = max(step, center + offset * step)
        priced = theoretical_option(symbol, kind, spot, strike, t, iv)
        candidates.append({**priced, "strike": strike, "expiry": expiry, "iv": iv})
    return min(candidates, key=lambda x: abs(abs(x["delta"]) - 0.35))


class OptionsPaperEngine:
    def __init__(self, bridge, feed=None, model_ensemble=None):
        self.bridge, self.config, self.store = bridge, bridge.config, bridge.store
        self.feed = feed or KeylessUnderlyingFeed()
        self.model_ensemble = model_ensemble
        self.lock = asyncio.Lock()
        self.last_error = None
        self._force_counter = 0

    @property
    def owner_chat_id(self) -> int:
        return self.config.allowed_chat_ids[0]

    def _state(self) -> dict:
        try:
            return json.loads(self.store.get("options_paper_state", "{}"))
        except json.JSONDecodeError:
            return {}

    def running(self) -> bool:
        return self._state().get("running") is True

    def set_running(self, chat_id: int, running: bool) -> dict:
        if chat_id != self.owner_chat_id:
            raise OptionsPaperError("Options paper controls are restricted to the owner chat.")
        state = {"running": bool(running), "changed_at": time.time()}
        self.store.set("options_paper_state", json.dumps(state, sort_keys=True))
        return {"running": state["running"], "paper_only": True, "symbols": list(SYMBOLS)}

    def positions(self) -> list[dict]:
        with self.store.lock:
            rows = self.store.db.execute("SELECT * FROM option_positions WHERE status='open' ORDER BY entry_time DESC").fetchall()
        return [dict(r) for r in rows]

    def trades(self, limit: int = 20) -> list[dict]:
        with self.store.lock:
            rows = self.store.db.execute("SELECT * FROM option_positions WHERE status='closed' ORDER BY exit_time DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def pnl(self) -> dict:
        with self.store.lock:
            row = self.store.db.execute("SELECT COALESCE(SUM(pnl),0),COUNT(*) FROM option_positions WHERE status='closed'").fetchone()
            open_rows = self.store.db.execute("SELECT entry_premium,last_premium,contracts,multiplier FROM option_positions WHERE status='open'").fetchall()
        realized = float(row[0])
        unrealized = sum((r["last_premium"] - r["entry_premium"]) * r["contracts"] * r["multiplier"] for r in open_rows)
        return {"starting_cash": STARTING_CASH, "realized_pnl": round(realized, 2), "unrealized_pnl": round(unrealized, 2),
                "paper_equity": round(STARTING_CASH + realized + unrealized, 2), "closed_trades": int(row[1]),
                "open_positions": len(open_rows)}

    def status(self) -> dict:
        return {"running": self.running(), "paper_only": True, "symbols": list(SYMBOLS),
                "interval_seconds": INTERVAL_SECONDS, "positions": self.positions(), "pnl": self.pnl(),
                "last_error": self.last_error}

    def model_status(self) -> dict:
        if self.model_ensemble is None:
            return {"enabled": bool(getattr(self.config, "options_models_enabled", False)),
                    "available": [], "errors": {},
                    "device": getattr(self.config, "options_model_device", "auto")}
        result = self.model_ensemble.status()
        return {"enabled": True, "device": getattr(self.config, "options_model_device", "auto"), **result}

    def _slot(self, force: bool = False) -> int:
        base = int(time.time() // INTERVAL_SECONDS)
        if force:
            self._force_counter += 1
            return base + self._force_counter + 10_000_000
        return base

    def _record_decision(self, symbol: str, slot: int, action: str, reason: str) -> bool:
        try:
            with self.store.transaction() as db:
                db.execute("INSERT INTO option_decisions VALUES (?,?,?,?,?)", (symbol, slot, action, reason, time.time()))
            return True
        except sqlite3.IntegrityError:
            return False

    def _record_model_signal(self, symbol: str, slot: int, result) -> None:
        components = json.dumps([vote.as_dict() for vote in result.votes], sort_keys=True)
        with self.store.transaction() as db:
            db.execute("""INSERT OR IGNORE INTO option_model_signals
                (symbol,decision_slot,decision,confidence,score,components,degraded,reason,created)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (symbol, slot, result.kind, float(result.confidence), float(result.score), components,
                 1 if result.degraded else 0, result.reason, time.time()))

    def _daily_loss_blocked(self) -> bool:
        et = datetime.now(NY)
        start = et.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        with self.store.lock:
            loss = float(self.store.db.execute(
                "SELECT COALESCE(SUM(pnl),0) FROM option_positions WHERE status='closed' AND exit_time>=?", (start,)).fetchone()[0])
        return loss <= -(STARTING_CASH * DAILY_LOSS_FRACTION)

    async def _mark_or_close(self, symbol: str, snapshot: dict, kind: str, strength: float, now: datetime):
        with self.store.lock:
            row = self.store.db.execute("SELECT * FROM option_positions WHERE status='open' AND symbol=?", (symbol,)).fetchone()
        if not row:
            return None
        pos = dict(row)
        iv = _volatility(snapshot["daily_closes"])
        priced = theoretical_option(symbol, pos["option_type"], float(snapshot["spot"]), float(pos["strike"]),
                                    _time_to_expiry(pos["expiry"], now), iv)
        premium = priced["premium"]
        reason = None
        if premium >= pos["entry_premium"] * (1 + TARGET_PCT):
            reason = "target"
        elif premium <= pos["entry_premium"] * (1 - STOP_PCT):
            reason = "stop"
        elif kind in {"call", "put"} and kind != pos["option_type"] and strength >= 0.55:
            reason = "signal_reversal"
        expiry_close = datetime.fromisoformat(pos["expiry"] + "T16:00:00").replace(tzinfo=NY)
        if now >= expiry_close:
            reason = "expiry"
        pnl = (premium - pos["entry_premium"]) * pos["contracts"] * pos["multiplier"]
        with self.store.transaction() as db:
            if reason:
                db.execute("""UPDATE option_positions SET status='closed',last_premium=?,last_spot=?,last_mark_time=?,
                    exit_premium=?,exit_spot=?,exit_time=?,exit_reason=?,pnl=? WHERE id=?""",
                           (premium, snapshot["spot"], time.time(), premium, snapshot["spot"], time.time(), reason, pnl, pos["id"]))
            else:
                db.execute("UPDATE option_positions SET last_premium=?,last_spot=?,last_mark_time=?,iv=?,delta=? WHERE id=?",
                           (premium, snapshot["spot"], time.time(), iv, priced["delta"], pos["id"]))
        if reason:
            with self.store.lock:
                closed = dict(self.store.db.execute("SELECT * FROM option_positions WHERE id=?", (pos["id"],)).fetchone())
            return {"action": "closed", "position": closed}
        return {"action": "holding", "position": {**pos, "last_premium": premium, "last_spot": snapshot["spot"]}}

    async def _notify(self, text: str, key: str):
        rid = str(uuid.uuid5(uuid.NAMESPACE_URL, "options-paper:" + key))
        return await self.bridge.send(self.owner_chat_id, text, rid, _options_notice=True)

    async def _process_symbol(self, symbol: str, slot: int) -> dict:
        snapshot = await self.feed.snapshot(symbol)
        if snapshot.get("market_open") is False:
            if self._record_decision(symbol, slot, "hold", "regular market session is closed or underlying data is stale"):
                return {"action": "hold", "symbol": symbol, "reason": "market_closed"}
            return {"action": "already_decided", "symbol": symbol}
        kind, strength, signal_reason = _signal(snapshot["closes_5m"])
        if self.model_ensemble is not None:
            try:
                ensemble = await asyncio.to_thread(self.model_ensemble.evaluate, snapshot, kind, strength)
                self._record_model_signal(symbol, slot, ensemble)
                kind, strength, signal_reason = ensemble.kind, ensemble.confidence, ensemble.reason
            except Exception as exc:
                self.last_error = f"Options model ensemble failed: {str(exc)[:160]}"
        now = datetime.fromtimestamp(snapshot.get("timestamp", time.time()), NY)
        marked = await self._mark_or_close(symbol, snapshot, kind, strength, now)
        if marked:
            self._record_decision(symbol, slot, marked["action"], marked["position"].get("exit_reason") or "position marked")
            if marked["action"] == "closed":
                await self._notify(f"Options PAPER closed: {symbol} | {marked['position']['exit_reason']} | P/L {marked['position']['pnl']:.2f} USD", marked["position"]["id"] + ":close")
            return marked
        if not self._record_decision(symbol, slot, kind, signal_reason):
            return {"action": "already_decided", "symbol": symbol}
        if kind == "hold":
            return {"action": "hold", "symbol": symbol, "reason": signal_reason}
        if self._daily_loss_blocked():
            return {"action": "hold", "symbol": symbol, "reason": "daily paper loss limit reached"}
        iv = _volatility(snapshot["daily_closes"])
        expiry = _expiry(strength, now)
        contract = _select_contract(symbol, kind, float(snapshot["spot"]), expiry, iv, now)
        equity = self.pnl()["paper_equity"]
        budget = max(0.0, equity * RISK_FRACTION)
        contract_cost = contract["premium"] * MULTIPLIER
        contracts = int(budget // contract_cost)
        if contracts < 1:
            return {"action": "hold", "symbol": symbol, "reason": "paper risk budget cannot buy one contract"}
        contracts = min(contracts, 10)
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"paper-option:{symbol}:{slot}"))
        with self.store.transaction() as db:
            db.execute("""INSERT INTO option_positions
                (id,symbol,option_type,strike,expiry,contracts,multiplier,entry_premium,entry_spot,entry_time,
                 signal_strength,delta,iv,settlement,style,status,last_premium,last_spot,last_mark_time)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'open',?,?,?)""",
                (pid, symbol, kind, contract["strike"], expiry, contracts, MULTIPLIER, contract["premium"],
                 snapshot["spot"], time.time(), strength, contract["delta"], iv, contract["settlement"],
                 contract["style"], contract["premium"], snapshot["spot"], time.time()))
        with self.store.lock:
            position = dict(self.store.db.execute("SELECT * FROM option_positions WHERE id=?", (pid,)).fetchone())
        await self._notify(f"Options PAPER opened: {symbol} {kind.upper()} {contract['strike']:g} exp {expiry} | theoretical premium {contract['premium']:.2f} | x{contracts}", pid + ":open")
        return {"action": "opened", "position": position}

    async def once(self, *, symbols=SYMBOLS, force_new_slot: bool = False) -> dict:
        async with self.lock:
            if not self.running():
                return {"action": "stopped"}
            if self.bridge.poll_error or self.bridge.last_poll_ok is None or time.time() - self.bridge.last_poll_ok > 120:
                return {"action": "unhealthy_receiver"}
            if not self.store.owns_lease(self.bridge.owner):
                return {"action": "unhealthy_receiver"}
            slot = self._slot(force_new_slot)
            selected = tuple(symbol for symbol in symbols if symbol in SYMBOLS)
            results = [await self._process_symbol(symbol, slot) for symbol in selected]
            if len(selected) == 1:
                return results[0] if results else {"action": "already_decided"}
            return {"action": "cycle", "results": results}

    async def run(self):
        while not self.bridge.stop.is_set():
            delay = INTERVAL_SECONDS
            try:
                if self.running():
                    await self.once()
            except Exception as exc:
                self.last_error = str(exc)[:200] if isinstance(exc, OptionsPaperError) else "Options paper cycle failed."
                delay = 30
            try:
                await asyncio.wait_for(self.bridge.stop.wait(), timeout=delay)
            except TimeoutError:
                pass

    async def close(self):
        close = getattr(self.feed, "close", None)
        if close:
            await close()
