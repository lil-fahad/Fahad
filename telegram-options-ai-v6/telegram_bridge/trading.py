"""US-equity limit orders with confirmed manual and guarded automatic-paper paths.

Broker submissions are claimed in SQLite before POST. Ambiguous responses are reconciled
by client_order_id, never retried as a new submission. MCP only reads or prepares drafts.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo

import httpx

from .market import symbol_checked

BASES = {"paper": "https://paper-api.alpaca.markets", "live": "https://api.alpaca.markets"}
TERMINAL = {"filled", "canceled", "expired", "rejected", "replaced"}


class TradeError(ValueError):
    pass


class BrokerError(TradeError):
    def __init__(self, status: int = 0, ambiguous: bool = False):
        self.status, self.ambiguous = status, ambiguous
        messages = {401: "مفاتيح الوسيط غير صالحة لهذا الحساب.", 403: "الوسيط منع الطلب أو الحساب مقيّد.",
                    404: "لم يظهر الأمر لدى الوسيط بعد.", 422: "الوسيط رفض معلمات الأمر أو الرصيد المتاح.",
                    429: "تم بلوغ حد طلبات الوسيط؛ حاول القراءة لاحقًا."}
        super().__init__(messages.get(status, "تعذر تأكيد استجابة الوسيط؛ تحقق من حالة الأمر قبل أي طلب جديد."))


def number(value) -> Decimal:
    try:
        if isinstance(value, bool):
            raise InvalidOperation()
        result = Decimal(str(value))
        if not result.is_finite():
            raise InvalidOperation()
        return result
    except (InvalidOperation, TypeError, ValueError):
        raise TradeError("قيمة رقمية غير صالحة في الطلب أو بيانات الوسيط.") from None


def timestamp(value) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError()
        return parsed
    except (ValueError, AttributeError):
        raise TradeError("تاريخ غير صالح لدى الوسيط.") from None


def order_values(side: str, symbol: str, qty: str, limit_price: str) -> dict:
    symbol = symbol_checked(symbol)
    if side not in {"buy", "sell"} or not re.fullmatch(r"[1-9][0-9]{0,5}", str(qty)):
        raise TradeError("استخدم buy أو sell وكمية صحيحة موجبة من الأسهم.")
    if not re.fullmatch(r"[0-9]{1,7}(?:\.[0-9]{1,4})?", str(limit_price)):
        raise TradeError("السعر المحدد غير صالح.")
    price = number(limit_price)
    precision = Decimal("0.01") if price >= 1 else Decimal("0.0001")
    if price <= 0 or price.quantize(precision) != price:
        raise TradeError("السعر يجب أن يكون موجبًا؛ منزلتان عشريتان للأسعار من دولار وأربع لما دون ذلك.")
    return {"symbol": symbol, "qty": str(int(qty)), "side": side, "type": "limit",
            "limit_price": format(price, "f"), "time_in_force": "day", "extended_hours": False}


class AlpacaAPI:
    def __init__(self, mode: str, key: str, secret: str, client: httpx.AsyncClient | None = None):
        if mode not in BASES:
            raise TradeError("بيئة الوسيط غير صالحة.")
        self.mode, self.base = mode, BASES[mode]
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self.client = client or httpx.AsyncClient(timeout=15, follow_redirects=False, trust_env=False)
        for name in ("httpx", "httpcore"):
            logging.getLogger(name).setLevel(logging.CRITICAL)

    async def call(self, method: str, path: str, *, data=False, params=None, body=None):
        allowed = not data and method == "GET" and (
            path in {"/v2/account", "/v2/clock", "/v2/positions", "/v2/orders", "/v2/orders:by_client_order_id", "/v2/calendar"}
            or re.fullmatch(r"/v2/assets/[A-Z][A-Z0-9.-]{0,14}", path))
        allowed = allowed or (data and method == "GET" and re.fullmatch(r"/v2/stocks/[A-Z][A-Z0-9.-]{0,14}/quotes/latest", path))
        allowed = allowed or (not data and method == "POST" and path == "/v2/orders")
        allowed = allowed or (not data and method == "DELETE" and re.fullmatch(r"/v2/orders/[0-9a-f-]{36}", path))
        if not allowed:
            raise TradeError("عملية وسيط غير مدعومة.")
        url = ("https://data.alpaca.markets" if data else self.base) + path
        ambiguous = method in {"POST", "DELETE"}
        try:
            async with self.client.stream(method, url, params=params, json=body, headers=self._headers) as response:
                if response.status_code == 204:
                    return None
                if not 200 <= response.status_code < 300:
                    raise BrokerError(response.status_code, ambiguous=ambiguous and (response.status_code >= 500 or response.status_code in {408, 409}))
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > 2_000_000:
                        raise BrokerError(0, ambiguous)
                    chunks.append(chunk)
                return json.loads(b"".join(chunks))
        except (httpx.HTTPError, json.JSONDecodeError):
            raise BrokerError(0, ambiguous) from None

    async def close(self):
        await self.client.aclose()


class TradingService:
    def __init__(self, config, store, api: AlpacaAPI):
        self.config, self.store, self.api = config, store, api
        self.lock = asyncio.Lock()
        self._auto_guard = None
        self._auto_owner = None
        self.halt_key = "trading_halted:" + config.trading_mode + ":" + config.trading_account_id

    def set_automatic_guard(self, guard, *, owner: str | None = None):
        """Install a synchronous final safety gate owned by the active Telegram bridge."""
        self._auto_guard = guard
        self._auto_owner = owner

    def _automatic_runtime_allowed(self) -> bool:
        return self._auto_guard is None or self._auto_guard() is True

    def enabled(self, chat_id: int | None = None):
        cfg = self.config
        if not cfg.trading_enabled:
            raise TradeError("التداول غير مفعّل. اضبط حساب الوسيط محليًا بأمر configure-trading.")
        if chat_id is not None and chat_id != cfg.trading_chat_id:
            raise TradeError("هذه المحادثة ليست محادثة صاحب حساب التداول.")
        if self.api.mode != cfg.trading_mode or self.api.base != BASES[cfg.trading_mode]:
            raise TradeError("بيئة الوسيط لا تطابق الإعداد المحفوظ.")

    def paused(self) -> bool:
        return self.store.get(self.halt_key, "0") == "1"

    def halt(self, chat_id: int, paused: bool) -> dict:
        self.enabled(chat_id)
        self.store.set(self.halt_key, "1" if paused else "0")
        return {"paused": paused, "mode": self.config.trading_mode,
                "detail": "إيقاف الأوامر الجديدة لا يلغي الأوامر الموجودة لدى الوسيط."}

    def _check_account(self, account: dict):
        cfg = self.config
        if not isinstance(account, dict) or account.get("id") != cfg.trading_account_id:
            raise TradeError("هوية حساب الوسيط تختلف عن الحساب الذي تم إعداده.")
        if (account.get("status") != "ACTIVE" or account.get("currency") != "USD"
                or any(account.get(flag) is not False for flag in ("trading_blocked", "account_blocked", "trade_suspended_by_user"))):
            raise TradeError("حساب الوسيط غير نشط أو التداول محظور أو العملة غير مدعومة.")

    async def account(self) -> dict:
        self.enabled()
        account, positions = await asyncio.gather(self.api.call("GET", "/v2/account"), self.api.call("GET", "/v2/positions"))
        self._check_account(account)
        if not isinstance(positions, list):
            raise TradeError("تعذر قراءة المحفظة.")
        return {"mode": self.config.trading_mode, "paused": self.paused(),
                "cash": str(number(account["cash"])), "buying_power": str(number(account["buying_power"])),
                "equity": str(number(account["equity"])), "currency": "USD",
                "max_order_usd": self.config.max_order_usd, "daily_buy_limit_usd": self.config.daily_buy_limit_usd,
                "positions": [{k: p.get(k) for k in ("symbol", "qty", "qty_available", "side", "market_value", "unrealized_pl")}
                              for p in positions[:50]], "positions_truncated": len(positions) > 50}

    def _row(self, ticket: str, chat_id: int | None = None) -> dict:
        self.enabled(chat_id)
        if not re.fullmatch(r"[0-9a-f]{12}", ticket):
            raise TradeError("رمز الأمر غير صالح.")
        with self.store.lock:
            row = self.store.db.execute("SELECT * FROM trades WHERE ticket=?", (ticket,)).fetchone()
        if (not row or row["chat_id"] != self.config.trading_chat_id or row["mode"] != self.config.trading_mode
                or row["account_id"] != self.config.trading_account_id):
            raise TradeError("الأمر غير موجود في الحساب والمحادثة الحاليين.")
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        result["result"] = json.loads(result["result"]) if result["result"] else None
        return result

    @staticmethod
    def view(row: dict) -> dict:
        return {"ticket": row["ticket"], "mode": row["mode"], "state": row["state"], "order": row["payload"],
                "expires_at": datetime.fromtimestamp(row["expires"], timezone.utc).isoformat(),
                "result": row["result"], "cancel_state": row["cancel_state"]}

    async def preflight(self, payload: dict) -> dict:
        if self.paused():
            raise TradeError("قبول صفقات جديدة متوقف. استخدم /resume في Telegram لاستئنافه.")
        symbol, side = payload["symbol"], payload["side"]
        account, clock, asset, positions, orders, quote_data = await asyncio.gather(
            self.api.call("GET", "/v2/account"), self.api.call("GET", "/v2/clock"),
            self.api.call("GET", "/v2/assets/" + symbol), self.api.call("GET", "/v2/positions"),
            self.api.call("GET", "/v2/orders", params={"status": "open", "limit": 100, "nested": "true"}),
            self.api.call("GET", f"/v2/stocks/{symbol}/quotes/latest", data=True, params={"feed": "iex"}))
        self._check_account(account)
        now = datetime.now(timezone.utc)
        if not isinstance(clock, dict) or clock.get("is_open") is not True:
            raise TradeError("الجلسة العادية للسوق مغلقة؛ لم يُرسل أمر.")
        if abs((now - timestamp(clock.get("timestamp"))).total_seconds()) > 120:
            raise TradeError("ساعة السوق قديمة؛ لم يُرسل أمر.")
        if not isinstance(asset, dict) or asset.get("symbol") != symbol or asset.get("class") != "us_equity" or asset.get("status") != "active" or asset.get("tradable") is not True:
            raise TradeError("الأصل ليس سهمًا أمريكيًا نشطًا قابلًا للتداول في هذه النسخة.")
        if not isinstance(orders, list) or not isinstance(positions, list):
            raise TradeError("تعذر التحقق من الأوامر أو الأسهم المتاحة.")
        if len(orders) >= self.config.max_open_orders:
            raise TradeError("تم بلوغ حد الأوامر المفتوحة. راجع /orders قبل طلب جديد.")
        if not isinstance(quote_data, dict) or quote_data.get("symbol") != symbol or not isinstance(quote_data.get("quote"), dict):
            raise TradeError("تعذر التحقق من رمز عرض السعر؛ لم يُرسل أمر.")
        quote = quote_data["quote"]
        age = (now - timestamp(quote.get("t"))).total_seconds()
        bid, ask = number(quote.get("bp")), number(quote.get("ap"))
        if not -5 <= age <= 60 or not 0 < bid <= ask:
            raise TradeError("عرض السعر غير صالح أو أقدم من دقيقة؛ لم يُرسل أمر.")
        price, qty = number(payload["limit_price"]), number(payload["qty"])
        reference = ask if side == "buy" else bid
        if abs(price / reference - 1) > Decimal("0.05"):
            raise TradeError("السعر المحدد يبتعد أكثر من 5% عن عرض IEX الحالي؛ راجعه.")
        notional = price * qty
        if notional > number(self.config.max_order_usd):
            raise TradeError("قيمة الأمر تتجاوز الحد المحلي لكل صفقة.")
        if side == "buy":
            # Reserve cash for every existing account order, including orders placed elsewhere.
            reserved = Decimal(0)
            for order in orders:
                if order.get("side") == "buy":
                    if order.get("limit_price") is None or order.get("notional") is not None:
                        raise TradeError("يوجد أمر شراء مفتوح لا يمكن تقدير حجزه النقدي؛ انتظر تسويته.")
                    reserved += max(number(order["qty"]) - number(order.get("filled_qty", "0")), 0) * number(order["limit_price"])
            available = min(number(account["buying_power"]), number(account["cash"]) - reserved)
            if notional > available:
                raise TradeError("النقد أو القوة الشرائية المتاحة لا تكفي لهذا الأمر.")
        else:
            position = next((p for p in positions if p.get("symbol") == symbol and p.get("side") == "long"), None)
            if not position or position.get("qty_available") is None or qty > number(position["qty_available"]):
                raise TradeError("عدد الأسهم المملوكة والمتاحة للبيع غير كافٍ؛ البيع على المكشوف غير مدعوم.")
        return {"quote_bid": str(bid), "quote_ask": str(ask), "quote_time": quote["t"], "quote_feed": "IEX",
                "notional_usd": str(notional), "checked_at": now.isoformat()}

    async def prepare(self, chat_id: int, side: str, symbol: str, qty: str, limit_price: str, request_id: str) -> dict:
        self.enabled(chat_id)
        try:
            request_id = str(uuid.UUID(request_id))
        except (ValueError, AttributeError):
            raise TradeError("معرّف الطلب يجب أن يكون UUID ثابتًا عند التكرار.") from None
        payload = order_values(side, symbol, qty, limit_price)
        client_id = "fahad-" + uuid.uuid5(uuid.NAMESPACE_URL, self.config.trading_mode + ":" + self.config.trading_account_id + ":" + request_id).hex
        payload["client_order_id"] = client_id
        async with self.lock:
            with self.store.lock:
                old = self.store.db.execute("SELECT ticket FROM trades WHERE request_id=?", (request_id,)).fetchone()
            if old:
                row = self._row(old["ticket"], chat_id)
                if row["payload"] != payload:
                    raise TradeError("معرّف الطلب استُخدم سابقًا لمعلمات مختلفة.")
                return self.view(row)
            checks = await self.preflight(payload)
            ticket, created = uuid.uuid4().hex[:12], time.time()
            with self.store.transaction() as db:
                drafts = db.execute("SELECT count(*) FROM trades WHERE state='draft' AND expires>?", (created,)).fetchone()[0]
                if drafts >= 20:
                    raise TradeError("هناك مسودات كثيرة؛ ألغِ المسودات غير المطلوبة.")
                db.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?, 'draft',NULL,NULL,?,NULL)",
                           (ticket, request_id, chat_id, self.config.trading_mode, self.config.trading_account_id,
                            json.dumps(payload, sort_keys=True), created, created + 120, json.dumps({"checks": checks})))
            return self.view(self._row(ticket, chat_id))

    def _claim(self, row: dict, *, automatic: bool = False) -> bool:
        day = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        auto_key = auto_fingerprint = None
        if automatic:
            from .autotrading import auto_state_key, policy_fingerprint
            auto_key = auto_state_key(self.config)
            auto_fingerprint = policy_fingerprint(self.config)
        with self.store.transaction() as db:
            halted = db.execute("SELECT value FROM kv WHERE key=?", (self.halt_key,)).fetchone()
            if halted and halted[0] == "1":
                raise TradeError("التداول متوقف؛ لم يُرسل أمر.")
            if automatic:
                saved = db.execute("SELECT value FROM kv WHERE key=?", (auto_key,)).fetchone()
                try:
                    auto_state = json.loads(saved[0]) if saved else {}
                except json.JSONDecodeError:
                    auto_state = {}
                if auto_state.get("running") is not True or auto_state.get("fingerprint") != auto_fingerprint:
                    raise TradeError("Automatic paper trading was stopped before the order was claimed.")
                if self._auto_owner:
                    lease = db.execute("SELECT owner,expires FROM leases WHERE name='poller'").fetchone()
                    if not lease or lease[0] != self._auto_owner or lease[1] <= time.time():
                        raise TradeError("Automatic paper trading lost the active receiver lease before the order was claimed.")
            fresh = db.execute("SELECT state,expires FROM trades WHERE ticket=?", (row["ticket"],)).fetchone()
            if fresh[0] != "draft":
                return False
            if fresh[1] < time.time():
                raise TradeError("انتهت مهلة التأكيد؛ أنشئ طلبًا جديدًا بسعر حالي.")
            unresolved = db.execute("SELECT 1 FROM trades WHERE mode=? AND account_id=? AND state IN ('submitting','unknown') LIMIT 1",
                                    (self.config.trading_mode, self.config.trading_account_id)).fetchone()
            if unresolved:
                raise TradeError("يوجد أمر غير محسوم؛ راجع /orders قبل إضافة أمر آخر.")
            active = db.execute("SELECT state,payload FROM trades WHERE mode=? AND account_id=? AND submitted_day=?",
                                (self.config.trading_mode, self.config.trading_account_id, day)).fetchall()
            spent = Decimal(0)
            for item in active:
                item_payload = json.loads(item["payload"])
                if item_payload["side"] == "buy" and item["state"] != "rejected":
                    spent += number(item_payload["qty"]) * number(item_payload["limit_price"])
            payload = row["payload"]
            if payload["side"] == "buy" and spent + number(payload["qty"]) * number(payload["limit_price"]) > number(self.config.daily_buy_limit_usd):
                raise TradeError("تم بلوغ سقف قيمة الشراء اليومي للبوت.")
            db.execute("UPDATE trades SET state='submitting',submitted_day=? WHERE ticket=?", (day, row["ticket"]))
            return True

    def _record(self, row: dict, order: dict):
        payload = row["payload"]
        try:
            valid = (isinstance(order, dict) and order.get("client_order_id") == payload["client_order_id"]
                     and all(str(order.get(k)) == str(payload[k]) for k in ("symbol", "side", "type", "time_in_force"))
                     and number(order.get("qty")) == number(payload["qty"])
                     and number(order.get("limit_price")) == number(payload["limit_price"])
                     and order.get("extended_hours") is False)
            remote_id = str(uuid.UUID(order.get("id", "")))
            if not valid or not isinstance(order.get("status"), str):
                raise ValueError()
            filled = number(order.get("filled_qty", "0"))
            if not 0 <= filled <= number(payload["qty"]):
                raise ValueError()
            average = order.get("filled_avg_price")
            if filled > 0 and (average is None or number(average) <= 0):
                raise ValueError()
            if average is not None:
                average = str(number(average))
            if order["status"] == "filled" and filled != number(payload["qty"]):
                raise ValueError()
        except (ValueError, AttributeError, TypeError):
            raise BrokerError(0, True) from None
        fields = {k: order.get(k) for k in ("status", "filled_qty", "filled_avg_price", "submitted_at", "filled_at", "canceled_at")}
        fields["filled_qty"] = str(filled)
        fields["filled_avg_price"] = average
        with self.store.transaction() as db:
            db.execute("UPDATE trades SET state=?,remote_id=?,result=? WHERE ticket=?",
                       ("rejected" if fields["status"] == "rejected" else "submitted", remote_id, json.dumps(fields), row["ticket"]))

    async def reconcile(self, ticket: str, chat_id: int | None = None) -> dict:
        row = self._row(ticket, chat_id)
        if row["state"] in {"submitting", "unknown", "submitted"}:
            try:
                order = await self.api.call("GET", "/v2/orders:by_client_order_id", params={"client_order_id": row["payload"]["client_order_id"]})
                self._record(row, order)
            except BrokerError:
                # A missing lookup after an interrupted POST is NOT evidence that no order exists.
                with self.store.transaction() as db:
                    db.execute("UPDATE trades SET state='unknown' WHERE ticket=?", (ticket,))
            row = self._row(ticket, chat_id)
        return self.view(row)

    def _automatic_request(self, request_id: str) -> bool:
        with self.store.lock:
            return self.store.db.execute("SELECT 1 FROM auto_decisions WHERE request_id=?", (request_id,)).fetchone() is not None

    def _automatic_running(self) -> bool:
        from .autotrading import auto_state_key, policy_fingerprint
        if not self.config.auto_enabled or self.config.trading_mode != "paper":
            return False
        try:
            state = json.loads(self.store.get(auto_state_key(self.config), "{}"))
        except json.JSONDecodeError:
            return False
        return state.get("running") is True and state.get("fingerprint") == policy_fingerprint(self.config)

    async def confirm(self, chat_id: int, ticket: str) -> dict:
        self.enabled(chat_id)
        async with self.lock:
            row = self._row(ticket, chat_id)
            if self._automatic_request(row["request_id"]):
                raise TradeError("This automatic order is controlled by the automatic paper-trading engine.")
            if row["state"] != "draft":
                return await self.reconcile(ticket, chat_id)
            if row["expires"] < time.time():
                raise TradeError("انتهت مهلة التأكيد؛ أنشئ طلبًا جديدًا بسعر حالي.")
            await self.preflight(row["payload"])
            if not self._claim(row):
                return await self.reconcile(ticket, chat_id)
            try:
                self._record(row, await self.api.call("POST", "/v2/orders", body=row["payload"]))
            except BrokerError as error:
                with self.store.transaction() as db:
                    db.execute("UPDATE trades SET state=?,result=? WHERE ticket=?",
                               ("unknown" if error.ambiguous else "rejected", json.dumps({"detail": str(error)}), ticket))
                if error.ambiguous:
                    return await self.reconcile(ticket, chat_id)
            return self.view(self._row(ticket, chat_id))

    async def submit_automatic(self, ticket: str) -> dict:
        """Submit only a durable auto_decision draft while the exact paper policy is running."""
        self.enabled(self.config.trading_chat_id)
        async with self.lock:
            row = self._row(ticket, self.config.trading_chat_id)
            if not self._automatic_request(row["request_id"]):
                raise TradeError("Automatic submission requires a durable automatic decision.")
            if not self._automatic_running() or not self._automatic_runtime_allowed():
                raise TradeError("Automatic paper trading is stopped or the receiver is unhealthy.")
            if row["state"] != "draft":
                return await self.reconcile(ticket, self.config.trading_chat_id)
            await self.preflight(row["payload"])
            if not self._automatic_running() or not self._automatic_runtime_allowed():
                raise TradeError("Automatic paper trading was stopped or the receiver became unhealthy before submission.")
            if not self._claim(row, automatic=True):
                return await self.reconcile(ticket, self.config.trading_chat_id)
            try:
                self._record(row, await self.api.call("POST", "/v2/orders", body=row["payload"]))
            except BrokerError as error:
                with self.store.transaction() as db:
                    db.execute("UPDATE trades SET state=?,result=? WHERE ticket=?",
                               ("unknown" if error.ambiguous else "rejected", json.dumps({"detail": str(error)}), ticket))
                if error.ambiguous:
                    return await self.reconcile(ticket, self.config.trading_chat_id)
            return self.view(self._row(ticket, self.config.trading_chat_id))

    async def cancel(self, chat_id: int, ticket: str) -> dict:
        self.enabled(chat_id)
        async with self.lock:
            row = self._row(ticket, chat_id)
            if row["state"] == "draft":
                with self.store.transaction() as db:
                    db.execute("UPDATE trades SET state='canceled' WHERE ticket=? AND state='draft'", (ticket,))
                return self.view(self._row(ticket, chat_id))
            await self.reconcile(ticket, chat_id)
            row = self._row(ticket, chat_id)
            if row["state"] in {"rejected", "canceled"} or (row["result"] or {}).get("status") in TERMINAL:
                return self.view(row)
            if row["state"] == "unknown" or not row["remote_id"]:
                raise TradeError("لم يُحسم وجود الأمر؛ تحقق منه لدى الوسيط قبل محاولة إلغائه.")
            with self.store.transaction() as db:
                current = db.execute("SELECT cancel_state FROM trades WHERE ticket=?", (ticket,)).fetchone()[0]
                if current:
                    return self.view(row)
                db.execute("UPDATE trades SET cancel_state='requesting' WHERE ticket=?", (ticket,))
            try:
                await self.api.call("DELETE", "/v2/orders/" + row["remote_id"])
                cancel_state = "requested"
            except BrokerError as error:
                cancel_state = "unknown" if error.ambiguous else "rejected"
            with self.store.transaction() as db:
                db.execute("UPDATE trades SET cancel_state=? WHERE ticket=?", (cancel_state, ticket))
            return await self.reconcile(ticket, chat_id)

    async def orders(self) -> dict:
        self.enabled()
        with self.store.lock:
            rows = self.store.db.execute("SELECT ticket FROM trades WHERE mode=? AND account_id=? AND chat_id=? ORDER BY CASE WHEN state IN ('submitting','unknown') THEN 0 ELSE 1 END,created DESC LIMIT 10",
                (self.config.trading_mode, self.config.trading_account_id, self.config.trading_chat_id)).fetchall()
        # Limit request burst to ten broker reads; no resubmission occurs during reconciliation.
        results = [await self.reconcile(row["ticket"]) for row in rows]
        return {"mode": self.config.trading_mode, "orders": results, "coverage": "Up to 10 bot orders, unresolved submissions first, then newest. All account open orders are checked before submissions."}


def render_order(result: dict) -> str:
    order = result["order"]
    mode = "تجريبي — أموال افتراضية" if result["mode"] == "paper" else "حقيقي — أموال فعلية"
    side = "شراء" if order["side"] == "buy" else "بيع"
    lines = [mode, f"{side} {order['qty']} من {order['symbol']}",
             f"السعر المحدد: {order['limit_price']} USD | صلاحية الأمر: جلسة اليوم",
             f"القيمة عند السعر المحدد: {number(order['qty']) * number(order['limit_price'])} USD",
             f"رمز الأمر: {result['ticket']}"]
    if result["state"] == "draft":
        checks = (result["result"] or {}).get("checks", {})
        lines += [f"عرض IEX: شراء {checks.get('quote_ask', '—')} / بيع {checks.get('quote_bid', '—')}",
                  "لم يُرسل إلى الوسيط بعد. مهلة التأكيد دقيقتان من إنشاء المسودة.",
                  f"للتأكيد: /confirm {result['ticket']}", f"للإلغاء: /cancel {result['ticket']}"]
    elif result["state"] in {"submitting", "unknown"}:
        lines.append("حالة الإرسال غير محسومة. استخدم /orders؛ لا تُنشئ أمرًا بديلًا قبل التحقق لدى الوسيط.")
    elif result["state"] == "submitted":
        status = result["result"] or {}
        labels = {"new": "مفتوح", "accepted": "مقبول", "pending_new": "قيد القبول", "partially_filled": "منفذ جزئيًا",
                  "filled": "منفذ بالكامل", "canceled": "ملغى", "expired": "منتهي", "rejected": "مرفوض", "pending_cancel": "الإلغاء قيد المعالجة"}
        lines += ["حالة الوسيط: " + labels.get(status.get("status"), "قيد المتابعة لدى الوسيط"),
                  f"الكمية المنفذة: {status.get('filled_qty', '0')} | متوسط التنفيذ: {status.get('filled_avg_price') or '—'}"]
    else:
        lines.append("الطلب ملغى محليًا." if result["state"] == "canceled" else "الطلب مرفوض ولم يُعَد إرساله.")
    if result.get("cancel_state") and (result["result"] or {}).get("status") not in TERMINAL:
        lines.append("طلب الإلغاء يحتاج متابعة حالة الوسيط؛ قد يتزامن مع تنفيذ جزئي أو كامل.")
    return "\n".join(lines)
