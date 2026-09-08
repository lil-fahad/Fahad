"""Only explicit fresh commands in allowlisted private chats can enqueue a reply."""
from __future__ import annotations

import asyncio
import re
import time
import uuid

from .market import MarketError, symbol_checked
from .prediction import HORIZONS, render
from .trading import TradeError, order_values, render_order
from .options_paper import OptionsPaperError

TRADE_COMMANDS = {"buy", "sell", "confirm", "cancel", "account", "orders", "order", "pause", "resume",
                  "autoon", "autooff", "autostatus"}
OPTIONS_COMMANDS = {"optionson", "optionsoff", "optionsstatus", "optionsmodels", "optionpositions", "optiontrades", "optionpnl"}
OPTIONS_HELP = """عقود Paper بدون وسيط:
/optionson — تشغيل محاكاة SPY + SPX
/optionsoff — إيقاف فتح صفقات جديدة
/optionsstatus — حالة المحرك
/optionsmodels — حالة Chronos-2 + TimesFM 2.5 + FinBERT
/optionpositions — المراكز المفتوحة
/optiontrades — آخر الصفقات المغلقة
/optionpnl — الربح والخسارة التجريبي
الأسعار Premium نظرية وليست أسعار عقود قابلة للتنفيذ."""
TRADE_HELP = """أوامر التداول في محادثة صاحب الحساب:
/buy AAPL 1 250 — معاينة شراء سهم بسعر محدد
/sell AAPL 1 250 — معاينة بيع سهم مملوك
/confirm CODE — تأكيد المسودة خلال دقيقتين
/cancel CODE — طلب إلغاء مسودة أو أمر
/order CODE — حالة أمر محدد
/orders — آخر أوامر البوت
/account — الرصيد والمحفظة
/pause — إيقاف قبول أوامر جديدة
/resume — استئناف قبول الأوامر
/autoon — تشغيل التداول التلقائي التجريبي
/autooff — إيقاف الصفقات التلقائية الجديدة
/autostatus — حالة المحرك التلقائي
الأسعار أعلاه أمثلة؛ أدخل السعر الذي تريده. التوقع وحده لا يرسل أمر تداول."""

HELP = """بوت فهد للتوقعات التجريبية — أسهم أمريكية، بيانات يومية.
/predict AAPL 5 — توقع بعد 5 جلسات تداول
/backtest AAPL 5 — عرض الاختبار التاريخي
/status — حالة مصدر البيانات
/help — هذه المساعدة
الأفق: 1 أو 5 أو 20 جلسة؛ الافتراضي 5. الأوامر تعمل في محادثتك الخاصة المسموحة فقط.
تُعرض الاحتمالات ونتيجة المقارنة والتاريخ؛ نتيجة التوقع غير مضمونة."""


def parse_update(update: dict, config) -> dict | None:
    message = update.get("message")  # Edited, channel, caption and forwarded content is not a command.
    if not isinstance(message, dict):
        return None
    chat, sender = message.get("chat", {}), message.get("from", {})
    if (chat.get("id") not in config.allowed_chat_ids or chat.get("type") != "private"
            or sender.get("id") != chat.get("id") or sender.get("is_bot")
            or any(k in message for k in ("forward_origin", "forward_date", "via_bot"))
            or not 0 <= time.time() - message.get("date", 0) <= 600):
        return None
    text = message.get("text", "")
    if not isinstance(text, str) or len(text) > 150:
        return None
    # Telegram supplies a bot_command entity for real slash commands; don't execute quoted prose.
    entities = message.get("entities", [])
    first = next((e for e in entities if e.get("type") == "bot_command" and e.get("offset") == 0), None)
    match = re.fullmatch(r"/([a-z]+)(?:@([A-Za-z0-9_]+))?(?:\s+(.*))?", text.strip(), re.DOTALL)
    if not first or not match or first.get("length") != len(text.split()[0]):
        return None
    verb, addressed, tail = match.groups()
    if addressed and addressed.casefold() != config.expected_bot_username.casefold():
        return None
    if verb not in {"predict", "backtest", "start", "help", "status"} | TRADE_COMMANDS | OPTIONS_COMMANDS:
        return None
    if verb not in OPTIONS_COMMANDS and not config.prediction_enabled and chat.get("id") != config.trading_chat_id:
        return None
    if verb in OPTIONS_COMMANDS and chat.get("id") != config.allowed_chat_ids[0]:
        return None
    if verb in TRADE_COMMANDS and (not config.trading_enabled or chat.get("id") != config.trading_chat_id
                                   or time.time() - message["date"] > 120):
        return None
    if verb in {"predict", "backtest"} and not config.prediction_enabled:
        return None
    request = {"verb": verb, "message_id": message["message_id"], "issued_at": message["date"]}
    parts = (tail or "").split()
    if verb in {"buy", "sell"}:
        try:
            if len(parts) != 3:
                raise TradeError("صيغة الأمر: /buy AAPL 1 250 أو /sell AAPL 1 250.")
            values = order_values(verb, *parts)
            request.update(symbol=values["symbol"], qty=values["qty"], limit_price=values["limit_price"])
        except ValueError as exc:
            request["error"] = str(exc)
    elif verb in {"confirm", "cancel", "order"}:
        if len(parts) != 1 or not re.fullmatch(r"[0-9a-f]{12}", parts[0]):
            request["error"] = "أدخل رمز الأمر كما ظهر في المعاينة. مثال: /confirm ثم الرمز."
        else:
            request["ticket"] = parts[0]
    elif verb in {"predict", "backtest"}:
        try:
            if not 1 <= len(parts) <= 2:
                raise ValueError()
            symbol = symbol_checked(parts[0])
            horizon = int(parts[1]) if len(parts) == 2 else 5
            if horizon not in HORIZONS:
                raise ValueError()
            request.update(symbol=symbol, horizon=horizon)
        except ValueError:
            request["error"] = "استخدم /predict AAPL 5 أو /backtest AAPL 5. الأفق: 1 أو 5 أو 20 جلسة."
    elif parts and verb != "start":
        request["error"] = OPTIONS_HELP if verb in OPTIONS_COMMANDS else HELP
    return request


async def trade_command(bridge, job) -> str:
    request, service = job["request"], bridge.trading
    verb, chat_id = request["verb"], job["chat_id"]
    service.enabled(chat_id)
    if time.time() - request["issued_at"] > 120:
        return "طلب التداول قديم؛ أرسل أمرًا جديدًا. لم يُرسل طلب جديد للوسيط."
    if verb in {"buy", "sell"}:
        request_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"trade:{bridge.config.expected_bot_username}:{chat_id}:{job['update_id']}"))
        return render_order(await service.prepare(chat_id, verb, request["symbol"], request["qty"], request["limit_price"], request_id))
    if verb == "confirm":
        return render_order(await service.confirm(chat_id, request["ticket"]))
    if verb == "cancel":
        return render_order(await service.cancel(chat_id, request["ticket"]))
    if verb == "order":
        return render_order(await service.reconcile(request["ticket"], chat_id))
    if verb in {"pause", "resume"}:
        result = service.halt(chat_id, verb == "pause")
        return ("قبول أوامر جديدة متوقف." if result["paused"] else "قبول أوامر جديدة مفعّل؛ كل صفقة تحتاج تأكيدك.") + "\n" + result["detail"]
    if verb in {"autoon", "autooff"}:
        if not bridge.config.auto_enabled:
            raise TradeError("Automatic trading is not configured. Use configure-autotrading locally first.")
        result = bridge.autotrading.set_running(chat_id, verb == "autoon")
        return ("التداول التلقائي التجريبي يعمل." if result["running"] else "التداول التلقائي التجريبي متوقف.")
    if verb == "autostatus":
        result = bridge.autotrading.status()
        latest = result.get("latest") or {}
        lines = ["التداول التلقائي: " + ("يعمل" if result["running"] else "متوقف"),
                 "الرموز: " + (", ".join(result["symbols"]) or "—"),
                 f"الأفق: {result['horizon_sessions']} جلسات | الفحص كل {result['interval_seconds']} ثانية"]
        if latest:
            lines.append(f"آخر قرار: {latest.get('session_date')} {latest.get('symbol')} — {latest.get('decision')} / {latest.get('state')}")
            if latest.get("reason"):
                lines.append("السبب: " + latest["reason"])
        if result.get("last_error"):
            lines.append("آخر خطأ آمن: " + result["last_error"])
        return "\n".join(lines)
    if verb == "account":
        account = await service.account()
        lines = ["الحساب: " + ("تجريبي" if account["mode"] == "paper" else "حقيقي"),
                 f"النقد: {account['cash']} USD | القوة الشرائية: {account['buying_power']} USD",
                 f"قيمة الحساب: {account['equity']} USD", "المراكز (حتى 10):"]
        lines += [f"{p['symbol']}: {p['qty']} سهم | المتاح {p['qty_available']}" for p in account["positions"][:10]]
        if not account["positions"]:
            lines.append("لا توجد مراكز مفتوحة.")
        return "\n".join(lines)
    results = await service.orders()
    lines = ["أوامر البوت — " + ("تجريبي" if results["mode"] == "paper" else "حقيقي")]
    for item in results["orders"]:
        order = item["order"]
        status = (item["result"] or {}).get("status", item["state"]) if item["state"] == "submitted" else item["state"]
        lines.append(f"{item['ticket']} | {order['side']} {order['symbol']} × {order['qty']} | {status}")
    return "\n".join(lines + (["لا توجد أوامر محفوظة."] if not results["orders"] else ["للتفاصيل: /order ثم الرمز."]))


async def options_command(bridge, job) -> str:
    verb, chat_id = job["request"]["verb"], job["chat_id"]
    engine = bridge.options_paper
    if verb in {"optionson", "optionsoff"}:
        result = engine.set_running(chat_id, verb == "optionson")
        return "محاكاة العقود تعمل — SPY + SPX." if result["running"] else "محاكاة العقود متوقفة."
    if verb == "optionsstatus":
        result = engine.status()
        return ("Options PAPER: " + ("يعمل" if result["running"] else "متوقف") +
                f"\nSPY + SPX | كل {result['interval_seconds']} ثانية" +
                f"\nالمراكز: {len(result['positions'])} | Paper equity: {result['pnl']['paper_equity']:.2f} USD" +
                (f"\nآخر خطأ: {result['last_error']}" if result.get("last_error") else ""))
    if verb == "optionsmodels":
        result = engine.model_status()
        loaded = ", ".join(result.get("available", [])) or "—"
        lines = ["Options AI models: " + ("مفعلة" if result.get("enabled") else "غير مفعلة"),
                 "المتاح: " + loaded, "الجهاز: " + result.get("device", "auto")]
        if result.get("errors"):
            lines.append("أخطاء معزولة: " + "; ".join(f"{k}: {v}" for k, v in result["errors"].items()))
        return "\n".join(lines)
    if verb == "optionpositions":
        rows = engine.positions()
        if not rows: return "لا توجد مراكز عقود Paper مفتوحة."
        return "\n".join(["مراكز العقود Paper:"] + [f"{r['symbol']} {r['option_type'].upper()} {r['strike']:g} exp {r['expiry']} ×{r['contracts']} | premium {r['last_premium']:.2f}" for r in rows])
    if verb == "optiontrades":
        rows = engine.trades()
        if not rows: return "لا توجد صفقات عقود Paper مغلقة."
        return "\n".join(["آخر صفقات العقود Paper:"] + [f"{r['symbol']} {r['option_type'].upper()} {r['strike']:g} | {r['exit_reason']} | P/L {r['pnl']:.2f} USD" for r in rows[:10]])
    p = engine.pnl()
    return f"Options PAPER P/L\nRealized: {p['realized_pnl']:.2f} USD\nUnrealized: {p['unrealized_pnl']:.2f} USD\nEquity: {p['paper_equity']:.2f} USD"


class CommandWorker:
    def __init__(self, bridge, service):
        self.bridge, self.service = bridge, service
        self.last_error = None

    async def once(self) -> bool:
        cfg, store = self.bridge.config, self.bridge.store
        if not store.acquire_lease(self.bridge.owner):
            return False
        job = store.next_prediction_job()
        if not job:
            return False
        if job["chat_id"] not in cfg.allowed_chat_ids or job["chat_id"] <= 0:
            store.finish_prediction_job(job["update_id"], {"status": "recipient_removed"})
            return True
        request, response = job["request"], job["response"]
        if response is None:
            if request.get("error"):
                response = request["error"]
            elif request["verb"] in OPTIONS_COMMANDS:
                try:
                    response = await options_command(self.bridge, job)
                except OptionsPaperError as exc:
                    response = str(exc)
                except Exception:
                    response = "تعذر تنفيذ أمر محاكاة العقود؛ لم يتم إرسال أي أمر حقيقي."
                    self.last_error = "Options paper command failed."
            elif request["verb"] in TRADE_COMMANDS:
                try:
                    response = await trade_command(self.bridge, job)
                except TradeError as exc:
                    response = str(exc)
                except Exception:
                    response = "تعذرت متابعة الطلب. استخدم /orders وتحقق من حساب الوسيط قبل إنشاء أمر بديل."
                    self.last_error = "Trade command failed; the ledger retains submission state."
            elif request["verb"] in {"start", "help"}:
                response = HELP + "\n\n" + OPTIONS_HELP + ("\n\n" + TRADE_HELP if cfg.trading_enabled and job["chat_id"] == cfg.trading_chat_id else "")
            elif request["verb"] == "status":
                ready = bool(cfg.market_api_key) if cfg.market_provider not in {"csv", "alpaca"} else (bool(cfg.market_csv_dir) if cfg.market_provider == "csv" else bool(cfg.alpaca_key_id and cfg.alpaca_secret_key))
                response = (f"أوامر التوقع مفعّلة.\nالمصدر المحدد: {cfg.market_provider}\n"
                            + ("الإعداد موجود؛ نجاح الجلب يُختبر عند طلب توقع." if ready else "مصدر الأسعار يحتاج إعدادًا محليًا: configure-predictions.")
                            + "\nالأسهم الأمريكية اليومية فقط.")
                if cfg.trading_enabled and job["chat_id"] == cfg.trading_chat_id:
                    response += "\nحساب التداول: " + cfg.trading_mode + (" — قبول الأوامر متوقف" if self.bridge.trading.paused() else " — التأكيد مطلوب للأوامر اليدوية")
            else:
                try:
                    response = render(await self.service.predict(request["symbol"], request["horizon"]),
                                      backtest_only=request["verb"] == "backtest")
                except MarketError as exc:
                    response = "تعذر إعداد التوقع: " + str(exc)
                except Exception:
                    response = "تعذر الحساب بسبب خطأ داخلي. راجع حالة الخادم ثم أرسل طلبًا جديدًا."
                    self.last_error = "Prediction computation failed."
            # Freeze exact content BEFORE sending, so restarts reuse the same idempotency payload.
            store.prepare_prediction_reply(job["update_id"], response)
        request_id = str(uuid.uuid5(uuid.NAMESPACE_URL,
                                   f"telegram-prediction:{cfg.expected_bot_username}:{job['chat_id']}:{job['update_id']}"))
        delivery = await self.bridge.send(job["chat_id"], response, request_id,
                                          reply_to=request["message_id"], _command_reply=True)
        store.finish_prediction_job(job["update_id"], delivery)
        return True

    async def run(self):
        while not self.bridge.stop.is_set():
            try:
                worked = await self.once()
                delay = 0.5 if worked else 1
            except Exception:
                self.last_error = "Command processing paused; persisted replies are retained."
                delay = 5
            try:
                await asyncio.wait_for(self.bridge.stop.wait(), timeout=delay)
            except TimeoutError:
                pass
