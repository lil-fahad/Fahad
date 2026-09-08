"""Bounded daily US equity data adapters. No ChatGPT connector credentials are inherited."""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

import httpx

MAX_BARS = 10000
BASES = {"split_adjusted", "unverified", "synthetic"}


class MarketError(ValueError):
    """Only controlled, non-secret messages are exposed to users."""


def symbol_checked(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", symbol):
        raise MarketError("رمز السهم غير صالح. مثال: AAPL أو MSFT أو BRK.B.")
    return symbol


def completed_day(now: datetime | None = None) -> date:
    # Conservative: exclude the current US date until 18:00 Eastern, including early closes.
    now = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York"))
    return now.date() if now.hour >= 18 else now.date() - timedelta(days=1)


@dataclass(frozen=True)
class Bar:
    date: str
    close: float
    raw_close: float


@dataclass(frozen=True)
class Series:
    symbol: str
    source: str
    basis: str
    bars: tuple[Bar, ...]
    retrieved_at: str
    warnings: tuple[str, ...] = ()

    def checked(self, cutoff: date | None = None) -> "Series":
        cutoff = cutoff or completed_day()
        symbol_checked(self.symbol)
        if self.basis not in BASES or not 1 <= len(self.bars) <= MAX_BARS:
            raise MarketError("بيانات الأسعار أو نوع تعديلها غير صالح.")
        seen, valid = set(), []
        for bar in sorted(self.bars, key=lambda x: x.date):
            try:
                day = date.fromisoformat(bar.date)
            except (TypeError, ValueError):
                raise MarketError("تاريخ غير صالح في بيانات الأسعار.") from None
            if bar.date in seen or day.weekday() >= 5:
                raise MarketError("توجد تواريخ مكررة أو شموع غير يومية في بيانات الأسهم.")
            seen.add(bar.date)
            if any(not math.isfinite(x) or x <= 0 for x in (bar.close, bar.raw_close)):
                raise MarketError("الأسعار يجب أن تكون أرقامًا موجبة ومحدودة.")
            if day <= cutoff:
                valid.append(bar)
        if not valid:
            raise MarketError("لا توجد شموع يومية مكتملة.")
        # Never guess whether a large discontinuity is a split, bad data, or a real crash.
        if any(abs(math.log(b.close / a.close)) > math.log(1.8) for a, b in zip(valid, valid[1:])):
            raise MarketError("قفزة سعرية كبيرة تحتاج مراجعة بيانات التجزئة قبل تشغيل النموذج.")
        if any((date.fromisoformat(b.date) - date.fromisoformat(a.date)).days > 10
               for a, b in zip(valid, valid[1:])):
            raise MarketError("يوجد انقطاع كبير في السجل؛ لا يمكن اعتباره جلسات تداول متتالية.")
        return replace(self, bars=tuple(valid))


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=".market-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_csv(path: Path, symbol: str, basis: str = "unverified", *, cutoff: date | None = None) -> Series:
    if path.stat().st_size > 2_000_000:
        raise MarketError("ملف CSV أكبر من الحد المسموح.")
    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            bars = tuple(Bar(row["date"], float(row["close"]), float(row.get("raw_close") or row["close"]))
                         for row in csv.DictReader(source))
    except (KeyError, ValueError, csv.Error):
        raise MarketError("CSV يحتاج date وclose؛ ويمكن إضافة raw_close.") from None
    return Series(symbol_checked(symbol), "CSV supplied locally", basis, bars,
                  datetime.now(timezone.utc).isoformat()).checked(cutoff)


class MarketData:
    def __init__(self, provider: str, api_key: str, cache_dir: Path, csv_dir: Path | None = None,
                 csv_basis: str = "unverified", transport=None, alpaca_key_id: str = "", alpaca_secret_key: str = ""):
        self.provider, self._api_key, self.cache_dir = provider, api_key, cache_dir
        self.csv_dir, self.csv_basis, self.transport = csv_dir, csv_basis, transport
        self._alpaca_key, self._alpaca_secret = alpaca_key_id, alpaca_secret_key
        # Alpha Vantage puts its key in the query string; HTTP debug logs must stay off.
        for name in ("httpx", "httpcore"):
            logging.getLogger(name).setLevel(logging.CRITICAL)

    @staticmethod
    def _json(client, url: str, **kwargs) -> dict:
        try:
            with client.stream("GET", url, **kwargs) as response:
                if response.status_code != 200:
                    descriptions = {401: "مفتاح مصدر البيانات غير صالح.", 402: "رصيد مصدر البيانات أو الاشتراك غير كافٍ.",
                                    403: "الاشتراك لا يسمح بهذه البيانات.", 429: "تم بلوغ حد طلبات مصدر البيانات."}
                    raise MarketError(descriptions.get(response.status_code, "تعذر جلب الأسعار من المصدر."))
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > 10_000_000:
                        raise MarketError("استجابة مصدر الأسعار أكبر من الحد المسموح.")
                    chunks.append(chunk)
                data = json.loads(b"".join(chunks))
        except (httpx.HTTPError, ValueError) as exc:
            if isinstance(exc, MarketError):
                raise
            raise MarketError("تعذر الاتصال بمصدر البيانات أو قراءة استجابته.") from None
        if not isinstance(data, dict):
            raise MarketError("استجابة الأسعار غير صالحة.")
        return data

    def _alpha(self, client, symbol: str, cutoff: date) -> Series:
        data = self._json(client, "https://www.alphavantage.co/query", params={
            "function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": symbol,
            "outputsize": "full", "datatype": "json", "apikey": self._api_key})
        rows = data.get("Time Series (Daily)")
        if not isinstance(rows, dict):
            raise MarketError("Alpha Vantage لم يُرجع سجلًا كاملًا. تحقق من المفتاح والخطة وحد الطلبات؛ Daily Adjusted يتطلب اشتراكًا مناسبًا.")
        meta = data.get("Meta Data", {})
        if str(meta.get("2. Symbol", "")).upper() != symbol:
            raise MarketError("رمز البيانات لا يطابق الرمز المطلوب.")
        if meta.get("5. Time Zone", "US/Eastern") not in {"US/Eastern", "America/New_York"}:
            raise MarketError("هذه النسخة مخصصة للأسهم الأمريكية اليومية.")
        bars, split_factor = [], 1.0
        try:
            for day, row in sorted(rows.items(), reverse=True):
                if date.fromisoformat(day) > cutoff:
                    continue
                close, split = float(row["4. close"]), float(row["8. split coefficient"])
                if not math.isfinite(split) or split <= 0:
                    raise ValueError()
                # Use raw closes and split events, NOT dividend-adjusted total-return prices.
                bars.append(Bar(day, close / split_factor, close))
                split_factor *= split
        except (ValueError, KeyError, TypeError, OverflowError, ZeroDivisionError):
            raise MarketError("بيانات التجزئة أو الإغلاق ناقصة لدى Alpha Vantage.") from None
        return Series(symbol, "Alpha Vantage / TIME_SERIES_DAILY_ADJUSTED", "split_adjusted",
                      tuple(reversed(bars)), datetime.now(timezone.utc).isoformat()).checked(cutoff)

    def _financial(self, client, symbol: str, cutoff: date) -> Series:
        url = "https://api.financialdatasets.ai/prices"
        params = {"ticker": symbol, "interval": "day", "interval_multiplier": 1,
                  "start_date": (cutoff - timedelta(days=365 * 8)).isoformat(), "end_date": cutoff.isoformat()}
        rows, seen = [], set()
        for _ in range(10):
            if url in seen:
                raise MarketError("تكررت صفحة لدى مصدر البيانات.")
            seen.add(url)
            data = self._json(client, url, params=params, headers={"X-API-KEY": self._api_key})
            if data.get("ticker", symbol) != symbol or not isinstance(data.get("prices"), list):
                raise MarketError("Financial Datasets لم يُرجع سجل الأسعار المطلوب.")
            rows.extend(data["prices"])
            if len(rows) > MAX_BARS:
                raise MarketError("سجل الأسعار أكبر من الحد المسموح.")
            next_url = data.get("next_page_url")
            if not next_url:
                break
            parsed = urlsplit(next_url)
            query = parse_qs(parsed.query)
            if (parsed.scheme != "https" or parsed.netloc != "api.financialdatasets.ai"
                    or parsed.path.rstrip("/") != "/prices" or parsed.fragment
                    or query.get("ticker", [symbol]) != [symbol]
                    or query.get("interval", ["day"]) != ["day"]):
                raise MarketError("رفض رابط ترقيم صفحات غير متوقع.")
            url, params = next_url, None
        else:
            raise MarketError("السجل يحتاج صفحات أكثر من الحد المسموح.")
        try:
            bars = tuple(Bar(str(row["time"])[:10], float(row["close"]), float(row["close"])) for row in rows)
        except (KeyError, TypeError, ValueError):
            raise MarketError("بيانات Financial Datasets ناقصة.") from None
        # The documented endpoint does not declare adjustment semantics. Do not invent them.
        return Series(symbol, "Financial Datasets / prices", "unverified", bars,
                      datetime.now(timezone.utc).isoformat(),
                      ("لم يُتحقق من تعديل الأسعار للتجزئة لدى هذا المصدر؛ النتيجة استكشافية.",)).checked(cutoff)

    def _alpaca(self, client, symbol: str, cutoff: date) -> Series:
        params = {"symbols": symbol, "timeframe": "1Day", "adjustment": "split", "feed": "iex",
                  "start": (cutoff - timedelta(days=365 * 8)).isoformat(),
                  "end": cutoff.isoformat() + "T23:59:59Z", "limit": 10000, "sort": "asc"}
        headers = {"APCA-API-KEY-ID": self._alpaca_key, "APCA-API-SECRET-KEY": self._alpaca_secret}
        rows, tokens = [], set()
        for _ in range(10):
            data = self._json(client, "https://data.alpaca.markets/v2/stocks/bars", params=params, headers=headers)
            page = data.get("bars", {}).get(symbol) if isinstance(data.get("bars"), dict) else None
            if not isinstance(page, list):
                raise MarketError("Alpaca لم يُرجع سجلًا يوميًا لهذا السهم.")
            rows.extend(page)
            if len(rows) > MAX_BARS:
                raise MarketError("سجل Alpaca أكبر من الحد المسموح.")
            token = data.get("next_page_token")
            if not token:
                break
            if not isinstance(token, str) or len(token) > 2048 or token in tokens:
                raise MarketError("ترقيم صفحات Alpaca غير صالح.")
            tokens.add(token)
            params["page_token"] = token
        else:
            raise MarketError("سجل Alpaca تجاوز عدد الصفحات المسموح.")
        try:
            parsed = []
            for row in rows:
                stamp = datetime.fromisoformat(row["t"].replace("Z", "+00:00"))
                if stamp.tzinfo is None:
                    raise ValueError("Timezone missing")
                parsed.append(Bar(stamp.astimezone(ZoneInfo("America/New_York")).date().isoformat(),
                                  float(row["c"]), float(row["c"])))
            bars = tuple(parsed)
        except (KeyError, TypeError, ValueError, AttributeError):
            raise MarketError("استجابة Alpaca تحتوي شموعًا غير صالحة.") from None
        return Series(symbol, "Alpaca IEX daily bars", "split_adjusted", bars,
                      datetime.now(timezone.utc).isoformat(),
                      ("المصدر IEX فقط؛ الإغلاق والحجم قد يختلفان عن تجميع جميع الأسواق.",)).checked(cutoff)

    def load(self, symbol: str) -> Series:
        symbol = symbol_checked(symbol)
        if self.provider == "csv":
            if not self.csv_dir:
                raise MarketError("لم يُحدد مجلد CSV في الإعدادات.")
            try:
                return read_csv(self.csv_dir / (symbol + ".csv"), symbol, self.csv_basis)
            except OSError:
                raise MarketError("لا يوجد ملف أسعار محلي لهذا الرمز.") from None
        path = self.cache_dir / (self.provider + "-" + symbol + ".json")
        cached = None
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                raw["bars"] = tuple(Bar(**row) for row in raw["bars"])
                raw["warnings"] = tuple(raw.get("warnings", []))
                cached = Series(**raw).checked()
                if cached.symbol != symbol:
                    cached = None
                elif 0 <= (datetime.now(timezone.utc) - datetime.fromisoformat(cached.retrieved_at)).total_seconds() < 21600:
                    return cached
            except (ValueError, KeyError, TypeError, OSError):
                cached = None
        if self.provider == "alpaca" and not (self._alpaca_key and self._alpaca_secret):
            raise MarketError("أضف مفاتيح Alpaca محليًا باستخدام configure-trading.")
        if self.provider != "alpaca" and not self._api_key:
            raise MarketError("أضف مفتاح مصدر البيانات محليًا بأمر configure-predictions. مفاتيح إضافات ChatGPT لا تنتقل تلقائيًا إلى البوت.")
        try:
            with httpx.Client(timeout=20, follow_redirects=False, trust_env=False, transport=self.transport) as client:
                if self.provider == "alpha_vantage":
                    series = self._alpha(client, symbol, completed_day())
                elif self.provider == "financial_datasets":
                    series = self._financial(client, symbol, completed_day())
                elif self.provider == "alpaca":
                    series = self._alpaca(client, symbol, completed_day())
                else:
                    raise MarketError("مصدر البيانات غير مدعوم.")
            atomic_json(path, asdict(series))
            return series
        except MarketError:
            if cached:
                return replace(cached, warnings=cached.warnings + ("تعذر تحديث المصدر؛ استُخدمت نسخة مخزنة بتاريخها المعلن.",))
            raise
