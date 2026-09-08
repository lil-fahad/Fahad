"""Interpretable historical-analogue forecasts with purged chronological evaluation.

No tuned-on-test parameters, future features, LLM probabilities or trading execution.
The empirical probability is an estimate, not a calibrated claim of confidence.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import statistics as stats
from collections import OrderedDict
from datetime import date, datetime, timezone

from .market import MarketData, MarketError, Series, symbol_checked

MODEL_VERSION = "historical-analogues-v1"
HORIZONS = (1, 5, 20)
LOOKBACK = 60
MIN_BARS = 600


def quantile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def features(closes: list[float]) -> dict[int, tuple[float, ...]]:
    result = {}
    logs = [math.log(x) for x in closes]
    returns = [0.0] + [b - a for a, b in zip(logs, logs[1:])]
    for i in range(LOOKBACK, len(closes)):
        sma20, sma60 = stats.fmean(closes[i-19:i+1]), stats.fmean(closes[i-59:i+1])
        changes = returns[i-13:i+1]
        up, down = sum(max(x, 0) for x in changes), sum(max(-x, 0) for x in changes)
        rsi = up / (up + down) if up + down else 0.5
        result[i] = tuple(logs[i] - logs[i-n] for n in (1, 5, 20, 60)) + (
            stats.pstdev(returns[i-19:i+1]), math.log(closes[i] / sma20), math.log(sma20 / sma60), rsi)
    return result


def at_origin(closes, matrix, origin: int, horizon: int) -> dict:
    # STRICT purge: every training label finishes before the prediction origin.
    candidates = list(range(LOOKBACK, origin - horizon))
    if len(candidates) < 160:
        raise MarketError("التاريخ السابق لهذه النقطة غير كافٍ لتقدير مستقل.")
    columns = list(zip(*(matrix[i] for i in candidates)))
    scales = [max(stats.pstdev(col), 1e-8) for col in columns]
    current = matrix[origin]
    ranked = sorted(candidates, key=lambda i: (
        sum(((a - b) / s) ** 2 for a, b, s in zip(matrix[i], current, scales)), -i))
    selected = []
    for i in ranked:
        if all(abs(i - j) >= horizon for j in selected):
            selected.append(i)
            if len(selected) == 40:
                break
    if len(selected) < 8:
        raise MarketError("عدد الحالات التاريخية غير المتداخلة غير كافٍ لهذا الأفق.")
    outcomes = [math.log(closes[i+horizon] / closes[i]) for i in selected]
    # Non-overlapping past windows for the historical-frequency baseline.
    baseline = [math.log(closes[i+horizon] / closes[i]) for i in candidates[::horizon]]
    prior_up = (sum(x > 0 for x in baseline) + 1) / (len(baseline) + 2)
    probability = (sum(x > 0 for x in outcomes) + 5 * prior_up) / (len(outcomes) + 5)
    return {"probability_up": probability, "median_log_return": stats.median(outcomes),
            "lower_log_return": quantile(outcomes, 0.1), "upper_log_return": quantile(outcomes, 0.9),
            "analog_count": len(selected), "baseline_probability_up": prior_up,
            "latest_training_label_index": max(i+horizon for i in candidates)}


def evaluate(series: Series, horizon: int = 5, *, today: date | None = None) -> dict:
    if type(horizon) is not int or horizon not in HORIZONS:
        raise MarketError("الأفق المدعوم هو 1 أو 5 أو 20 جلسة تداول.")
    series = series.checked()
    if len(series.bars) < MIN_BARS:
        raise MarketError(f"يلزم {MIN_BARS} إغلاق يومي على الأقل؛ المتاح {len(series.bars)} فقط.")
    # Fixed maximum computational window; older rows remain available in source cache.
    bars = series.bars[-2000:]
    closes = [b.close for b in bars]
    matrix = features(closes)
    last_origin = len(bars) - 1 - horizon
    start = max(260, last_origin - max(200, 40 * horizon) + 1)
    rows = []
    for origin in range(start, last_origin + 1, horizon):
        fit = at_origin(closes, matrix, origin, horizon)
        actual = math.log(closes[origin+horizon] / closes[origin])
        rows.append({"origin": bars[origin].date, "target_date": bars[origin+horizon].date,
                     "training_labels_end": bars[fit["latest_training_label_index"]].date,
                     "probability_up": fit["probability_up"], "baseline_probability_up": fit["baseline_probability_up"],
                     "predicted_return": math.expm1(fit["median_log_return"]), "actual_return": math.expm1(actual),
                     "covered": fit["lower_log_return"] <= actual <= fit["upper_log_return"]})
    n = len(rows)
    brier = stats.fmean((r["probability_up"] - (r["actual_return"] > 0)) ** 2 for r in rows)
    base_brier = stats.fmean((r["baseline_probability_up"] - (r["actual_return"] > 0)) ** 2 for r in rows)
    mae = stats.fmean(abs(r["predicted_return"] - r["actual_return"]) for r in rows)
    base_mae = stats.fmean(abs(r["actual_return"]) for r in rows)
    coverage = stats.fmean(r["covered"] for r in rows)
    fit = at_origin(closes, matrix, len(closes)-1, horizon)
    age = ((today or datetime.now(timezone.utc).date()) - date.fromisoformat(bars[-1].date)).days
    warnings = list(series.warnings)
    reasons = []
    if series.basis != "split_adjusted":
        reasons.append("بيانات محاكاة" if series.basis == "synthetic" else "تعديل التجزئة غير متحقق")
    if age > 7:
        reasons.append("البيانات قديمة")
    elif age > 4:
        warnings.append("آخر إغلاق أقدم من أربعة أيام تقويمية؛ راجع تحديث المصدر والعطلات.")
    if n < 40:
        reasons.append("عينة الاختبار أقل من 40 فترة غير متداخلة")
    if brier >= base_brier or mae >= base_mae:
        reasons.append("النموذج لم يتفوق على المقارنة البسيطة في المقياسين")
    if coverage < 0.70:
        reasons.append("التغطية التاريخية للنطاق أقل من 70%")
    if 0.45 <= fit["probability_up"] <= 0.55:
        reasons.append("احتمال الاتجاه قريب من التعادل")
    signal = "abstain" if reasons else ("up" if fit["probability_up"] > 0.5 else "down")
    current = bars[-1].raw_close
    checksum = hashlib.sha256("\n".join(f"{b.date},{b.close:.12g}" for b in bars).encode()).hexdigest()
    return {"model_version": MODEL_VERSION, "symbol": series.symbol, "horizon_sessions": horizon,
            "source": series.source, "basis": series.basis, "currency": "USD", "as_of": bars[-1].date,
            "retrieved_at": series.retrieved_at, "data_age_calendar_days": age, "bars_used": len(bars),
            "data_sha256": checksum, "synthetic": series.basis == "synthetic", "stale": age > 7,
            "signal": signal, "abstain_reasons": reasons, "warnings": warnings,
            "last_close": current, "probability_up": fit["probability_up"],
            "probability_kind": "smoothed empirical frequency; not calibrated or guaranteed",
            "analog_count": fit["analog_count"],
            "forecast_price": None if age > 7 else current * math.exp(fit["median_log_return"]),
            "forecast_return_pct": None if age > 7 else 100 * math.expm1(fit["median_log_return"]),
            "price_interval": None if age > 7 else [current * math.exp(fit[key]) for key in ("lower_log_return", "upper_log_return")],
            "interval_nominal_coverage": 0.8, "interval_kind": "historical analogue 10th–90th percentiles; no coverage guarantee",
            "backtest": {"n": n, "start": rows[0]["origin"], "end": rows[-1]["target_date"],
                         "direction_accuracy": stats.fmean((r["probability_up"] > 0.5) == (r["actual_return"] > 0) for r in rows),
                         "baseline_direction_accuracy": stats.fmean((r["baseline_probability_up"] > 0.5) == (r["actual_return"] > 0) for r in rows),
                         "brier": brier, "baseline_brier": base_brier, "mae_pct": 100 * mae,
                         "baseline_mae_pct": 100 * base_mae, "interval_coverage": coverage,
                         "method": "expanding-window, purged labels, non-overlapping test outcomes",
                         "rows": rows}}


def render(result: dict, backtest_only: bool = False) -> str:
    bt = result["backtest"]
    lines = [("تجربة محاكاة — ليست بيانات سوق\n" if result["synthetic"] else "") +
             f"{result['symbol']} | {result['horizon_sessions']} جلسات تداول",
             f"آخر إغلاق: {result['last_close']:.2f} USD — {result['as_of']}",
             f"المصدر: {result['source']}"]
    if not backtest_only:
        if result["stale"]:
            lines.append("لا يوجد توقع حالي: السجل قديم؛ حدّث مصدر البيانات.")
        else:
            low, high = result["price_interval"]
            lines += [f"احتمال الصعود التقديري: {result['probability_up']:.0%} ({result['analog_count']} حالة مشابهة)",
                      f"وسيط السعر المتوقع: {result['forecast_price']:.2f} USD ({result['forecast_return_pct']:+.2f}%)",
                      f"النطاق التاريخي 80%: {low:.2f} – {high:.2f} USD"]
    lines += [f"اختبار متدرج: {bt['n']} فترة | {bt['start']} إلى {bt['end']}",
              f"دقة الاتجاه: {bt['direction_accuracy']:.1%} | مقارنة تاريخية: {bt['baseline_direction_accuracy']:.1%}",
              f"Brier (الأقل أفضل): {bt['brier']:.3f} | المقارنة: {bt['baseline_brier']:.3f}",
              f"خطأ العائد: {bt['mae_pct']:.2f} نقطة مئوية | ثبات السعر: {bt['baseline_mae_pct']:.2f}",
              f"تغطية النطاق الفعلية في الاختبار: {bt['interval_coverage']:.1%}"]
    if result["abstain_reasons"]:
        lines.append("لا إشارة معتمدة: " + "؛ ".join(result["abstain_reasons"]) + ".")
    else:
        lines.append("الميل الإحصائي: " + ("صعود" if result["signal"] == "up" else "هبوط") + "؛ يحتاج متابعة فعلية.")
    lines += result["warnings"]
    lines.append("تقدير تجريبي قابل للخطأ؛ النسبة ليست ضمانًا. الاختبار يقيس التوقع، ولا يحسب أرباح تداول أو رسومًا.")
    return "\n".join(lines)


class PredictionService:
    def __init__(self, market: MarketData):
        self.market = market
        self.lock = asyncio.Lock()
        self.cache = OrderedDict()

    async def predict(self, symbol: str, horizon: int = 5) -> dict:
        symbol = symbol_checked(symbol)
        if type(horizon) is not int or horizon not in HORIZONS:
            raise MarketError("الأفق المدعوم هو 1 أو 5 أو 20 جلسة تداول.")
        if self.lock.locked():
            raise MarketError("يجري حساب توقع آخر؛ أعد الطلب بعد قليل.")
        async with self.lock:
            series = await asyncio.to_thread(self.market.load, symbol)
            # Include the complete series, date and warnings: revisions/staleness invalidate cached results.
            key = (series, horizon, datetime.now(timezone.utc).date())
            if key not in self.cache:
                self.cache[key] = await asyncio.to_thread(evaluate, series, horizon)
                if len(self.cache) > 12:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(key)
            return self.cache[key]
