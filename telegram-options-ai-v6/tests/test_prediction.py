import asyncio
import json
import math
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest

from telegram_bridge.demo import demo_series
from telegram_bridge.market import Bar, MarketData, MarketError, Series, completed_day, read_csv
from telegram_bridge.prediction import at_origin, evaluate, features, render


@pytest.fixture(scope="module")
def history():
    return demo_series()


@pytest.fixture(scope="module")
def report(history):
    return evaluate(history, 5)


def test_future_mutations_cannot_change_past_forecast(history):
    closes = [bar.close for bar in history.bars]
    original = features(closes)
    changed = closes[:801] + [price * (1 + (i % 10) / 10) for i, price in enumerate(closes[801:])]
    mutated = features(changed)
    assert original[800] == mutated[800]
    first = at_origin(closes, original, 800, 20)
    assert first == at_origin(changed, mutated, 800, 20)
    assert first["latest_training_label_index"] < 800
    # The first feature is independently calculable from information available at the origin.
    assert original[800][0] == pytest.approx(math.log(closes[800] / closes[799]))


def test_walk_forward_metrics_are_honest_and_reproducible(report):
    rows = report["backtest"]["rows"]
    assert len(rows) == 40
    assert all(row["training_labels_end"] < row["origin"] < row["target_date"] for row in rows)
    assert all(a["target_date"] <= b["origin"] for a, b in zip(rows, rows[1:]))
    expected = sum((r["probability_up"] - (r["actual_return"] > 0)) ** 2 for r in rows) / len(rows)
    assert report["backtest"]["brier"] == pytest.approx(expected)
    assert report["synthetic"] and report["signal"] == "abstain"
    assert report["backtest"]["brier"] >= report["backtest"]["baseline_brier"]
    assert "النموذج لم يتفوق" in render(report)
    assert "ليست بيانات سوق" in render(report)
    assert 0 < report["probability_up"] < 1
    lower, upper = report["price_interval"]
    assert 0 < lower <= report["forecast_price"] <= upper
    assert "not calibrated" in report["probability_kind"]
    assert len(render(report).encode("utf-16-le")) // 2 < 4096


def test_stale_data_never_produces_a_current_price(history):
    last_day = date.fromisoformat(history.bars[-1].date)
    result = evaluate(history, 20, today=last_day + timedelta(days=30))
    assert result["stale"] and result["forecast_price"] is None and result["price_interval"] is None
    assert result["forecast_return_pct"] is None and result["signal"] == "abstain"
    assert "لا يوجد توقع حالي" in render(result)


def test_bad_short_and_discontinuous_data_fail(history):
    with pytest.raises(MarketError, match="600"):
        evaluate(replace(history, bars=history.bars[:100]), 5)
    for horizon in (0, 2, 50, True):
        with pytest.raises(MarketError):
            evaluate(history, horizon)
    for bad in (float("nan"), float("inf"), 0, -1):
        with pytest.raises(MarketError):
            replace(history, bars=(replace(history.bars[0], close=bad),)).checked()
    with pytest.raises(MarketError, match="مكررة"):
        replace(history, bars=(history.bars[0], history.bars[0])).checked()
    with pytest.raises(MarketError, match="قفزة"):
        replace(history, bars=(history.bars[0], replace(history.bars[1], close=history.bars[0].close / 4))).checked()


def test_completed_us_day_and_csv_validation(tmp_path):
    assert completed_day(datetime(2026, 8, 5, 20, 0, tzinfo=timezone.utc)) == date(2026, 8, 4)
    assert completed_day(datetime(2026, 8, 5, 22, 0, tzinfo=timezone.utc)) == date(2026, 8, 5)
    csv = tmp_path / "AAPL.csv"
    csv.write_text("date,close\n2026-08-03,10\n2026-08-04,11\n2026-08-05,12\n")
    series = read_csv(csv, "AAPL", cutoff=date(2026, 8, 4))
    assert len(series.bars) == 2 and series.basis == "unverified"
    csv.write_text("date,close\n2026-08-03,NaN\n")
    with pytest.raises(MarketError):
        read_csv(csv, "AAPL")


def test_alpha_uses_splits_not_dividend_adjusted_close(tmp_path):
    data = {"Meta Data": {"2. Symbol": "AAPL", "5. Time Zone": "US/Eastern"}, "Time Series (Daily)": {
        "2026-08-05": {"4. close": "52", "5. adjusted close": "40", "8. split coefficient": "1"},
        "2026-08-04": {"4. close": "50", "5. adjusted close": "39", "8. split coefficient": "2"},
        "2026-08-03": {"4. close": "98", "5. adjusted close": "38", "8. split coefficient": "1"}}}
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=data)
    market = MarketData("alpha_vantage", "synthetic-api-key", tmp_path, transport=httpx.MockTransport(handle))
    series = market.load("AAPL")
    assert [bar.close for bar in series.bars] == [49, 50, 52]
    assert [bar.raw_close for bar in series.bars] == [98, 50, 52]
    assert series.basis == "split_adjusted"
    assert "synthetic-api-key" not in repr(series)
    market.load("AAPL")
    assert len(requests) == 1  # Shared source cache across horizons prevents duplicate paid requests.
    assert requests[0].url.params["outputsize"] == "full"
    assert requests[0].url.params["function"] == "TIME_SERIES_DAILY_ADJUSTED"


def test_provider_failures_do_not_expose_keys(tmp_path):
    secret = "test-api-key-that-must-remain-private"
    def timeout(request):
        raise httpx.ReadTimeout("secret=" + secret, request=request)
    market = MarketData("alpha_vantage", secret, tmp_path, transport=httpx.MockTransport(timeout))
    with pytest.raises(MarketError) as error:
        market.load("AAPL")
    assert secret not in str(error.value)
    for status in (401, 402, 403, 429, 500):
        transport = httpx.MockTransport(lambda _: httpx.Response(status, json={"error": secret}))
        with pytest.raises(MarketError) as error:
            MarketData("financial_datasets", secret, tmp_path, transport=transport).load("AAPL")
        assert secret not in str(error.value)
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"Information": secret}))
    with pytest.raises(MarketError):
        MarketData("alpha_vantage", secret, tmp_path, transport=transport).load("AAPL")


def test_financial_pagination_and_unknown_adjustments(tmp_path):
    requests = []
    def handle(request):
        requests.append(request)
        row = {"time": "2026-08-03", "close": 10}
        if len(requests) == 1:
            return httpx.Response(200, json={"ticker": "AAPL", "prices": [row],
                "next_page_url": "https://api.financialdatasets.ai/prices?ticker=AAPL&interval=day&cursor=next"})
        return httpx.Response(200, json={"ticker": "AAPL", "prices": [{**row, "time": "2026-08-04", "close": 11}]})
    market = MarketData("financial_datasets", "synthetic-api-key", tmp_path, transport=httpx.MockTransport(handle))
    series = market.load("AAPL")
    assert len(series.bars) == 2 and series.basis == "unverified" and series.warnings
    assert all(request.headers["X-API-KEY"] == "synthetic-api-key" for request in requests)


def test_pagination_never_forwards_credentials_to_other_hosts(tmp_path):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={"prices": [], "next_page_url": "https://attacker.invalid/prices"})
    market = MarketData("financial_datasets", "synthetic-api-key", tmp_path, transport=httpx.MockTransport(handle))
    with pytest.raises(MarketError, match="ترقيم"):
        market.load("AAPL")
    assert len(calls) == 1
    for symbol in ("../../config", "AAPL?api_key=secret", "AAPL USD", "https://example.com"):
        with pytest.raises(MarketError):
            market.load(symbol)


def test_unverified_data_cannot_get_approved_signal(history):
    result = evaluate(replace(history, basis="unverified", source="Unknown corporate actions"), 5)
    assert result["signal"] == "abstain" and "تعديل التجزئة غير متحقق" in result["abstain_reasons"]
