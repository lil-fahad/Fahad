import pandas as pd


def test_timesfm_examples_keep_raw_scale_and_symbols_separate():
    from heavy_lab.training.datasets import build_timesfm_examples

    rows = []
    for symbol, base in [("SPY", 100.0), ("SPX", 5000.0)]:
        for i, ts in enumerate(pd.date_range("2026-09-08T13:30:00Z", periods=10, freq="5min")):
            rows.append(
                {
                    "timestamp_utc": ts,
                    "symbol": symbol,
                    "open": base + i,
                    "high": base + i + 0.5,
                    "low": base + i - 0.5,
                    "close": base + i,
                    "volume": 1000.0,
                    "session": "regular",
                    "source": "test",
                    "source_revision": "v1",
                }
            )
    frame = pd.DataFrame(rows).sort_values(["symbol", "timestamp_utc"]).reset_index(drop=True)

    examples = build_timesfm_examples(frame, context_length=4, prediction_length=2, stride=2)
    assert examples
    assert {item["symbol"] for item in examples} == {"SPY", "SPX"}
    for item in examples:
        assert len(item["past_values"]) == 4
        assert len(item["future_values"]) == 2
        assert all(isinstance(v, float) for v in item["past_values"] + item["future_values"])
        if item["symbol"] == "SPY":
            assert max(item["past_values"] + item["future_values"]) < 1000
        else:
            assert min(item["past_values"] + item["future_values"]) > 1000
        assert item["forecast_origin_utc"] < item["target_end_utc"]
