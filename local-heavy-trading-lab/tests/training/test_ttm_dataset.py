import pandas as pd


def test_ttm_examples_are_chronological_and_do_not_mix_symbols():
    from heavy_lab.training.datasets import build_ttm_examples

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

    examples = build_ttm_examples(frame, context_length=5, prediction_length=2, stride=2)
    assert examples
    assert {example["symbol"] for example in examples} == {"SPY", "SPX"}
    for example in examples:
        assert len(example["past_values"]) == 5
        assert len(example["future_values"]) == 2
        assert all(len(row) == 1 for row in example["past_values"])
        assert all(len(row) == 1 for row in example["future_values"])
        past = [row[0] for row in example["past_values"]]
        future = [row[0] for row in example["future_values"]]
        if example["symbol"] == "SPY":
            assert max(past + future) < 1000
        else:
            assert min(past + future) > 1000
        assert example["forecast_origin_utc"] < example["target_end_utc"]
