from __future__ import annotations

import pandas as pd


def _frame() -> pd.DataFrame:
    ts = pd.date_range("2026-01-02 14:30", periods=5, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "symbol": ["SPY"] * 5,
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, 101, 102, 103, 104],
            "volume": [1000, 1100, 1200, 1300, 1400],
            "session": ["regular"] * 5,
            "source": ["fixture"] * 5,
            "source_revision": ["v1"] * 5,
        }
    )


def test_targets_are_separate_future_returns_with_tail_nan():
    from heavy_lab.data.targets import build_targets

    targets = build_targets(_frame(), horizons=[1, 2])

    assert list(targets.columns) == [
        "timestamp_utc",
        "symbol",
        "target_return_h1",
        "target_return_h2",
    ]
    assert targets.loc[0, "target_return_h1"] == 0.01
    assert targets.loc[0, "target_return_h2"] == 0.02
    assert pd.isna(targets.loc[4, "target_return_h1"])
    assert pd.isna(targets.loc[3, "target_return_h2"])
    assert not any(name.startswith("target_") for name in _frame().columns)
