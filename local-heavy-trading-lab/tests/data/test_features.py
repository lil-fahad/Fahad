import pandas as pd


def _frame():
    ts = pd.date_range("2026-09-08T13:30:00Z", periods=12, freq="5min")
    close = pd.Series([100, 101, 102, 101, 103, 104, 105, 104, 106, 107, 108, 109], dtype=float)
    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "symbol": "SPY",
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000.0,
            "session": "regular",
            "source": "test",
            "source_revision": "v1",
        }
    )


def test_features_are_past_only_and_stable_when_future_changes():
    from heavy_lab.data.features import build_features

    base = _frame()
    changed = base.copy()
    changed.loc[8:, "close"] = changed.loc[8:, "close"] * 10
    changed.loc[8:, "open"] = changed.loc[8:, "close"]
    changed.loc[8:, "high"] = changed.loc[8:, "close"] + 0.5
    changed.loc[8:, "low"] = changed.loc[8:, "close"] - 0.5

    left = build_features(base)
    right = build_features(changed)

    feature_cols = [c for c in left.columns if c.startswith("feat_")]
    assert feature_cols
    pd.testing.assert_frame_equal(left.loc[:7, feature_cols], right.loc[:7, feature_cols])


def test_feature_builder_never_backfills_from_future():
    from heavy_lab.data.features import build_features

    out = build_features(_frame())
    feature_cols = [c for c in out.columns if c.startswith("feat_")]
    assert out.loc[0, feature_cols].isna().all()
