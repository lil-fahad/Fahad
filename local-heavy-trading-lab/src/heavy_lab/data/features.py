from __future__ import annotations

import numpy as np
import pandas as pd

from heavy_lab.data.schema import validate_canonical_frame


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    validate_canonical_frame(df)
    out = df.copy()
    groups = out.groupby("symbol", sort=False, group_keys=False)

    out["feat_return_1"] = groups["close"].pct_change(fill_method=None)
    out["feat_log_return_1"] = groups["close"].transform(lambda s: np.log(s).diff())
    out["feat_range_pct"] = (out["high"] - out["low"]) / out["close"]
    out["feat_body_pct"] = (out["close"] - out["open"]) / out["open"]
    out["feat_volume_change_1"] = groups["volume"].pct_change(fill_method=None)
    out["feat_sma_3_ratio"] = groups["close"].transform(lambda s: s.rolling(3, min_periods=3).mean() / s - 1.0)
    out["feat_ema_5_ratio"] = groups["close"].transform(lambda s: s.ewm(span=5, adjust=False, min_periods=2).mean() / s - 1.0)
    out["feat_volatility_5"] = groups["close"].transform(
        lambda s: s.pct_change(fill_method=None).rolling(5, min_periods=5).std()
    )

    # No backfill or centered windows: every feature at row t uses only rows <= t.
    return out
