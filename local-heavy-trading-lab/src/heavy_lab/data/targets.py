from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from heavy_lab.data.schema import validate_canonical_frame


def build_targets(df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    """Build future-return targets in a separate table.

    Targets intentionally live outside the feature frame. Each target at row t
    uses close[t + horizon] / close[t] - 1 and therefore remains NaN where the
    requested future observation is unavailable.
    """
    validate_canonical_frame(df)
    normalized = tuple(int(h) for h in horizons)
    if not normalized or any(h <= 0 for h in normalized):
        raise ValueError("horizons must contain positive integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("horizons must not contain duplicates")

    out = df[["timestamp_utc", "symbol"]].copy()
    for horizon in normalized:
        future_close = df.groupby("symbol", sort=False)["close"].shift(-horizon)
        out[f"target_return_h{horizon}"] = future_close / df["close"] - 1.0
    return out
