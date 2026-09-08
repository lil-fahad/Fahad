from __future__ import annotations

from datetime import datetime

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator


REQUIRED_BAR_COLUMNS = (
    "timestamp_utc",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "session",
    "source",
    "source_revision",
)


class CanonicalBar(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp_utc: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    session: str
    source: str
    source_revision: str

    @field_validator("symbol", "session", "source", "source_revision")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("value must be non-empty")
        return value


def validate_canonical_frame(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_BAR_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing canonical columns: {missing}")
    if df.empty:
        raise ValueError("canonical frame is empty")

    timestamps = df["timestamp_utc"]
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        raise ValueError("timestamp_utc must be datetime64")
    timezone = getattr(timestamps.dt, "tz", None)
    if timezone is None or str(timezone).upper() != "UTC":
        raise ValueError("timestamp_utc must use UTC timezone")

    if df.duplicated(subset=["symbol", "timestamp_utc"]).any():
        raise ValueError("duplicate symbol/timestamp rows are not allowed")

    for column in ("open", "high", "low", "close", "volume"):
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"{column} must be numeric")
        if df[column].isna().any():
            raise ValueError(f"{column} contains NaN")

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (df["volume"] < 0).any():
        raise ValueError("volume must be non-negative")

    max_body = df[["open", "close", "low"]].max(axis=1)
    min_body = df[["open", "close", "high"]].min(axis=1)
    if (df["high"] < max_body).any():
        raise ValueError("high is below another OHLC value")
    if (df["low"] > min_body).any():
        raise ValueError("low is above another OHLC value")

    for column in ("symbol", "session", "source", "source_revision"):
        if df[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} contains empty values")

    ordered = df.sort_values(["symbol", "timestamp_utc"], kind="stable").index
    if not ordered.equals(df.index):
        raise ValueError("canonical rows must be ordered by symbol and timestamp_utc")
