from datetime import datetime, timezone

import pandas as pd
import pytest

from heavy_lab.data.schema import REQUIRED_BAR_COLUMNS, validate_canonical_frame
from heavy_lab.data.provenance import DatasetManifest


def _valid_frame():
    return pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2026-09-08T14:30:00Z", "2026-09-08T14:35:00Z"], utc=True),
        "symbol": ["SPY", "SPY"],
        "open": [100.0, 100.5],
        "high": [101.0, 101.2],
        "low": [99.5, 100.1],
        "close": [100.5, 101.0],
        "volume": [1000.0, 1200.0],
        "session": ["regular", "regular"],
        "source": ["fixture", "fixture"],
        "source_revision": ["abc", "abc"],
    })


def test_required_columns_are_stable():
    assert REQUIRED_BAR_COLUMNS == (
        "timestamp_utc", "symbol", "open", "high", "low", "close", "volume",
        "session", "source", "source_revision",
    )


def test_valid_frame_passes():
    validate_canonical_frame(_valid_frame())


def test_future_or_duplicate_shape_is_rejected():
    df = _valid_frame()
    df.loc[1, "timestamp_utc"] = df.loc[0, "timestamp_utc"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_canonical_frame(df)


def test_non_utc_timestamp_is_rejected():
    df = _valid_frame()
    df["timestamp_utc"] = df["timestamp_utc"].dt.tz_convert("America/New_York")
    with pytest.raises(ValueError, match="UTC"):
        validate_canonical_frame(df)


def test_manifest_tracks_raw_hash_and_time_bounds():
    manifest = DatasetManifest(
        dataset_id="spy-5m-fixture",
        source="fixture",
        source_revision="abc",
        raw_sha256="0" * 64,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        rows=2,
        symbols=("SPY",),
        interval="5m",
        start_utc="2026-09-08T14:30:00+00:00",
        end_utc="2026-09-08T14:35:00+00:00",
    )
    assert manifest.raw_sha256 == "0" * 64
    assert manifest.rows == 2
