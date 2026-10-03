from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner


def _canonical_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-02 14:30", periods=200, freq="5min", tz="UTC")
    rows = []
    for symbol, base in (("SPY", 500.0), ("SPX", 6000.0)):
        for i, ts in enumerate(timestamps):
            price = base + i * 0.1
            rows.append(
                {
                    "timestamp_utc": ts,
                    "symbol": symbol,
                    "open": price,
                    "high": price + 0.2,
                    "low": price - 0.2,
                    "close": price + 0.05,
                    "volume": 1000.0 + i,
                    "session": "regular",
                    "source": "fixture",
                    "source_revision": "v1",
                }
            )
    return pd.DataFrame(rows).sort_values(["symbol", "timestamp_utc"], kind="stable").reset_index(drop=True)


def test_prepare_writes_chronological_campaign_splits_and_manifest(tmp_path: Path):
    from heavy_lab.cli import app

    source = tmp_path / "canonical.parquet"
    _canonical_frame().to_parquet(source, index=False)

    result = CliRunner().invoke(
        app,
        [
            "prepare",
            "--profile",
            "intraday-v1",
            "--root",
            str(tmp_path),
            "--input",
            str(source),
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)

    train_path = Path(payload["train_parquet"])
    validation_path = Path(payload["validation_parquet"])
    test_path = Path(payload["test_parquet"])
    manifest_path = Path(payload["manifest"])

    assert train_path.is_file()
    assert validation_path.is_file()
    assert test_path.is_file()
    assert manifest_path.is_file()

    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(validation_path)
    test = pd.read_parquet(test_path)

    assert train["timestamp_utc"].max() < validation["timestamp_utc"].min()
    assert validation["timestamp_utc"].max() < test["timestamp_utc"].min()
    assert set(train["symbol"]) == {"SPY", "SPX"}
    assert set(validation["symbol"]) == {"SPY", "SPX"}
    assert set(test["symbol"]) == {"SPY", "SPX"}

    unique = _canonical_frame()["timestamp_utc"].drop_duplicates().sort_values().reset_index(drop=True)
    train_last_pos = unique[unique == train["timestamp_utc"].max()].index[0]
    val_first_pos = unique[unique == validation["timestamp_utc"].min()].index[0]
    val_last_pos = unique[unique == validation["timestamp_utc"].max()].index[0]
    test_first_pos = unique[unique == test["timestamp_utc"].min()].index[0]

    assert val_first_pos - train_last_pos - 1 == 12
    assert test_first_pos - val_last_pos - 1 == 12

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["profile"] == "intraday-v1"
    assert manifest["horizon"] == 12
    assert manifest["embargo"] == 12
    assert manifest["locked_test"] is True
