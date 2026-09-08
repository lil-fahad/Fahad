import json
from pathlib import Path

import pandas as pd


def test_nasdaq_bar_parser_normalizes_to_utc():
    from heavy_lab.data.providers.nasdaq import parse_bars

    payload = {
        "SPY": [
            {"t": "2026-09-08T09:30:00.000", "o": 100.0, "h": 101.0, "l": 99.5, "c": 100.5, "v": 1000},
            {"t": "2026-09-08T09:35:00.000", "o": 100.5, "h": 101.2, "l": 100.1, "c": 101.0, "v": 1200},
        ]
    }
    df = parse_bars(payload, source_revision="nasdaq-v2")
    assert list(df["symbol"]) == ["SPY", "SPY"]
    assert str(df["timestamp_utc"].dt.tz) == "UTC"
    assert df.iloc[0]["timestamp_utc"] == pd.Timestamp("2026-09-08T13:30:00Z")
    assert list(df["session"]) == ["regular", "regular"]


def test_raw_payload_is_content_addressed_and_not_overwritten(tmp_path: Path):
    from heavy_lab.data.ingest import write_raw_payload

    payload = {"SPY": [{"t": "2026-09-08T09:30:00.000", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}]}
    first = write_raw_payload(tmp_path, source="nasdaq", symbol="SPY", interval="5m", payload=payload)
    before = first.stat().st_mtime_ns
    second = write_raw_payload(tmp_path, source="nasdaq", symbol="SPY", interval="5m", payload=payload)
    assert first == second
    assert second.stat().st_mtime_ns == before
    assert json.loads(second.read_text(encoding="utf-8")) == payload
    assert second.stem == __import__("hashlib").sha256(second.read_bytes()).hexdigest()
