from __future__ import annotations

import json

from typer.testing import CliRunner


def test_ingest_dry_run_is_offline_and_deterministic(tmp_path):
    from heavy_lab.cli import app

    result = CliRunner().invoke(
        app,
        ["ingest", "--profile", "spy-spx", "--dry-run", "--root", str(tmp_path)],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload == {
        "profile": "spy-spx",
        "provider": "nasdaq",
        "symbols": ["SPY", "SPX"],
        "interval": "5m",
        "dry_run": True,
        "requires": ["NASDAQ_CLIENT_ID", "NASDAQ_CLIENT_SECRET", "NASDAQ_BASE_URL"],
    }
    assert not (tmp_path / "data" / "raw").exists()


def test_prepare_dry_run_describes_leakage_safe_profile(tmp_path):
    from heavy_lab.cli import app

    result = CliRunner().invoke(
        app,
        ["prepare", "--profile", "intraday-v1", "--dry-run", "--root", str(tmp_path)],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["profile"] == "intraday-v1"
    assert payload["dry_run"] is True
    assert payload["horizons"] == [1, 6, 12]
    assert payload["normalization"] == "fold-local-standard"
    assert payload["timesfm_normalization"] == "none"
    assert payload["split"]["horizon"] == 12
    assert payload["split"]["embargo"] == 12
    assert payload["requires_input"] is True
    assert not (tmp_path / "data" / "splits").exists()
