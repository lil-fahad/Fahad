from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner


def test_train_all_uses_manifest_and_profile(monkeypatch, tmp_path: Path):
    from heavy_lab.cli import app
    import heavy_lab.training.campaign as campaign

    manifest = tmp_path / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "train_parquet": "train.parquet",
                "validation_parquet": "validation.parquet",
                "finbert_train_jsonl": "finbert-train.jsonl",
                "finbert_validation_jsonl": "finbert-validation.jsonl",
                "profile": "smoke",
            }
        ),
        encoding="utf-8",
    )

    seen = {}

    def fake_run_training_campaign(*, root, manifest_path, profile=None):
        seen["root"] = Path(root)
        seen["manifest_path"] = Path(manifest_path)
        seen["profile"] = profile
        return {
            "profile": profile or "smoke",
            "models": {
                "ttm": {"status": "COMPLETED", "run_id": "run-ttm"},
                "timesfm25": {"status": "COMPLETED", "run_id": "run-timesfm"},
                "chronos2": {"status": "COMPLETED", "run_id": "run-chronos"},
                "kronos": {"status": "COMPLETED", "run_id": "run-kronos"},
                "finbert": {"status": "COMPLETED", "run_id": "run-finbert"},
            },
        }

    monkeypatch.setattr(campaign, "run_training_campaign", fake_run_training_campaign)

    result = CliRunner().invoke(
        app,
        [
            "train-all",
            "--root",
            str(tmp_path),
            "--manifest",
            str(manifest),
            "--profile",
            "max",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["profile"] == "max"
    assert list(payload["models"]) == ["ttm", "timesfm25", "chronos2", "kronos", "finbert"]
    assert seen == {
        "root": tmp_path,
        "manifest_path": manifest,
        "profile": "max",
    }
