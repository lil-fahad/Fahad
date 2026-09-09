import json
from pathlib import Path

from typer.testing import CliRunner


def test_training_ready_cli_reports_missing_requirements(tmp_path: Path):
    from heavy_lab.cli import app

    result = CliRunner().invoke(app, ["training-ready", "--root", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.stdout
    report = json.loads(result.stdout)
    assert report["ready"] is False
    assert report["missing"]
    assert "train_parquet" in report["missing"]
    assert "finbert_train_jsonl" in report["missing"]
