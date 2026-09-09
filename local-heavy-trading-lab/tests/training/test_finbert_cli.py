import json
from pathlib import Path

from typer.testing import CliRunner


def test_finbert_cli_reads_jsonl_and_calls_real_launcher_contract(tmp_path: Path, monkeypatch):
    from heavy_lab.cli import app
    from heavy_lab.training import launch

    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_rows = [
        {"document_id": "a", "timestamp_utc": "2026-01-01T10:00:00Z", "text": "profits rose", "label": 2},
        {"document_id": "b", "timestamp_utc": "2026-01-01T11:00:00Z", "text": "loss widened", "label": 0},
    ]
    validation_rows = [
        {"document_id": "c", "timestamp_utc": "2026-01-02T10:00:00Z", "text": "outlook stable", "label": 1},
    ]
    train_path.write_text("\n".join(json.dumps(row) for row in train_rows) + "\n", encoding="utf-8")
    validation_path.write_text("\n".join(json.dumps(row) for row in validation_rows) + "\n", encoding="utf-8")

    calls = {}

    class Result:
        run_id = "finbert-test-run"
        status = "COMPLETED"
        checkpoint = str(tmp_path / "checkpoint")
        metrics = {"finetuned_nll": 0.5}

    def fake_run_finbert_training(**kwargs):
        calls.update(kwargs)
        return Result()

    monkeypatch.setattr(launch, "run_finbert_training", fake_run_finbert_training)
    result = CliRunner().invoke(
        app,
        [
            "train-finbert",
            "--train-jsonl", str(train_path),
            "--validation-jsonl", str(validation_path),
            "--root", str(tmp_path),
            "--epochs", "2",
            "--learning-rate", "0.00001",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert calls["train_examples"] == train_rows
    assert calls["validation_examples"] == validation_rows
    assert calls["epochs"] == 2
    assert calls["learning_rate"] == 1e-5
    assert "finbert-test-run" in result.stdout
