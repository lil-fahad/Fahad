from pathlib import Path

import numpy as np
import pytest


def test_finbert_trains_and_calibrates_on_validation_only(tmp_path: Path):
    from heavy_lab.runs import RunRegistry
    from heavy_lab.training.finbert import FinBERTTrainer

    updates: list[tuple[str, int]] = []

    class TinyFinBERT:
        def train_step(self, *, text, label, learning_rate):
            updates.append((text, label))
            return 0.4

        def predict_logits(self, examples):
            rows = []
            for item in examples:
                label = int(item["label"])
                logits = [-1.0, -1.0, -1.0]
                logits[label] = 2.0 if updates else 1.0
                rows.append(logits)
            return np.asarray(rows, dtype=float)

        def save_pretrained(self, path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "model.txt").write_text("trained", encoding="utf-8")

        def save_tokenizer(self, path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "tokenizer.txt").write_text("saved", encoding="utf-8")

    train = [
        {"document_id": "a", "timestamp_utc": "2026-01-01T10:00:00Z", "text": "profits rose", "label": 2},
        {"document_id": "b", "timestamp_utc": "2026-01-01T11:00:00Z", "text": "loss widened", "label": 0},
    ]
    validation = [
        {"document_id": "c", "timestamp_utc": "2026-01-02T10:00:00Z", "text": "outlook stable", "label": 1},
        {"document_id": "d", "timestamp_utc": "2026-01-02T11:00:00Z", "text": "guidance raised", "label": 2},
    ]

    registry = RunRegistry(tmp_path / "runs")
    run = registry.start("finbert", {"profile": "smoke"})
    trainer = FinBERTTrainer(registry, model=TinyFinBERT())

    with pytest.raises(ValueError, match="overlap"):
        trainer.prepare(
            run,
            dataset={"train": train, "validation": [dict(validation[0], document_id="a")]},
            hardware={"device": "cpu"},
            output_root=tmp_path / "bad",
        )

    plan = trainer.prepare(
        run,
        dataset={"train": train, "validation": validation},
        hardware={"device": "cpu"},
        output_root=tmp_path / "trained",
        epochs=1,
        learning_rate=1e-4,
    )
    result = trainer.train(plan)

    assert updates == [("profits rose", 2), ("loss widened", 0)]
    assert result.metrics["finetuned_nll"] <= result.metrics["baseline_nll"]
    assert result.metrics["calibration_temperature"] > 0
    assert Path(result.checkpoint, "model", "model.txt").is_file()
    assert Path(result.checkpoint, "tokenizer", "tokenizer.txt").is_file()
    assert Path(result.checkpoint, "calibration.json").is_file()

    probs = trainer.calibrated_probabilities(validation)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(validation)), atol=1e-7)
