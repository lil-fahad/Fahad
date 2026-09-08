from pathlib import Path

import pytest


def test_kronos_enforces_pair_revision_and_ohlcv_order(tmp_path: Path):
    from heavy_lab.runs import RunRegistry
    from heavy_lab.training.base import MemoryProbeResult
    from heavy_lab.training.kronos import KronosTrainer

    calls: list[dict] = []

    class TinyKronos:
        def evaluate(self, windows):
            return 1.0 if not calls else 0.3

        def train_step(self, *, ohlcv, learning_rate, gradient_checkpointing):
            calls.append({"ohlcv": ohlcv, "gradient_checkpointing": gradient_checkpointing})
            return 0.4

        def save_pretrained(self, path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "kronos.txt").write_text("trained", encoding="utf-8")

    paired_model = {"name": "kronos", "requested_revision": "release-1", "resolved_revision": "model-sha"}
    paired_tokenizer = {
        "name": "kronos_tokenizer",
        "requested_revision": "release-1",
        "resolved_revision": "tokenizer-sha",
    }
    bad_tokenizer = dict(paired_tokenizer, requested_revision="release-2")

    registry = RunRegistry(tmp_path / "runs")
    run = registry.start("kronos", {"profile": "smoke"})
    trainer = KronosTrainer(registry, model=TinyKronos())
    rows = [
        {"ohlcv": [[100.0, 102.0, 99.0, 101.0, 1000.0], [101.0, 103.0, 100.0, 102.0, 1200.0]]},
        {"ohlcv": [[200.0, 201.0, 198.0, 199.0, 900.0], [199.0, 202.0, 198.0, 201.0, 1100.0]]},
    ]

    with pytest.raises(ValueError, match="revision"):
        trainer.prepare(
            run,
            dataset={"train": rows, "validation": rows[:1]},
            hardware={"device": "cuda", "vram_gb": 8.0},
            preflight=MemoryProbeResult(False, 1, "below 24 GB", 5.0),
            model_manifest=paired_model,
            tokenizer_manifest=bad_tokenizer,
            output_root=tmp_path / "bad",
        )

    plan = trainer.prepare(
        run,
        dataset={"train": rows, "validation": rows[:1]},
        hardware={"device": "cuda", "vram_gb": 8.0},
        preflight=MemoryProbeResult(False, 1, "below 24 GB", 5.0),
        model_manifest=paired_model,
        tokenizer_manifest=paired_tokenizer,
        output_root=tmp_path / "trained",
        epochs=1,
        learning_rate=1e-4,
    )
    result = trainer.train(plan)

    assert calls[0]["ohlcv"] == rows[0]["ohlcv"]
    assert calls[0]["gradient_checkpointing"] is True
    assert result.metrics["finetuned_loss"] < result.metrics["baseline_loss"]
    assert Path(result.checkpoint, "model", "kronos.txt").is_file()
    assert Path(result.checkpoint, "checkpoint.ok").is_file()
