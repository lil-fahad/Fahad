from pathlib import Path


def test_chronos2_low_vram_path_updates_and_checkpoints(tmp_path: Path):
    from heavy_lab.runs import RunRegistry
    from heavy_lab.training.base import MemoryProbeResult
    from heavy_lab.training.chronos2 import Chronos2Trainer

    updates: list[dict] = []

    class TinyChronos:
        def evaluate(self, windows):
            return 1.0 if not updates else 0.4

        def train_step(self, *, context, target, learning_rate, gradient_checkpointing):
            updates.append(
                {
                    "context": list(context),
                    "target": list(target),
                    "gradient_checkpointing": gradient_checkpointing,
                }
            )
            return 0.5

        def save_pretrained(self, path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "chronos.txt").write_text("trained", encoding="utf-8")

    registry = RunRegistry(tmp_path / "runs")
    run = registry.start("chronos2", {"profile": "smoke"})
    windows = [
        {"context": [100.0, 101.0, 102.0], "target": [103.0, 104.0]},
        {"context": [200.0, 199.0, 201.0], "target": [202.0, 203.0]},
    ]
    trainer = Chronos2Trainer(registry, model=TinyChronos())
    plan = trainer.prepare(
        run,
        dataset={"train": windows, "validation": windows[:1]},
        hardware={"device": "cuda", "vram_gb": 8.0},
        preflight=MemoryProbeResult(False, 1, "below 24 GB", 5.0),
        output_root=tmp_path / "trained",
        epochs=1,
        learning_rate=1e-4,
    )
    result = trainer.train(plan)

    assert updates
    assert updates[0]["context"] == [100.0, 101.0, 102.0]
    assert updates[0]["target"] == [103.0, 104.0]
    assert updates[0]["gradient_checkpointing"] is True
    assert result.metrics["finetuned_loss"] < result.metrics["baseline_loss"]
    assert Path(result.checkpoint, "model", "chronos.txt").is_file()
    assert Path(result.checkpoint, "checkpoint.ok").is_file()
