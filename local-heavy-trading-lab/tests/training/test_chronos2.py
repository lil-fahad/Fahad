from pathlib import Path


def test_chronos2_uses_native_fit_with_separate_validation(tmp_path: Path):
    from heavy_lab.runs import RunRegistry
    from heavy_lab.training.base import MemoryProbeResult
    from heavy_lab.training.chronos2 import Chronos2Trainer

    fit_calls: list[dict] = []

    class FineTunedChronos:
        def save_pretrained(self, path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "chronos.txt").write_text("trained", encoding="utf-8")

    class TinyChronosPipeline:
        def fit(self, inputs, **kwargs):
            fit_calls.append({"inputs": inputs, **kwargs})
            return FineTunedChronos()

    registry = RunRegistry(tmp_path / "runs")
    run = registry.start("chronos2", {"profile": "smoke"})
    train_windows = [
        {"context": [100.0, 101.0, 102.0], "target": [103.0, 104.0]},
        {"context": [200.0, 199.0, 201.0], "target": [202.0, 203.0]},
    ]
    validation_windows = [
        {"context": [300.0, 301.0, 302.0], "target": [303.0, 304.0]},
    ]
    trainer = Chronos2Trainer(registry, model=TinyChronosPipeline())
    plan = trainer.prepare(
        run,
        dataset={"train": train_windows, "validation": validation_windows},
        hardware={"device": "cuda", "vram_gb": 8.0},
        preflight=MemoryProbeResult(False, 1, "below 24 GB", 5.0),
        output_root=tmp_path / "trained",
        epochs=2,
        learning_rate=1e-4,
    )
    result = trainer.train(plan)

    assert len(fit_calls) == 1
    call = fit_calls[0]
    assert call["inputs"] == [[100.0, 101.0, 102.0, 103.0, 104.0], [200.0, 199.0, 201.0, 202.0, 203.0]]
    assert call["validation_inputs"] == [[300.0, 301.0, 302.0, 303.0, 304.0]]
    assert call["prediction_length"] == 2
    assert call["num_steps"] == 2
    assert call["batch_size"] == 1
    assert call["learning_rate"] == 1e-4
    assert result.status == "COMPLETED"
    assert Path(result.checkpoint, "model", "chronos.txt").is_file()
    assert Path(result.checkpoint, "checkpoint.ok").is_file()
