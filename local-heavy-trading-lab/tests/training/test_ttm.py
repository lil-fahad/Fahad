from pathlib import Path


def test_ttm_runs_baseline_then_real_update_and_saves_checkpoint(tmp_path: Path):
    from heavy_lab.runs import RunRegistry
    from heavy_lab.training.ttm import TTMTrainer

    events: list[str] = []

    class TinyTTM:
        def __init__(self):
            self.weight = 1.0

        def evaluate(self, rows):
            events.append("baseline" if self.weight == 1.0 else "finetuned_eval")
            return abs(self.weight - 2.0)

        def train_epoch(self, rows, *, learning_rate: float):
            events.append("update")
            self.weight += learning_rate
            return abs(self.weight - 2.0)

        def save_pretrained(self, path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "model.txt").write_text(str(self.weight), encoding="utf-8")

    registry = RunRegistry(tmp_path / "runs")
    run = registry.start("ttm", {"profile": "smoke"})
    model = TinyTTM()
    trainer = TTMTrainer(registry, model=model)
    plan = trainer.prepare(
        run,
        dataset={"train": [1, 2, 3], "validation": [4, 5]},
        hardware={"device": "cpu"},
        output_root=tmp_path / "trained",
        epochs=2,
        learning_rate=0.25,
    )
    result = trainer.train(plan)

    assert events[0] == "baseline"
    assert "update" in events
    assert model.weight > 1.0
    assert result.metrics["baseline_loss"] == 1.0
    assert result.metrics["finetuned_loss"] < result.metrics["baseline_loss"]
    assert result.checkpoint is not None
    assert Path(result.checkpoint, "checkpoint.ok").is_file()
    assert registry.resume(run.run_id).status == "COMPLETED"
