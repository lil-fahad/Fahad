from pathlib import Path


def test_shared_trainer_records_run_and_latest_checkpoint(tmp_path: Path):
    from heavy_lab.training.base import TrainPlan, TrainResult, TrainerAdapter
    from heavy_lab.runs import RunRegistry

    class FakeTrainer(TrainerAdapter):
        kind = "fake"

        def prepare(self, run, dataset, hardware):
            return TrainPlan(run_id=run.run_id, model_kind=self.kind, config={"dataset": dataset}, checkpoint_dir=tmp_path / "ckpt")

        def train(self, plan):
            plan.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (plan.checkpoint_dir / "checkpoint.ok").write_text("ok", encoding="utf-8")
            self.registry.mark_checkpoint(plan.run_id, plan.checkpoint_dir)
            self.registry.set_status(plan.run_id, "COMPLETED")
            return TrainResult(run_id=plan.run_id, status="COMPLETED", checkpoint=str(plan.checkpoint_dir), metrics={"loss": 0.1})

        def resume(self, run_id):
            run = self.registry.resume(run_id)
            return TrainResult(run_id=run.run_id, status=run.status, checkpoint=run.latest_checkpoint, metrics={})

    registry = RunRegistry(tmp_path / "runs")
    run = registry.start("fake", {"seed": 7})
    trainer = FakeTrainer(registry)
    result = trainer.train(trainer.prepare(run, "dataset-v1", {"device": "cpu"}))

    resumed = registry.resume(run.run_id)
    assert result.status == "COMPLETED"
    assert resumed.config["seed"] == 7
    assert resumed.latest_checkpoint == str((tmp_path / "ckpt").resolve())
