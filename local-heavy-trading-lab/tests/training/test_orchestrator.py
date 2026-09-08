from dataclasses import dataclass


def test_train_all_is_sequential_failure_isolated_and_resumable():
    from heavy_lab.training.orchestrator import HeavyTrainingOrchestrator, TrainingJob

    events: list[str] = []

    @dataclass
    class Result:
        run_id: str
        status: str

    class FakeTrainer:
        def __init__(self, name: str, fail: bool = False):
            self.name = name
            self.fail = fail

        def train(self, plan):
            events.append(f"train:{self.name}")
            if self.fail:
                raise RuntimeError(f"{self.name} failed")
            return Result(plan.run_id, "COMPLETED")

        def resume(self, run_id):
            events.append(f"resume:{self.name}:{run_id}")
            return Result(run_id, "COMPLETED")

    @dataclass
    class Plan:
        run_id: str

    jobs = [
        TrainingJob("ttm", FakeTrainer("ttm"), Plan("run-ttm")),
        TrainingJob("timesfm25", FakeTrainer("timesfm25", fail=True), Plan("run-timesfm")),
        TrainingJob("chronos2", FakeTrainer("chronos2"), Plan("run-chronos")),
    ]
    orchestrator = HeavyTrainingOrchestrator()
    summary = orchestrator.run(jobs)

    assert events[:3] == ["train:ttm", "train:timesfm25", "train:chronos2"]
    assert summary["ttm"].status == "COMPLETED"
    assert summary["timesfm25"].status == "FAILED"
    assert "timesfm25 failed" in summary["timesfm25"].error
    assert summary["chronos2"].status == "COMPLETED"

    resumed = orchestrator.resume(jobs[0])
    assert resumed.status == "COMPLETED"
    assert events[-1] == "resume:ttm:run-ttm"
