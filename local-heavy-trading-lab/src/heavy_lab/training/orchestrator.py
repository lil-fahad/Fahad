from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingJob:
    name: str
    trainer: Any
    plan: Any


@dataclass(frozen=True)
class TrainingOutcome:
    name: str
    run_id: str
    status: str
    result: Any | None = None
    error: str | None = None


class HeavyTrainingOrchestrator:
    """Run heavy model jobs sequentially so a single GPU is never contended."""

    def run(self, jobs: list[TrainingJob]) -> dict[str, TrainingOutcome]:
        outcomes: dict[str, TrainingOutcome] = {}
        for job in jobs:
            run_id = str(job.plan.run_id)
            try:
                result = job.trainer.train(job.plan)
            except Exception as exc:
                registry = getattr(job.trainer, "registry", None)
                if registry is not None:
                    try:
                        registry.set_status(run_id, "FAILED")
                        registry.append_event(run_id, {"event": "training_failed", "error": str(exc)})
                    except Exception:
                        pass
                outcomes[job.name] = TrainingOutcome(
                    name=job.name,
                    run_id=run_id,
                    status="FAILED",
                    error=str(exc),
                )
                continue

            outcomes[job.name] = TrainingOutcome(
                name=job.name,
                run_id=run_id,
                status=str(getattr(result, "status", "COMPLETED")),
                result=result,
            )
        return outcomes

    def resume(self, job: TrainingJob):
        return job.trainer.resume(str(job.plan.run_id))
