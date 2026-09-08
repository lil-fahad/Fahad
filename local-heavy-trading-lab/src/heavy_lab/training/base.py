from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from heavy_lab.runs import RunRegistry


@dataclass(frozen=True)
class TrainPlan:
    run_id: str
    model_kind: str
    config: dict[str, Any]
    checkpoint_dir: Path


@dataclass(frozen=True)
class TrainResult:
    run_id: str
    status: str
    checkpoint: str | None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryProbeResult:
    safe_full_finetune: bool
    max_micro_batch_size: int
    reason: str
    peak_memory_gb: float | None = None


class TrainerAdapter(ABC):
    kind: str

    def __init__(self, registry: RunRegistry) -> None:
        self.registry = registry

    @abstractmethod
    def prepare(self, run, dataset, hardware) -> TrainPlan:
        raise NotImplementedError

    @abstractmethod
    def train(self, plan: TrainPlan) -> TrainResult:
        raise NotImplementedError

    @abstractmethod
    def resume(self, run_id: str) -> TrainResult:
        raise NotImplementedError
