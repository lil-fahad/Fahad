from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from heavy_lab.hardware import HardwareProfile, detect_hardware
from heavy_lab.models.catalog import MODEL_CATALOG
from heavy_lab.paths import LabPaths
from heavy_lab.runs import RunRegistry
from heavy_lab.training.base import MemoryProbeResult
from heavy_lab.training.chronos2 import Chronos2Trainer
from heavy_lab.training.datasets import build_timesfm_examples, build_ttm_examples
from heavy_lab.training.finbert import FinBERTTrainer
from heavy_lab.training.integrations.chronos2_native import load_chronos2_pipeline
from heavy_lab.training.integrations.finbert_hf import FinBERTHFModelAdapter
from heavy_lab.training.integrations.timesfm_hf import TimesFMHFModelAdapter
from heavy_lab.training.integrations.ttm_hf import TTMHFModelAdapter
from heavy_lab.training.policy import plan_training
from heavy_lab.training.timesfm25 import TimesFM25Trainer
from heavy_lab.training.ttm import TTMTrainer


def run_ttm_training(*, root: Path, train_frame: pd.DataFrame, validation_frame: pd.DataFrame, context_length: int, prediction_length: int, epochs: int = 1, learning_rate: float = 1e-4, hardware: HardwareProfile | None = None, adapter_factory: Callable[..., Any] = TTMHFModelAdapter):
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    paths = LabPaths.from_root(Path(root)); paths.ensure_runtime_dirs()
    hardware = hardware or detect_hardware(paths.root); policy = plan_training("ttm", hardware)
    train_examples = build_ttm_examples(train_frame, context_length=context_length, prediction_length=prediction_length)
    validation_examples = build_ttm_examples(validation_frame, context_length=context_length, prediction_length=prediction_length)
    if not train_examples or not validation_examples: raise ValueError("TTM splits need more chronological rows")
    model_path = paths.models_base / MODEL_CATALOG["ttm"].local_name
    adapter = adapter_factory(model_path=model_path, output_root=paths.artifacts / "ttm-trainer", batch_size=policy.micro_batch_size, precision=policy.precision)
    registry = RunRegistry(paths.runs); run = registry.start("ttm", {"context_length": context_length, "prediction_length": prediction_length, "epochs": epochs, "learning_rate": learning_rate, "model_path": str(model_path)})
    trainer = TTMTrainer(registry, model=adapter)
    return trainer.train(trainer.prepare(run, {"train": train_examples, "validation": validation_examples}, hardware.as_dict(), output_root=paths.checkpoints, epochs=epochs, learning_rate=learning_rate))


def run_timesfm_training(*, root: Path, train_frame: pd.DataFrame, validation_frame: pd.DataFrame, context_length: int, prediction_length: int, epochs: int = 1, learning_rate: float = 1e-4, hardware: HardwareProfile | None = None, preflight: MemoryProbeResult, adapter_factory: Callable[..., Any] = TimesFMHFModelAdapter):
    if epochs <= 0 or learning_rate <= 0: raise ValueError("epochs and learning_rate must be positive")
    paths = LabPaths.from_root(Path(root)); paths.ensure_runtime_dirs()
    hardware = hardware or detect_hardware(paths.root); policy = plan_training("timesfm25", hardware)
    train_examples = build_timesfm_examples(train_frame, context_length=context_length, prediction_length=prediction_length)
    validation_examples = build_timesfm_examples(validation_frame, context_length=context_length, prediction_length=prediction_length)
    if not train_examples or not validation_examples: raise ValueError("TimesFM splits need more chronological rows")
    model_path = paths.models_base / MODEL_CATALOG["timesfm25"].local_name
    adapter = adapter_factory(model_path=model_path, device=hardware.device, precision=policy.precision)
    registry = RunRegistry(paths.runs); run = registry.start("timesfm25", {"context_length": context_length, "prediction_length": prediction_length, "epochs": epochs, "learning_rate": learning_rate, "model_path": str(model_path), "finetune_mode": "full" if preflight.safe_full_finetune else "lora", "preflight_reason": preflight.reason})
    trainer = TimesFM25Trainer(registry, model=adapter)
    return trainer.train(trainer.prepare(run, {"train": train_examples, "validation": validation_examples}, hardware.as_dict(), preflight=preflight, output_root=paths.checkpoints, epochs=epochs, learning_rate=learning_rate))


def run_chronos2_training(*, root: Path, train_frame: pd.DataFrame, validation_frame: pd.DataFrame, context_length: int, prediction_length: int, steps: int = 1000, learning_rate: float = 1e-5, hardware: HardwareProfile | None = None, preflight: MemoryProbeResult, pipeline_loader: Callable[..., Any] = load_chronos2_pipeline):
    if steps <= 0 or learning_rate <= 0: raise ValueError("steps and learning_rate must be positive")
    paths = LabPaths.from_root(Path(root)); paths.ensure_runtime_dirs()
    hardware = hardware or detect_hardware(paths.root); policy = plan_training("chronos2", hardware)
    def windows(frame):
        return [{"context": e["past_values"], "target": e["future_values"]} for e in build_timesfm_examples(frame, context_length=context_length, prediction_length=prediction_length)]
    train_windows, validation_windows = windows(train_frame), windows(validation_frame)
    if not train_windows or not validation_windows: raise ValueError("Chronos-2 splits need more chronological rows")
    model_path = paths.models_base / MODEL_CATALOG["chronos2"].local_name
    pipeline = pipeline_loader(model_path=model_path, device=hardware.device, precision=policy.precision)
    registry = RunRegistry(paths.runs); run = registry.start("chronos2", {"context_length": context_length, "prediction_length": prediction_length, "steps": steps, "learning_rate": learning_rate, "model_path": str(model_path), "finetune_mode": "full" if preflight.safe_full_finetune else "lora", "preflight_reason": preflight.reason})
    trainer = Chronos2Trainer(registry, model=pipeline)
    return trainer.train(trainer.prepare(run, {"train": train_windows, "validation": validation_windows}, hardware.as_dict(), preflight=preflight, output_root=paths.checkpoints, epochs=steps, learning_rate=learning_rate))


def run_finbert_training(*, root: Path, train_examples, validation_examples, epochs: int = 1, learning_rate: float = 1e-5, hardware: HardwareProfile | None = None, adapter_factory: Callable[..., Any] = FinBERTHFModelAdapter):
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    paths = LabPaths.from_root(Path(root)); paths.ensure_runtime_dirs()
    hardware = hardware or detect_hardware(paths.root)
    model_path = paths.models_base / MODEL_CATALOG["finbert"].local_name
    adapter = adapter_factory(model_path=model_path, device=hardware.device)
    registry = RunRegistry(paths.runs)
    run = registry.start("finbert", {"epochs": epochs, "learning_rate": learning_rate, "model_path": str(model_path)})
    trainer = FinBERTTrainer(registry, model=adapter)
    plan = trainer.prepare(
        run,
        {"train": list(train_examples), "validation": list(validation_examples)},
        hardware.as_dict(),
        output_root=paths.checkpoints,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    return trainer.train(plan)
