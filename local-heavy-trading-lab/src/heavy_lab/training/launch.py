from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from heavy_lab.hardware import HardwareProfile, detect_hardware
from heavy_lab.models.catalog import MODEL_CATALOG
from heavy_lab.paths import LabPaths
from heavy_lab.runs import RunRegistry
from heavy_lab.training.base import MemoryProbeResult, TrainResult
from heavy_lab.training.chronos2 import Chronos2Trainer
from heavy_lab.training.datasets import build_timesfm_examples, build_ttm_examples
from heavy_lab.training.finbert import FinBERTTrainer
from heavy_lab.training.integrations.chronos2_native import load_chronos2_pipeline
from heavy_lab.training.integrations.finbert_hf import FinBERTHFModelAdapter
from heavy_lab.training.integrations.kronos_official import (
    ensure_kronos_source,
    prepare_kronos_official_files,
    run_kronos_sequential,
)
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


def run_kronos_training(
    *,
    root: Path,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    lookback_window: int,
    predict_window: int,
    tokenizer_epochs: int = 1,
    predictor_epochs: int = 1,
    hardware: HardwareProfile | None = None,
    ensure_source: Callable[..., Path] = ensure_kronos_source,
    prepare_files: Callable[..., Any] = prepare_kronos_official_files,
    run_sequential: Callable[..., Any] = run_kronos_sequential,
) -> TrainResult:
    if tokenizer_epochs <= 0 or predictor_epochs <= 0:
        raise ValueError("Kronos epochs must be positive")
    if lookback_window < 2 or predict_window < 1:
        raise ValueError("Kronos lookback and prediction windows must be positive")

    paths = LabPaths.from_root(Path(root)); paths.ensure_runtime_dirs()
    hardware = hardware or detect_hardware(paths.root)
    policy = plan_training("kronos", hardware)
    model_path = paths.models_base / MODEL_CATALOG["kronos"].local_name
    tokenizer_path = paths.models_base / MODEL_CATALOG["kronos_tokenizer"].local_name

    for path, label in ((model_path, "Kronos model"), (tokenizer_path, "Kronos tokenizer")):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} snapshot not found: {path}")
        if not (path / "manifest.json").is_file():
            raise FileNotFoundError(f"{label} manifest not found: {path / 'manifest.json'}")

    registry = RunRegistry(paths.runs)
    run = registry.start(
        "kronos",
        {
            "lookback_window": int(lookback_window),
            "predict_window": int(predict_window),
            "tokenizer_epochs": int(tokenizer_epochs),
            "predictor_epochs": int(predictor_epochs),
            "model_path": str(model_path),
            "tokenizer_path": str(tokenizer_path),
            "mode": policy.mode,
            "precision": policy.precision,
            "micro_batch_size": policy.micro_batch_size,
            "gradient_accumulation_steps": policy.gradient_accumulation_steps,
        },
    )
    work_dir = paths.artifacts / "kronos-work" / run.run_id
    checkpoint_dir = paths.checkpoints / run.run_id / "checkpoint"

    try:
        source_root = ensure_source(vendor_root=paths.root / "vendor")
        prepared = prepare_files(
            train_frame=train_frame,
            validation_frame=validation_frame,
            work_dir=work_dir,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            output_root=checkpoint_dir,
            lookback_window=int(lookback_window),
            predict_window=int(predict_window),
            batch_size=int(policy.micro_batch_size),
            accumulation_steps=int(policy.gradient_accumulation_steps),
            tokenizer_epochs=int(tokenizer_epochs),
            predictor_epochs=int(predictor_epochs),
            use_cuda=bool(hardware.cuda),
        )
        completed = run_sequential(source_root=source_root, config_path=prepared.config_path)
        if int(getattr(completed, "returncode", 0)) != 0:
            raise RuntimeError("Pinned Kronos sequential trainer returned a non-zero exit code")

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "checkpoint.ok").write_text("ok", encoding="utf-8")
        registry.mark_checkpoint(run.run_id, checkpoint_dir)
        registry.append_event(
            run.run_id,
            {
                "event": "kronos_official_completed",
                "source_root": str(Path(source_root).resolve()),
                "config_path": str(Path(prepared.config_path).resolve()),
            },
        )
        registry.set_status(run.run_id, "COMPLETED")
        return TrainResult(
            run_id=run.run_id,
            status="COMPLETED",
            checkpoint=str(checkpoint_dir.resolve()),
            metrics={},
        )
    except Exception as exc:
        registry.set_status(run.run_id, "FAILED")
        registry.append_event(run.run_id, {"event": "training_failed", "error": str(exc)})
        raise


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
