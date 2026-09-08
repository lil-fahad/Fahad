from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from heavy_lab.hardware import HardwareProfile, detect_hardware
from heavy_lab.models.catalog import MODEL_CATALOG
from heavy_lab.paths import LabPaths
from heavy_lab.runs import RunRegistry
from heavy_lab.training.datasets import build_ttm_examples
from heavy_lab.training.integrations.ttm_hf import TTMHFModelAdapter
from heavy_lab.training.policy import plan_training
from heavy_lab.training.ttm import TTMTrainer


def run_ttm_training(
    *,
    root: Path,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    context_length: int,
    prediction_length: int,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    hardware: HardwareProfile | None = None,
    adapter_factory: Callable[..., Any] = TTMHFModelAdapter,
):
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    paths = LabPaths.from_root(Path(root))
    paths.ensure_runtime_dirs()
    hardware = hardware or detect_hardware(paths.root)
    policy = plan_training("ttm", hardware)

    train_examples = build_ttm_examples(
        train_frame,
        context_length=context_length,
        prediction_length=prediction_length,
    )
    validation_examples = build_ttm_examples(
        validation_frame,
        context_length=context_length,
        prediction_length=prediction_length,
    )
    if not train_examples:
        raise ValueError("TTM training split does not contain enough chronological rows")
    if not validation_examples:
        raise ValueError("TTM validation split does not contain enough chronological rows")

    spec = MODEL_CATALOG["ttm"]
    model_path = paths.models_base / spec.local_name
    adapter = adapter_factory(
        model_path=model_path,
        output_root=paths.artifacts / "ttm-trainer",
        batch_size=policy.micro_batch_size,
        precision=policy.precision,
    )

    registry = RunRegistry(paths.runs)
    run = registry.start(
        "ttm",
        {
            "context_length": int(context_length),
            "prediction_length": int(prediction_length),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "model_path": str(model_path),
            "policy": {
                "mode": policy.mode,
                "precision": policy.precision,
                "micro_batch_size": policy.micro_batch_size,
                "gradient_accumulation_steps": policy.gradient_accumulation_steps,
                "gradient_checkpointing": policy.gradient_checkpointing,
            },
        },
    )

    trainer = TTMTrainer(registry, model=adapter)
    plan = trainer.prepare(
        run,
        {"train": train_examples, "validation": validation_examples},
        hardware.as_dict(),
        output_root=paths.checkpoints,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    return trainer.train(plan)
