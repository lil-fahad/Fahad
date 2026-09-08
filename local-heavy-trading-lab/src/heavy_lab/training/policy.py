from __future__ import annotations

from dataclasses import dataclass

from heavy_lab.hardware import HardwareProfile


@dataclass(frozen=True)
class TrainingPlan:
    model: str
    mode: str
    precision: str
    micro_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool


def _precision(hw: HardwareProfile) -> str:
    if hw.cuda or hw.mps:
        if hw.bf16:
            return "bf16"
        if hw.fp16:
            return "fp16"
    return "fp32"


def plan_training(model_name: str, hw: HardwareProfile) -> TrainingPlan:
    model = model_name.strip().lower()
    precision = _precision(hw)
    vram = float(hw.vram_gb or 0.0)

    if model == "ttm":
        return TrainingPlan(model, "full", precision, 8 if hw.cuda else 2, 1, False)

    if model == "finbert":
        return TrainingPlan(model, "full", precision, 4 if hw.cuda else 1, 4 if hw.cuda else 8, True)

    if model == "timesfm25":
        if hw.cuda and vram >= 24.0:
            return TrainingPlan(model, "full", precision, 1, 8, True)
        if hw.cuda:
            return TrainingPlan(model, "lora", precision, 1, 16 if vram <= 10.0 else 8, True)
        return TrainingPlan(model, "lora", precision, 1, 16, True)

    if model in {"chronos2", "kronos"}:
        if hw.cuda and vram >= 24.0:
            return TrainingPlan(model, "native-full", precision, 2, 4, True)
        return TrainingPlan(model, "native-low-vram", precision, 1, 8, True)

    raise ValueError(f"unknown trainable model: {model_name}")
