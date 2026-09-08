from __future__ import annotations

from collections.abc import Callable, Iterable

from heavy_lab.hardware import HardwareProfile
from heavy_lab.training.base import MemoryProbeResult


_FULL_FINETUNE_MIN_VRAM_GB = {
    "timesfm25": 24.0,
    "chronos2": 24.0,
    "kronos": 24.0,
}


def probe_training_memory(
    hardware: HardwareProfile,
    *,
    model_name: str,
    candidate_batch_sizes: Iterable[int],
    trial: Callable[[int], float],
) -> MemoryProbeResult:
    candidates = tuple(int(size) for size in candidate_batch_sizes)
    if not candidates or any(size <= 0 for size in candidates):
        raise ValueError("candidate_batch_sizes must contain positive integers")

    max_safe = 0
    peak_memory_gb: float | None = None
    failure_reason: str | None = None

    for batch_size in candidates:
        try:
            observed_peak = float(trial(batch_size))
        except (MemoryError, RuntimeError) as exc:
            failure_reason = str(exc) or exc.__class__.__name__
            break
        max_safe = batch_size
        peak_memory_gb = observed_peak

    if max_safe == 0:
        return MemoryProbeResult(
            safe_full_finetune=False,
            max_micro_batch_size=0,
            reason=failure_reason or "No candidate micro-batch completed the memory probe",
            peak_memory_gb=peak_memory_gb,
        )

    model = model_name.strip().lower()
    required_vram = _FULL_FINETUNE_MIN_VRAM_GB.get(model)
    has_full_memory = required_vram is None or float(hardware.vram_gb or 0.0) >= required_vram
    safe_full = bool(hardware.cuda and has_full_memory and failure_reason is None)

    reasons: list[str] = []
    if required_vram is not None and not has_full_memory:
        reasons.append(f"Full fine-tuning for {model} requires at least {required_vram:g} GB VRAM")
    if failure_reason:
        reasons.append(f"Memory probe stopped after batch {max_safe}: {failure_reason}")
    if not reasons:
        reasons.append("Memory probe completed all requested batch sizes")

    return MemoryProbeResult(
        safe_full_finetune=safe_full,
        max_micro_batch_size=max_safe,
        reason="; ".join(reasons),
        peak_memory_gb=peak_memory_gb,
    )
