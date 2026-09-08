from __future__ import annotations

from pathlib import Path

from heavy_lab.hardware import detect_hardware, select_training_profile


LOW_DISK_WARNING_GB = 100.0


def doctor_report(storage_root: Path | str) -> dict:
    profile = detect_hardware(storage_root)
    warnings: list[dict[str, str | float]] = []
    if profile.disk_free_gb < LOW_DISK_WARNING_GB:
        warnings.append(
            {
                "code": "LOW_DISK_FOR_FULL_RUN",
                "message": "Full local research run should have at least 100 GB free disk.",
                "disk_free_gb": profile.disk_free_gb,
            }
        )
    if profile.device == "cpu":
        warnings.append(
            {
                "code": "CPU_ONLY_HEAVY_LIMIT",
                "message": "Heavy model snapshots can be installed, but full heavy training will be gated by preflight.",
            }
        )
    return {
        "hardware": profile.as_dict(),
        "training_profile": select_training_profile(profile),
        "warnings": warnings,
    }
