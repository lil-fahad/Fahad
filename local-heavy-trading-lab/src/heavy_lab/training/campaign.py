from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd

from heavy_lab.hardware import detect_hardware
from heavy_lab.training.base import MemoryProbeResult
from heavy_lab.training.launch import (
    run_chronos2_training,
    run_finbert_training,
    run_kronos_training,
    run_timesfm_training,
    run_ttm_training,
)
from heavy_lab.training.readiness import training_readiness


_REQUIRED_PATHS = (
    "train_parquet",
    "validation_parquet",
    "finbert_train_jsonl",
    "finbert_validation_jsonl",
)


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid campaign manifest JSON: {exc.msg}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Campaign manifest must be a JSON object")

    missing = [key for key in _REQUIRED_PATHS if not str(raw.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Campaign manifest is missing required fields: {', '.join(missing)}")

    base = path.parent
    resolved = dict(raw)
    for key in _REQUIRED_PATHS:
        candidate = Path(str(raw[key])).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        resolved[key] = candidate.resolve()

    manifest_profile = str(raw.get("profile", "smoke")).strip().lower()
    if manifest_profile not in {"smoke", "max"}:
        raise ValueError("Campaign manifest profile must be smoke or max")
    resolved["profile"] = manifest_profile
    resolved["manifest_path"] = path
    return resolved


def _read_market_splits(train_path: Path, validation_path: Path):
    return pd.read_parquet(train_path), pd.read_parquet(validation_path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"JSONL row {line_number} in {path} must be an object")
        rows.append(item)
    if not rows:
        raise ValueError(f"JSONL file is empty: {path}")
    return rows


def _guarded_preflight(hardware, model_name: str) -> MemoryProbeResult:
    vram_gb = float(getattr(hardware, "vram_gb", 0.0) or 0.0)
    if vram_gb < 24.0:
        reason = (
            f"{model_name} full fine-tuning requires at least 24 GB VRAM; "
            f"detected {vram_gb:g} GB, using the low-VRAM/adaptation path"
        )
    else:
        reason = (
            f"{model_name} campaign execution remains on the guarded adaptation path "
            "until a dedicated forward/backward memory probe explicitly approves full fine-tuning"
        )
    return MemoryProbeResult(
        safe_full_finetune=False,
        max_micro_batch_size=1,
        reason=reason,
        peak_memory_gb=None,
    )


def _cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _profile_settings(profile: str) -> dict[str, dict[str, Any]]:
    if profile == "smoke":
        return {
            "ttm": {"context_length": 64, "prediction_length": 12, "epochs": 1, "learning_rate": 1e-4},
            "timesfm25": {"context_length": 128, "prediction_length": 24, "epochs": 1, "learning_rate": 1e-4},
            "chronos2": {"context_length": 128, "prediction_length": 24, "steps": 50, "learning_rate": 1e-5},
            "kronos": {"lookback_window": 64, "predict_window": 16, "tokenizer_epochs": 1, "predictor_epochs": 1},
            "finbert": {"epochs": 1, "learning_rate": 1e-5},
        }
    if profile == "max":
        return {
            "ttm": {"context_length": 512, "prediction_length": 96, "epochs": 10, "learning_rate": 1e-4},
            "timesfm25": {"context_length": 512, "prediction_length": 96, "epochs": 5, "learning_rate": 1e-4},
            "chronos2": {"context_length": 512, "prediction_length": 96, "steps": 3000, "learning_rate": 1e-5},
            "kronos": {"lookback_window": 64, "predict_window": 16, "tokenizer_epochs": 3, "predictor_epochs": 5},
            "finbert": {"epochs": 5, "learning_rate": 1e-5},
        }
    raise ValueError("profile must be smoke or max")


def _result_payload(result: Any) -> dict[str, Any]:
    return {
        "status": str(getattr(result, "status", "COMPLETED")),
        "run_id": str(getattr(result, "run_id", "")),
        "checkpoint": getattr(result, "checkpoint", None),
        "metrics": getattr(result, "metrics", {}) or {},
    }


def run_training_campaign(
    *,
    root: Path,
    manifest_path: Path,
    profile: str | None = None,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    selected_profile = str(profile or manifest["profile"]).strip().lower()
    settings = _profile_settings(selected_profile)

    hardware = detect_hardware(root)
    readiness = training_readiness(
        root=root,
        train_parquet=manifest["train_parquet"],
        validation_parquet=manifest["validation_parquet"],
        finbert_train_jsonl=manifest["finbert_train_jsonl"],
        finbert_validation_jsonl=manifest["finbert_validation_jsonl"],
        hardware=hardware,
    )
    if not readiness.get("ready"):
        missing = ", ".join(str(item) for item in readiness.get("missing", []))
        raise RuntimeError(f"Training campaign is not ready: {missing or 'unknown readiness failure'}")

    train_frame, validation_frame = _read_market_splits(
        manifest["train_parquet"],
        manifest["validation_parquet"],
    )
    finbert_train = _read_jsonl(manifest["finbert_train_jsonl"])
    finbert_validation = _read_jsonl(manifest["finbert_validation_jsonl"])

    runners = [
        (
            "ttm",
            run_ttm_training,
            {
                "root": root,
                "train_frame": train_frame,
                "validation_frame": validation_frame,
                "hardware": hardware,
                **settings["ttm"],
            },
        ),
        (
            "timesfm25",
            run_timesfm_training,
            {
                "root": root,
                "train_frame": train_frame,
                "validation_frame": validation_frame,
                "hardware": hardware,
                "preflight": _guarded_preflight(hardware, "TimesFM 2.5"),
                **settings["timesfm25"],
            },
        ),
        (
            "chronos2",
            run_chronos2_training,
            {
                "root": root,
                "train_frame": train_frame,
                "validation_frame": validation_frame,
                "hardware": hardware,
                "preflight": _guarded_preflight(hardware, "Chronos-2"),
                **settings["chronos2"],
            },
        ),
        (
            "kronos",
            run_kronos_training,
            {
                "root": root,
                "train_frame": train_frame,
                "validation_frame": validation_frame,
                "hardware": hardware,
                **settings["kronos"],
            },
        ),
        (
            "finbert",
            run_finbert_training,
            {
                "root": root,
                "train_examples": finbert_train,
                "validation_examples": finbert_validation,
                "hardware": hardware,
                **settings["finbert"],
            },
        ),
    ]

    outcomes: dict[str, dict[str, Any]] = {}
    for name, runner, kwargs in runners:
        try:
            outcomes[name] = _result_payload(runner(**kwargs))
        except Exception as exc:
            outcomes[name] = {
                "status": "FAILED",
                "run_id": "",
                "checkpoint": None,
                "metrics": {},
                "error": str(exc),
            }
        finally:
            _cleanup_cuda()

    return {
        "manifest": str(manifest["manifest_path"]),
        "profile": selected_profile,
        "models": outcomes,
    }
