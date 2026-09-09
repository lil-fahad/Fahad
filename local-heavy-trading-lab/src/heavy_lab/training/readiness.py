from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pandas as pd

from heavy_lab.data.schema import validate_canonical_frame
from heavy_lab.hardware import HardwareProfile, detect_hardware
from heavy_lab.models.catalog import MODEL_CATALOG
from heavy_lab.paths import LabPaths
from heavy_lab.training.policy import plan_training


def _valid_market_split(path: Path | None) -> tuple[bool, str | None, pd.DataFrame | None]:
    if path is None:
        return False, None, None
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        return False, str(candidate), None
    try:
        frame = pd.read_parquet(candidate)
        validate_canonical_frame(frame)
    except Exception:
        return False, str(candidate), None
    return True, str(candidate), frame


def _valid_finbert_jsonl(path: Path | None) -> tuple[bool, str | None]:
    if path is None:
        return False, None
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        return False, str(candidate)
    valid_rows = 0
    try:
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            item = json.loads(raw)
            if not isinstance(item, dict):
                return False, str(candidate)
            if not str(item.get("text", "")).strip():
                return False, str(candidate)
            if item.get("label") not in {0, 1, 2}:
                return False, str(candidate)
            valid_rows += 1
    except Exception:
        return False, str(candidate)
    return valid_rows > 0, str(candidate)


def training_readiness(
    *,
    root: Path,
    train_parquet: Path | None = None,
    validation_parquet: Path | None = None,
    finbert_train_jsonl: Path | None = None,
    finbert_validation_jsonl: Path | None = None,
    hardware: HardwareProfile | None = None,
) -> dict[str, Any]:
    paths = LabPaths.from_root(Path(root))
    hardware = hardware or detect_hardware(paths.root)
    missing: list[str] = []

    model_status: dict[str, bool] = {}
    for name, spec in MODEL_CATALOG.items():
        model_root = paths.models_base / spec.local_name
        ready = model_root.is_dir() and (model_root / "manifest.json").is_file()
        model_status[name] = ready
        if not ready:
            missing.append(f"model:{name}")

    train_ok, train_path, train_frame = _valid_market_split(train_parquet)
    validation_ok, validation_path, validation_frame = _valid_market_split(validation_parquet)
    if not train_ok:
        missing.append("train_parquet")
    if not validation_ok:
        missing.append("validation_parquet")

    chronological = False
    if train_frame is not None and validation_frame is not None:
        train_last = train_frame["timestamp_utc"].max()
        validation_first = validation_frame["timestamp_utc"].min()
        chronological = bool(train_last < validation_first)
        if not chronological:
            missing.append("market_split_chronology")

    finbert_train_ok, finbert_train_path = _valid_finbert_jsonl(finbert_train_jsonl)
    finbert_validation_ok, finbert_validation_path = _valid_finbert_jsonl(finbert_validation_jsonl)
    if not finbert_train_ok:
        missing.append("finbert_train_jsonl")
    if not finbert_validation_ok:
        missing.append("finbert_validation_jsonl")

    trainable_models = ("ttm", "timesfm25", "chronos2", "kronos", "finbert")
    plans = {name: asdict(plan_training(name, hardware)) for name in trainable_models}

    # The heavy local program is intentionally CUDA-gated for the user's
    # maximum-compute profile. CPU/MPS can still use individual commands, but
    # a full five-model campaign is not declared ready without CUDA here.
    if not hardware.cuda:
        missing.append("cuda")

    return {
        "ready": not missing,
        "missing": missing,
        "hardware": hardware.as_dict(),
        "models": model_status,
        "market_data": {
            "train_parquet": train_path,
            "validation_parquet": validation_path,
            "train_valid": train_ok,
            "validation_valid": validation_ok,
            "chronological": chronological,
        },
        "finbert_data": {
            "train_jsonl": finbert_train_path,
            "validation_jsonl": finbert_validation_path,
            "train_valid": finbert_train_ok,
            "validation_valid": finbert_validation_ok,
        },
        "plans": plans,
    }
