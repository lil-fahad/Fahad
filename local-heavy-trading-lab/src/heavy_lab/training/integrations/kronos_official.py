from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import pandas as pd

from heavy_lab.data.schema import validate_canonical_frame


KRONOS_SOURCE_REPOSITORY = "https://github.com/shiyu-coder/Kronos.git"
KRONOS_SOURCE_REVISION = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"


@dataclass(frozen=True)
class KronosOfficialFiles:
    csv_path: Path
    config_path: Path


def ensure_kronos_source(
    *,
    vendor_root: Path,
    run_command: Callable[..., Any] = subprocess.run,
) -> Path:
    vendor_root = Path(vendor_root).expanduser().resolve()
    vendor_root.mkdir(parents=True, exist_ok=True)
    source_root = vendor_root / f"kronos-{KRONOS_SOURCE_REVISION[:12]}"
    marker_path = source_root / ".source.json"

    if source_root.exists():
        if not marker_path.is_file():
            raise RuntimeError(f"Existing Kronos source is missing revision marker: {source_root}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        expected = {
            "repository": KRONOS_SOURCE_REPOSITORY,
            "revision": KRONOS_SOURCE_REVISION,
        }
        if marker != expected:
            raise RuntimeError(
                "Existing Kronos source marker does not match the pinned source: "
                f"expected {expected}, found {marker}"
            )
        return source_root

    run_command(
        [
            "git",
            "clone",
            "--filter=blob:none",
            KRONOS_SOURCE_REPOSITORY,
            str(source_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    run_command(
        ["git", "-C", str(source_root), "checkout", "--detach", KRONOS_SOURCE_REVISION],
        check=True,
        capture_output=True,
        text=True,
    )

    script_path = source_root / "finetune_csv" / "train_sequential.py"
    if not script_path.is_file():
        raise FileNotFoundError(
            f"Pinned Kronos source does not contain the expected training script: {script_path}"
        )

    marker_path.write_text(
        json.dumps(
            {
                "repository": KRONOS_SOURCE_REPOSITORY,
                "revision": KRONOS_SOURCE_REVISION,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return source_root


def _single_matching_symbol(train_frame: pd.DataFrame, validation_frame: pd.DataFrame) -> str:
    train_symbols = list(train_frame["symbol"].drop_duplicates())
    validation_symbols = list(validation_frame["symbol"].drop_duplicates())
    if len(train_symbols) != 1 or len(validation_symbols) != 1:
        raise ValueError("Kronos official CSV training requires exactly one symbol per run")
    if train_symbols[0] != validation_symbols[0]:
        raise ValueError("Kronos train and validation symbols must match")
    return str(train_symbols[0])


def prepare_kronos_official_files(
    *,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    work_dir: Path,
    model_path: Path,
    tokenizer_path: Path,
    output_root: Path,
    lookback_window: int,
    predict_window: int,
    batch_size: int,
    accumulation_steps: int,
    tokenizer_epochs: int,
    predictor_epochs: int,
    use_cuda: bool,
) -> KronosOfficialFiles:
    validate_canonical_frame(train_frame)
    validate_canonical_frame(validation_frame)
    symbol = _single_matching_symbol(train_frame, validation_frame)

    if lookback_window < 2 or predict_window < 1:
        raise ValueError("Kronos lookback/predict windows must be positive")
    if batch_size < 1 or accumulation_steps < 1:
        raise ValueError("Kronos batch size and accumulation steps must be positive")
    if tokenizer_epochs < 1 or predictor_epochs < 1:
        raise ValueError("Kronos epochs must be positive")

    train_last = train_frame["timestamp_utc"].max()
    validation_first = validation_frame["timestamp_utc"].min()
    if not train_last < validation_first:
        raise ValueError("Kronos validation data must begin strictly after training data")

    minimum_rows = int(lookback_window) + int(predict_window) + 1
    if len(train_frame) < minimum_rows or len(validation_frame) < minimum_rows:
        raise ValueError(
            f"Kronos train and validation splits each need at least {minimum_rows} rows"
        )

    work_dir = Path(work_dir).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    tokenizer_path = Path(tokenizer_path).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    combined = pd.concat([train_frame, validation_frame], ignore_index=True)
    exported = pd.DataFrame(
        {
            "timestamps": combined["timestamp_utc"].map(lambda value: pd.Timestamp(value).isoformat()),
            "open": combined["open"].astype(float),
            "high": combined["high"].astype(float),
            "low": combined["low"].astype(float),
            "close": combined["close"].astype(float),
            "volume": combined["volume"].astype(float),
            "amount": 0.0,
        }
    )
    csv_path = work_dir / f"{symbol.lower()}-kronos.csv"
    exported.to_csv(csv_path, index=False)

    total_rows = len(combined)
    train_rows = len(train_frame)
    # Kronos uses int(total * train_ratio). Put the ratio safely inside the
    # interval that maps to exactly train_rows instead of relying on a float
    # representation of train_rows / total_rows at the integer boundary.
    train_ratio = (train_rows + 0.25) / total_rows
    val_ratio = 1.0 - train_ratio

    config: dict[str, Any] = {
        "data": {
            "data_path": str(csv_path.resolve()),
            "lookback_window": int(lookback_window),
            "predict_window": int(predict_window),
            "max_context": int(lookback_window),
            "clip": 5.0,
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": 0.0,
        },
        "training": {
            "tokenizer_epochs": int(tokenizer_epochs),
            "basemodel_epochs": int(predictor_epochs),
            "batch_size": int(batch_size),
            "log_interval": 10,
            "num_workers": 0,
            "seed": 100,
            "tokenizer_learning_rate": 2e-4,
            "predictor_learning_rate": 1e-6,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_weight_decay": 0.1,
            "accumulation_steps": int(accumulation_steps),
        },
        "model_paths": {
            "exp_name": output_root.name,
            "base_path": str(output_root.parent),
            "pretrained_tokenizer": str(tokenizer_path),
            "pretrained_predictor": str(model_path),
            "base_save_path": str(output_root),
            "tokenizer_save_name": "tokenizer",
            "basemodel_save_name": "basemodel",
            "finetuned_tokenizer": "",
        },
        "experiment": {
            "name": f"fahad_kronos_{symbol.lower()}",
            "description": "Pinned official Kronos sequential local fine-tuning",
            "use_comet": False,
            "train_tokenizer": True,
            "train_basemodel": True,
            "skip_existing": False,
            "pre_trained": True,
        },
        "device": {"use_cuda": bool(use_cuda), "device_id": 0},
        "distributed": {"use_ddp": False, "backend": "nccl"},
        "source": {
            "repository": KRONOS_SOURCE_REPOSITORY,
            "revision": KRONOS_SOURCE_REVISION,
        },
    }
    config_path = work_dir / "kronos-config.yaml"
    # JSON is valid YAML and avoids a hard PyYAML import in the lab core.
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return KronosOfficialFiles(csv_path=csv_path, config_path=config_path)


def run_kronos_sequential(
    *,
    source_root: Path,
    config_path: Path,
    run_command: Callable[..., Any] = subprocess.run,
):
    source_root = Path(source_root).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    marker_path = source_root / ".source.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"Kronos source revision marker not found: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("revision") != KRONOS_SOURCE_REVISION:
        raise RuntimeError(
            "Kronos source revision mismatch: "
            f"expected {KRONOS_SOURCE_REVISION}, found {marker.get('revision')!r}"
        )

    finetune_dir = source_root / "finetune_csv"
    script_path = finetune_dir / "train_sequential.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Kronos sequential training script not found: {script_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Kronos config not found: {config_path}")

    return run_command(
        [sys.executable, str(script_path.resolve()), "--config", str(config_path)],
        cwd=str(finetune_dir.resolve()),
        check=True,
        capture_output=True,
        text=True,
    )
