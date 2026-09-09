from __future__ import annotations

import json
from pathlib import Path

import typer

from heavy_lab import __version__
from heavy_lab.paths import LabPaths


app = typer.Typer(no_args_is_help=True, help="Local Heavy Trading Lab")


def _echo_train_result(result) -> None:
    typer.echo(json.dumps({"run_id": result.run_id, "status": result.status, "checkpoint": result.checkpoint, "metrics": result.metrics}, indent=2, sort_keys=True))


def _heavy_preflight(hardware, micro_batch_size: int, model_name: str):
    from heavy_lab.training.base import MemoryProbeResult

    vram_gb = float(hardware.vram_gb or 0.0)
    safe_full = bool(hardware.cuda and vram_gb >= 24.0)
    return MemoryProbeResult(
        safe_full_finetune=safe_full,
        max_micro_batch_size=int(micro_batch_size),
        reason=(
            "hardware full-finetune gate passed; run a memory smoke test before increasing batch size"
            if safe_full
            else f"{model_name} full fine-tuning requires at least 24 GB VRAM; detected {vram_gb:g} GB, using LoRA"
        ),
        peak_memory_gb=None,
    )


def _integration_not_ready(model_name: str) -> None:
    raise typer.BadParameter(
        f"{model_name} local training integration is not wired yet. "
        "The command is registered deliberately so dependency/setup gaps fail explicitly instead of silently falling back."
    )


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"Invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
        if not isinstance(item, dict):
            raise typer.BadParameter(f"JSONL row {line_number} in {path} must be an object")
        rows.append(item)
    if not rows:
        raise typer.BadParameter(f"JSONL file is empty: {path}")
    return rows


@app.command()
def version() -> None:
    typer.echo(__version__)


@app.command()
def init(root: Path = typer.Option(Path.cwd(), "--root")) -> None:
    paths = LabPaths.from_root(root)
    paths.ensure_runtime_dirs()
    typer.echo(json.dumps({"root": str(paths.root), "status": "initialized"}))


@app.command()
def doctor(root: Path = typer.Option(Path.cwd(), "--root"), json_output: bool = typer.Option(False, "--json")) -> None:
    from heavy_lab.doctor import doctor_report
    report = doctor_report(root)
    typer.echo(json.dumps(report, indent=2, sort_keys=True) if json_output else f"device={report['hardware']['device']} profile={report['training_profile']}")


@app.command("training-ready")
def training_ready_command(
    root: Path = typer.Option(Path.cwd(), "--root"),
    train_parquet: Path | None = typer.Option(None, "--train-parquet"),
    validation_parquet: Path | None = typer.Option(None, "--validation-parquet"),
    finbert_train_jsonl: Path | None = typer.Option(None, "--finbert-train-jsonl"),
    finbert_validation_jsonl: Path | None = typer.Option(None, "--finbert-validation-jsonl"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from heavy_lab.training.readiness import training_readiness

    report = training_readiness(
        root=Path(root).expanduser().resolve(),
        train_parquet=train_parquet,
        validation_parquet=validation_parquet,
        finbert_train_jsonl=finbert_train_jsonl,
        finbert_validation_jsonl=finbert_validation_jsonl,
    )
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
        return
    if report["ready"]:
        typer.echo("READY: all local heavy-training requirements are satisfied")
    else:
        typer.echo("NOT READY: " + ", ".join(report["missing"]))


@app.command("download-models")
def download_models_command(root: Path = typer.Option(Path.cwd(), "--root"), all_models: bool = typer.Option(False, "--all"), model: list[str] | None = typer.Option(None, "--model")) -> None:
    from heavy_lab.models.catalog import MODEL_CATALOG
    from heavy_lab.models.download import download_all, download_models
    paths = LabPaths.from_root(root); paths.ensure_runtime_dirs()
    selected = list(model or [])
    if all_models and selected: raise typer.BadParameter("Use either --all or --model, not both.")
    if not all_models and not selected: raise typer.BadParameter("Specify --all or at least one --model.")
    unknown = [name for name in selected if name not in MODEL_CATALOG]
    if unknown: raise typer.BadParameter(f"Unknown model(s): {', '.join(unknown)}")
    manifests = download_all(paths) if all_models else download_models(selected, paths)
    typer.echo(json.dumps([{"name": item.name, "repo_id": item.repo_id, "resolved_revision": item.resolved_revision, "local_path": item.local_path, "total_bytes": item.total_bytes, "manifest_path": item.manifest_path} for item in manifests], indent=2, sort_keys=True))


def _read_splits(train_parquet: Path, validation_parquet: Path):
    import pandas as pd
    return pd.read_parquet(train_parquet), pd.read_parquet(validation_parquet)


@app.command("train-ttm")
def train_ttm_command(train_parquet: Path = typer.Option(..., "--train-parquet", exists=True, dir_okay=False, readable=True), validation_parquet: Path = typer.Option(..., "--validation-parquet", exists=True, dir_okay=False, readable=True), root: Path = typer.Option(Path.cwd(), "--root"), context_length: int = typer.Option(512, "--context-length", min=2), prediction_length: int = typer.Option(96, "--prediction-length", min=1), epochs: int = typer.Option(1, "--epochs", min=1), learning_rate: float = typer.Option(1e-4, "--learning-rate", min=1e-12)) -> None:
    from heavy_lab.training.launch import run_ttm_training
    train_frame, validation_frame = _read_splits(train_parquet, validation_parquet)
    _echo_train_result(run_ttm_training(root=root, train_frame=train_frame, validation_frame=validation_frame, context_length=context_length, prediction_length=prediction_length, epochs=epochs, learning_rate=learning_rate))


@app.command("train-timesfm25")
def train_timesfm25_command(train_parquet: Path = typer.Option(..., "--train-parquet", exists=True, dir_okay=False, readable=True), validation_parquet: Path = typer.Option(..., "--validation-parquet", exists=True, dir_okay=False, readable=True), root: Path = typer.Option(Path.cwd(), "--root"), context_length: int = typer.Option(512, "--context-length", min=2), prediction_length: int = typer.Option(96, "--prediction-length", min=1), epochs: int = typer.Option(1, "--epochs", min=1), learning_rate: float = typer.Option(1e-4, "--learning-rate", min=1e-12), micro_batch_size: int = typer.Option(1, "--micro-batch-size", min=1)) -> None:
    from heavy_lab.hardware import detect_hardware
    from heavy_lab.training.launch import run_timesfm_training
    train_frame, validation_frame = _read_splits(train_parquet, validation_parquet)
    hardware = detect_hardware(Path(root))
    preflight = _heavy_preflight(hardware, micro_batch_size, "TimesFM 2.5")
    _echo_train_result(run_timesfm_training(root=root, train_frame=train_frame, validation_frame=validation_frame, context_length=context_length, prediction_length=prediction_length, epochs=epochs, learning_rate=learning_rate, hardware=hardware, preflight=preflight))


@app.command("train-chronos2")
def train_chronos2_command(train_parquet: Path = typer.Option(..., "--train-parquet", exists=True, dir_okay=False, readable=True), validation_parquet: Path = typer.Option(..., "--validation-parquet", exists=True, dir_okay=False, readable=True), root: Path = typer.Option(Path.cwd(), "--root"), context_length: int = typer.Option(512, "--context-length", min=2), prediction_length: int = typer.Option(96, "--prediction-length", min=1), steps: int = typer.Option(1000, "--steps", min=1), learning_rate: float = typer.Option(1e-5, "--learning-rate", min=1e-12), micro_batch_size: int = typer.Option(1, "--micro-batch-size", min=1)) -> None:
    from heavy_lab.hardware import detect_hardware
    from heavy_lab.training.launch import run_chronos2_training
    train_frame, validation_frame = _read_splits(train_parquet, validation_parquet)
    hardware = detect_hardware(Path(root))
    preflight = _heavy_preflight(hardware, micro_batch_size, "Chronos-2")
    _echo_train_result(run_chronos2_training(root=root, train_frame=train_frame, validation_frame=validation_frame, context_length=context_length, prediction_length=prediction_length, steps=steps, learning_rate=learning_rate, hardware=hardware, preflight=preflight))


@app.command("train-kronos")
def train_kronos_command(
    train_parquet: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    validation_parquet: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    root: Path = typer.Option(Path.cwd(), "--root"),
    lookback_window: int = typer.Option(64, "--lookback-window", min=2),
    predict_window: int = typer.Option(16, "--predict-window", min=1),
    tokenizer_epochs: int = typer.Option(1, "--tokenizer-epochs", min=1),
    predictor_epochs: int = typer.Option(1, "--predictor-epochs", min=1),
) -> None:
    from heavy_lab.training.launch import run_kronos_training

    train_frame, validation_frame = _read_splits(train_parquet, validation_parquet)
    _echo_train_result(
        run_kronos_training(
            root=Path(root).expanduser().resolve(),
            train_frame=train_frame,
            validation_frame=validation_frame,
            lookback_window=lookback_window,
            predict_window=predict_window,
            tokenizer_epochs=tokenizer_epochs,
            predictor_epochs=predictor_epochs,
        )
    )


@app.command("train-finbert")
def train_finbert_command(
    train_jsonl: Path = typer.Option(..., "--train-jsonl", exists=True, dir_okay=False, readable=True),
    validation_jsonl: Path = typer.Option(..., "--validation-jsonl", exists=True, dir_okay=False, readable=True),
    root: Path = typer.Option(Path.cwd(), "--root"),
    epochs: int = typer.Option(1, "--epochs", min=1),
    learning_rate: float = typer.Option(1e-5, "--learning-rate", min=1e-12),
) -> None:
    from heavy_lab.training.launch import run_finbert_training

    train_examples = _read_jsonl(train_jsonl)
    validation_examples = _read_jsonl(validation_jsonl)
    _echo_train_result(
        run_finbert_training(
            root=root,
            train_examples=train_examples,
            validation_examples=validation_examples,
            epochs=epochs,
            learning_rate=learning_rate,
        )
    )


@app.command("train-all")
def train_all_command(root: Path = typer.Option(Path.cwd(), "--root")) -> None:
    _ = root
    raise typer.BadParameter(
        "train-all remains blocked until one reproducible launch manifest can supply market and labeled-text inputs. "
        "Use the individual training commands meanwhile; they run one heavy model at a time."
    )


if __name__ == "__main__":
    app()
