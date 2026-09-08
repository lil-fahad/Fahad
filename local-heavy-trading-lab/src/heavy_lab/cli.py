from __future__ import annotations

import json
from pathlib import Path

import typer

from heavy_lab import __version__
from heavy_lab.paths import LabPaths


app = typer.Typer(no_args_is_help=True, help="Local Heavy Trading Lab")


def _echo_train_result(result) -> None:
    typer.echo(
        json.dumps(
            {
                "run_id": result.run_id,
                "status": result.status,
                "checkpoint": result.checkpoint,
                "metrics": result.metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command()
def version() -> None:
    """Print the local lab version."""
    typer.echo(__version__)


@app.command()
def init(root: Path = typer.Option(Path.cwd(), "--root")) -> None:
    """Create the local runtime directories without downloading models."""
    paths = LabPaths.from_root(root)
    paths.ensure_runtime_dirs()
    typer.echo(json.dumps({"root": str(paths.root), "status": "initialized"}))


@app.command()
def doctor(
    root: Path = typer.Option(Path.cwd(), "--root"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Inspect local hardware and storage capabilities."""
    from heavy_lab.doctor import doctor_report

    report = doctor_report(root)
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        typer.echo(f"device={report['hardware']['device']} profile={report['training_profile']}")


@app.command("download-models")
def download_models_command(
    root: Path = typer.Option(Path.cwd(), "--root"),
    all_models: bool = typer.Option(False, "--all", help="Download every required base model snapshot."),
    model: list[str] | None = typer.Option(None, "--model", help="Download one or more named models."),
) -> None:
    """Download full Hugging Face model snapshots and write immutable manifests."""
    from heavy_lab.models.catalog import MODEL_CATALOG
    from heavy_lab.models.download import download_all, download_models

    paths = LabPaths.from_root(root)
    paths.ensure_runtime_dirs()
    selected = list(model or [])
    if all_models and selected:
        raise typer.BadParameter("Use either --all or --model, not both.")
    if not all_models and not selected:
        raise typer.BadParameter("Specify --all or at least one --model.")
    unknown = [name for name in selected if name not in MODEL_CATALOG]
    if unknown:
        raise typer.BadParameter(f"Unknown model(s): {', '.join(unknown)}")

    manifests = download_all(paths) if all_models else download_models(selected, paths)
    payload = [
        {
            "name": item.name,
            "repo_id": item.repo_id,
            "resolved_revision": item.resolved_revision,
            "local_path": item.local_path,
            "total_bytes": item.total_bytes,
            "manifest_path": item.manifest_path,
        }
        for item in manifests
    ]
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("train-ttm")
def train_ttm_command(
    train_parquet: Path = typer.Option(..., "--train-parquet", exists=True, dir_okay=False, readable=True),
    validation_parquet: Path = typer.Option(..., "--validation-parquet", exists=True, dir_okay=False, readable=True),
    root: Path = typer.Option(Path.cwd(), "--root"),
    context_length: int = typer.Option(512, "--context-length", min=2),
    prediction_length: int = typer.Option(96, "--prediction-length", min=1),
    epochs: int = typer.Option(1, "--epochs", min=1),
    learning_rate: float = typer.Option(1e-4, "--learning-rate", min=1e-12),
) -> None:
    """Fine-tune IBM TTM locally on explicit chronological Parquet splits."""
    import pandas as pd

    from heavy_lab.training.launch import run_ttm_training

    train_frame = pd.read_parquet(train_parquet)
    validation_frame = pd.read_parquet(validation_parquet)
    result = run_ttm_training(
        root=root,
        train_frame=train_frame,
        validation_frame=validation_frame,
        context_length=context_length,
        prediction_length=prediction_length,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    _echo_train_result(result)


@app.command("train-timesfm25")
def train_timesfm25_command(
    train_parquet: Path = typer.Option(..., "--train-parquet", exists=True, dir_okay=False, readable=True),
    validation_parquet: Path = typer.Option(..., "--validation-parquet", exists=True, dir_okay=False, readable=True),
    root: Path = typer.Option(Path.cwd(), "--root"),
    context_length: int = typer.Option(512, "--context-length", min=2),
    prediction_length: int = typer.Option(96, "--prediction-length", min=1),
    epochs: int = typer.Option(1, "--epochs", min=1),
    learning_rate: float = typer.Option(1e-4, "--learning-rate", min=1e-12),
    micro_batch_size: int = typer.Option(1, "--micro-batch-size", min=1),
) -> None:
    """Fine-tune TimesFM 2.5 locally; 8 GB-class GPUs use LoRA by default."""
    import pandas as pd

    from heavy_lab.hardware import detect_hardware
    from heavy_lab.training.base import MemoryProbeResult
    from heavy_lab.training.launch import run_timesfm_training

    train_frame = pd.read_parquet(train_parquet)
    validation_frame = pd.read_parquet(validation_parquet)
    hardware = detect_hardware(Path(root))
    vram_gb = float(hardware.vram_gb or 0.0)
    safe_full = bool(hardware.cuda and vram_gb >= 24.0)
    preflight = MemoryProbeResult(
        safe_full_finetune=safe_full,
        max_micro_batch_size=int(micro_batch_size),
        reason=(
            "hardware full-finetune gate passed; run a memory smoke test before increasing batch size"
            if safe_full
            else f"TimesFM full fine-tuning requires at least 24 GB VRAM; detected {vram_gb:g} GB, using LoRA"
        ),
        peak_memory_gb=None,
    )
    result = run_timesfm_training(
        root=root,
        train_frame=train_frame,
        validation_frame=validation_frame,
        context_length=context_length,
        prediction_length=prediction_length,
        epochs=epochs,
        learning_rate=learning_rate,
        hardware=hardware,
        preflight=preflight,
    )
    _echo_train_result(result)


if __name__ == "__main__":
    app()
