from __future__ import annotations

import json
from pathlib import Path

import typer

from heavy_lab import __version__
from heavy_lab.paths import LabPaths


app = typer.Typer(no_args_is_help=True, help="Local Heavy Trading Lab")


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


if __name__ == "__main__":
    app()
