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


if __name__ == "__main__":
    app()
