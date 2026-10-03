from __future__ import annotations

from pathlib import Path
import tomllib

from click import unstyle
from typer.testing import CliRunner


def test_pyproject_exposes_lab_bot_script():
    root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["scripts"]["lab-bot"] == "heavy_lab.bot.runtime:main"


def test_lab_cli_exposes_bot_command_help():
    from heavy_lab.cli import app

    result = CliRunner().invoke(app, ["bot", "--help"])
    assert result.exit_code == 0, result.stdout
    output = unstyle(result.stdout)
    assert "--root" in output
    assert "--manifest" in output
