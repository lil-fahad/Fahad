import re

from typer.testing import CliRunner

from heavy_lab.cli import app


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def test_train_ttm_command_is_exposed_with_explicit_split_files():
    runner = CliRunner()
    result = runner.invoke(app, ["train-ttm", "--help"])
    assert result.exit_code == 0
    help_text = ANSI.sub("", result.stdout)
    assert "--train-parquet" in help_text
    assert "--validation-parquet" in help_text
    assert "--context-length" in help_text
    assert "--prediction-length" in help_text
    assert "--epochs" in help_text
