from typer.testing import CliRunner

from heavy_lab.cli import app


def test_train_ttm_command_is_exposed_with_explicit_split_files():
    runner = CliRunner()
    result = runner.invoke(app, ["train-ttm", "--help"])
    assert result.exit_code == 0
    assert "--train-parquet" in result.stdout
    assert "--validation-parquet" in result.stdout
    assert "--context-length" in result.stdout
    assert "--prediction-length" in result.stdout
    assert "--epochs" in result.stdout
