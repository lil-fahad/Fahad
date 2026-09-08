from typer.testing import CliRunner


def test_cli_exposes_timesfm_training_command():
    from heavy_lab.cli import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train-timesfm25" in result.stdout
