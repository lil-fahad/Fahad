from typer.testing import CliRunner


def _help_output() -> str:
    from heavy_lab.cli import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    return result.stdout


def test_cli_exposes_timesfm_training_command():
    assert "train-timesfm25" in _help_output()


def test_cli_exposes_chronos2_training_command():
    assert "train-chronos2" in _help_output()


def test_cli_exposes_kronos_training_command():
    assert "train-kronos" in _help_output()


def test_cli_exposes_finbert_training_command():
    assert "train-finbert" in _help_output()


def test_cli_exposes_train_all_command():
    assert "train-all" in _help_output()
