from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from typer.testing import CliRunner


def _bars(start: str, count: int) -> pd.DataFrame:
    ts = pd.date_range(start, periods=count, freq="5min", tz="UTC")
    close = [100.0 + i * 0.1 for i in range(count)]
    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "symbol": "SPY",
            "open": close,
            "high": [x + 0.2 for x in close],
            "low": [x - 0.2 for x in close],
            "close": close,
            "volume": [1000.0 + i for i in range(count)],
            "session": "regular",
            "source": "fixture",
            "source_revision": "v1",
        }
    )


def test_train_kronos_cli_invokes_local_launcher(tmp_path: Path, monkeypatch):
    import heavy_lab.training.launch as launch
    from heavy_lab.cli import app

    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"
    _bars("2026-08-01T13:30:00Z", 40).to_parquet(train_path, index=False)
    _bars("2026-08-02T13:30:00Z", 20).to_parquet(validation_path, index=False)

    captured = {}

    def fake_run_kronos_training(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            run_id="20260909-kronos-test",
            status="COMPLETED",
            checkpoint=str(tmp_path / "checkpoints" / "kronos"),
            metrics={},
        )

    monkeypatch.setattr(launch, "run_kronos_training", fake_run_kronos_training)
    result = CliRunner().invoke(
        app,
        [
            "train-kronos",
            str(train_path),
            str(validation_path),
            "--root",
            str(tmp_path),
            "--lookback-window",
            "8",
            "--predict-window",
            "4",
            "--tokenizer-epochs",
            "1",
            "--predictor-epochs",
            "1",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "20260909-kronos-test" in result.stdout
    assert captured["root"] == tmp_path.resolve()
    assert captured["lookback_window"] == 8
    assert captured["predict_window"] == 4
    assert len(captured["train_frame"]) == 40
    assert len(captured["validation_frame"]) == 20
