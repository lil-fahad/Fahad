import json
from pathlib import Path

import pandas as pd


def _bars(start: str, count: int, symbol: str = "SPY") -> pd.DataFrame:
    ts = pd.date_range(start, periods=count, freq="5min", tz="UTC")
    close = [100.0 + i for i in range(count)]
    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "symbol": symbol,
            "open": close,
            "high": [x + 0.5 for x in close],
            "low": [x - 0.5 for x in close],
            "close": close,
            "volume": [1000.0 + i for i in range(count)],
            "session": "regular",
            "source": "test",
            "source_revision": "v1",
        }
    )


def test_kronos_official_preparation_preserves_purged_time_gap(tmp_path: Path):
    from heavy_lab.training.integrations.kronos_official import prepare_kronos_official_files

    train = _bars("2026-09-01T13:30:00Z", 20)
    validation = _bars("2026-09-02T13:30:00Z", 12)
    model_path = tmp_path / "models" / "kronos-base"
    tokenizer_path = tmp_path / "models" / "kronos-tokenizer-base"
    output_root = tmp_path / "checkpoint"

    prepared = prepare_kronos_official_files(
        train_frame=train,
        validation_frame=validation,
        work_dir=tmp_path / "work",
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        output_root=output_root,
        lookback_window=4,
        predict_window=2,
        batch_size=1,
        accumulation_steps=8,
        tokenizer_epochs=1,
        predictor_epochs=1,
        use_cuda=True,
    )

    exported = pd.read_csv(prepared.csv_path)
    assert list(exported.columns) == ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    assert len(exported) == 32
    assert exported["amount"].eq(0.0).all()
    assert pd.Timestamp(exported.iloc[19]["timestamps"]) < pd.Timestamp(exported.iloc[20]["timestamps"])
    assert pd.Timestamp(exported.iloc[20]["timestamps"]) - pd.Timestamp(exported.iloc[19]["timestamps"]) > pd.Timedelta("5min")

    config = json.loads(prepared.config_path.read_text(encoding="utf-8"))
    assert config["data"]["data_path"] == str(prepared.csv_path.resolve())
    assert int(32 * config["data"]["train_ratio"]) == 20
    assert config["data"]["val_ratio"] > 0
    assert config["data"]["test_ratio"] == 0.0
    assert config["training"]["batch_size"] == 1
    assert config["training"]["accumulation_steps"] == 8
    assert config["model_paths"]["pretrained_predictor"] == str(model_path.resolve())
    assert config["model_paths"]["pretrained_tokenizer"] == str(tokenizer_path.resolve())
    assert config["model_paths"]["base_save_path"] == str(output_root.resolve())
    assert config["device"]["use_cuda"] is True
    assert config["distributed"]["use_ddp"] is False


def test_kronos_official_runner_uses_pinned_source_and_sequential_script(tmp_path: Path):
    from heavy_lab.training.integrations.kronos_official import (
        KRONOS_SOURCE_REVISION,
        run_kronos_sequential,
    )

    source = tmp_path / "kronos-source"
    finetune_dir = source / "finetune_csv"
    finetune_dir.mkdir(parents=True)
    script = finetune_dir / "train_sequential.py"
    script.write_text("# test fixture", encoding="utf-8")
    (source / ".source.json").write_text(
        json.dumps({"revision": KRONOS_SOURCE_REVISION}), encoding="utf-8"
    )
    config = tmp_path / "config.yaml"
    config.write_text("{}", encoding="utf-8")

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

        class Result:
            returncode = 0
            stdout = "training completed"
            stderr = ""

        return Result()

    result = run_kronos_sequential(source_root=source, config_path=config, run_command=fake_run)
    assert result.returncode == 0
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert Path(command[1]).resolve() == script.resolve()
    assert command[2:] == ["--config", str(config.resolve())]
    assert Path(kwargs["cwd"]).resolve() == finetune_dir.resolve()
    assert kwargs["check"] is True
