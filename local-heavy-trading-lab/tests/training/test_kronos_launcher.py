from pathlib import Path

import pandas as pd


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


def test_run_kronos_training_executes_pinned_official_flow(tmp_path: Path):
    from heavy_lab.hardware import HardwareProfile
    from heavy_lab.training.launch import run_kronos_training

    train = _bars("2026-08-01T13:30:00Z", 40)
    validation = _bars("2026-08-02T13:30:00Z", 20)

    for name in ("kronos-base", "kronos-tokenizer-base"):
        model_dir = tmp_path / "models" / "base" / name
        model_dir.mkdir(parents=True)
        (model_dir / "manifest.json").write_text("{}", encoding="utf-8")

    calls = {"ensure": 0, "prepare": 0, "run": 0}
    source = tmp_path / "vendor" / "kronos-pinned"
    source.mkdir(parents=True)

    def ensure_source(**kwargs):
        calls["ensure"] += 1
        return source

    class Prepared:
        csv_path = tmp_path / "work" / "spy.csv"
        config_path = tmp_path / "work" / "config.yaml"

    def prepare_files(**kwargs):
        calls["prepare"] += 1
        Prepared.csv_path.parent.mkdir(parents=True, exist_ok=True)
        Prepared.csv_path.write_text("timestamps,open,high,low,close,volume,amount\n", encoding="utf-8")
        Prepared.config_path.write_text("{}", encoding="utf-8")
        assert kwargs["batch_size"] == 1
        assert kwargs["accumulation_steps"] >= 8
        return Prepared()

    class Completed:
        returncode = 0
        stdout = "completed"
        stderr = ""

    def run_sequential(**kwargs):
        calls["run"] += 1
        assert kwargs["source_root"] == source
        return Completed()

    hardware = HardwareProfile(
        os_name="Windows",
        cpu_count=16,
        device="cuda",
        cuda=True,
        mps=False,
        ram_gb=32.0,
        vram_gb=8.0,
        disk_free_gb=200.0,
        gpu_name="NVIDIA GeForce RTX 3070 Ti",
        cuda_version="12.8",
        compute_capability="8.6",
        bf16=True,
        fp16=True,
    )

    result = run_kronos_training(
        root=tmp_path,
        train_frame=train,
        validation_frame=validation,
        lookback_window=8,
        predict_window=4,
        tokenizer_epochs=1,
        predictor_epochs=1,
        hardware=hardware,
        ensure_source=ensure_source,
        prepare_files=prepare_files,
        run_sequential=run_sequential,
    )

    assert result.status == "COMPLETED"
    assert calls == {"ensure": 1, "prepare": 1, "run": 1}
    checkpoint = Path(result.checkpoint)
    assert (checkpoint / "checkpoint.ok").read_text(encoding="utf-8") == "ok"
