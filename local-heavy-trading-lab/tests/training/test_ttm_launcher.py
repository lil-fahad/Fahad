from pathlib import Path

import pandas as pd

from heavy_lab.hardware import HardwareProfile


def _bars(start: str, count: int) -> pd.DataFrame:
    ts = pd.date_range(start, periods=count, freq="5min", tz="UTC")
    close = [100.0 + i for i in range(count)]
    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "symbol": "SPY",
            "open": close,
            "high": [x + 0.5 for x in close],
            "low": [x - 0.5 for x in close],
            "close": close,
            "volume": 1000.0,
            "session": "regular",
            "source": "test",
            "source_revision": "v1",
        }
    )


def test_ttm_launcher_creates_real_run_and_checkpoint_with_injected_adapter(tmp_path: Path):
    from heavy_lab.training.launch import run_ttm_training

    calls = {"train": 0, "evaluate": 0}

    class FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def evaluate(self, dataset):
            calls["evaluate"] += 1
            return 0.5 if calls["evaluate"] == 1 else 0.25

        def train_epoch(self, dataset, *, learning_rate):
            calls["train"] += 1
            return 0.3

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "weights.ok").write_text("ok", encoding="utf-8")

    hardware = HardwareProfile(
        os_name="Windows",
        cpu_count=16,
        device="cuda",
        cuda=True,
        mps=False,
        ram_gb=32.0,
        vram_gb=8.0,
        disk_free_gb=500.0,
        gpu_name="RTX 3070 Ti",
        cuda_version="12.8",
        compute_capability="8.6",
        bf16=False,
        fp16=True,
    )

    result = run_ttm_training(
        root=tmp_path,
        train_frame=_bars("2026-09-01", 20),
        validation_frame=_bars("2026-09-02", 12),
        context_length=5,
        prediction_length=2,
        epochs=2,
        learning_rate=1e-4,
        hardware=hardware,
        adapter_factory=FakeAdapter,
    )

    assert result.status == "COMPLETED"
    assert calls["train"] == 2
    assert calls["evaluate"] == 2
    checkpoint = Path(result.checkpoint)
    assert (checkpoint / "checkpoint.ok").exists()
    assert (checkpoint / "model" / "weights.ok").exists()
