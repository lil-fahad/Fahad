from pathlib import Path

import pandas as pd

from heavy_lab.hardware import HardwareProfile
from heavy_lab.training.base import MemoryProbeResult


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


def test_timesfm_launcher_forces_lora_and_writes_adapter_checkpoint(tmp_path: Path):
    from heavy_lab.training.launch import run_timesfm_training

    calls = {"lora": []}

    class FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.trained = False

        def evaluate(self, examples):
            return 0.5 if not self.trained else 0.25

        def train_step(self, *, past_values, future_values, learning_rate, use_lora):
            calls["lora"].append(use_lora)
            self.trained = True
            return 0.3

        def save_adapter(self, path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter.ok").write_text("ok", encoding="utf-8")

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
    preflight = MemoryProbeResult(
        safe_full_finetune=False,
        max_micro_batch_size=1,
        reason="8 GB VRAM below 24 GB full gate",
        peak_memory_gb=6.0,
    )

    result = run_timesfm_training(
        root=tmp_path,
        train_frame=_bars("2026-09-01", 20),
        validation_frame=_bars("2026-09-02", 12),
        context_length=4,
        prediction_length=2,
        epochs=1,
        learning_rate=1e-4,
        hardware=hardware,
        preflight=preflight,
        adapter_factory=FakeAdapter,
    )

    assert result.status == "COMPLETED"
    assert calls["lora"] and all(calls["lora"])
    checkpoint = Path(result.checkpoint)
    assert (checkpoint / "checkpoint.ok").exists()
    assert (checkpoint / "adapter" / "adapter.ok").exists()
