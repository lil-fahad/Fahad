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


def test_chronos_launcher_loads_local_snapshot_and_uses_lora_on_8gb(tmp_path: Path):
    from heavy_lab.training.launch import run_chronos2_training

    calls: list[dict] = []

    class FineTuned:
        def save_pretrained(self, path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "chronos.ok").write_text("ok", encoding="utf-8")

    class FakePipeline:
        def fit(self, inputs, **kwargs):
            calls.append({"inputs": inputs, **kwargs})
            return FineTuned()

    loaded: dict = {}

    def loader(*, model_path, device, precision):
        loaded.update(model_path=Path(model_path), device=device, precision=precision)
        return FakePipeline()

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
    preflight = MemoryProbeResult(False, 1, "8 GB below full gate", 6.0)

    result = run_chronos2_training(
        root=tmp_path,
        train_frame=_bars("2026-09-01", 20),
        validation_frame=_bars("2026-09-02", 12),
        context_length=4,
        prediction_length=2,
        steps=2,
        learning_rate=1e-4,
        hardware=hardware,
        preflight=preflight,
        pipeline_loader=loader,
    )

    assert result.status == "COMPLETED"
    assert loaded["device"] == "cuda"
    assert loaded["model_path"].name == "chronos-2"
    assert calls and calls[0]["finetune_mode"] == "lora"
    assert Path(result.checkpoint, "model", "chronos.ok").is_file()
