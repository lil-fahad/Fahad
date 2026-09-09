import json
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


def _hardware():
    from heavy_lab.hardware import HardwareProfile

    return HardwareProfile(
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


def test_training_readiness_accepts_complete_3070ti_setup(tmp_path: Path):
    from heavy_lab.models.catalog import MODEL_CATALOG
    from heavy_lab.training.readiness import training_readiness

    for spec in MODEL_CATALOG.values():
        target = tmp_path / "models" / "base" / spec.local_name
        target.mkdir(parents=True)
        (target / "manifest.json").write_text(
            json.dumps({"name": spec.name, "resolved_revision": "fixture"}),
            encoding="utf-8",
        )

    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"
    _bars("2026-08-01T13:30:00Z", 80).to_parquet(train_path, index=False)
    _bars("2026-08-02T13:30:00Z", 40).to_parquet(validation_path, index=False)

    finbert_train = tmp_path / "finbert-train.jsonl"
    finbert_validation = tmp_path / "finbert-validation.jsonl"
    finbert_train.write_text('{"text":"earnings improved","label":2}\n', encoding="utf-8")
    finbert_validation.write_text('{"text":"guidance unchanged","label":1}\n', encoding="utf-8")

    report = training_readiness(
        root=tmp_path,
        train_parquet=train_path,
        validation_parquet=validation_path,
        finbert_train_jsonl=finbert_train,
        finbert_validation_jsonl=finbert_validation,
        hardware=_hardware(),
    )

    assert report["ready"] is True
    assert report["hardware"]["gpu_name"] == "NVIDIA GeForce RTX 3070 Ti"
    assert report["plans"]["timesfm25"]["mode"] == "lora"
    assert report["plans"]["chronos2"]["mode"] == "native-low-vram"
    assert report["plans"]["kronos"]["micro_batch_size"] == 1
    assert all(report["models"].values())


def test_training_readiness_reports_missing_inputs_without_starting(tmp_path: Path):
    from heavy_lab.training.readiness import training_readiness

    report = training_readiness(root=tmp_path, hardware=_hardware())
    assert report["ready"] is False
    assert report["missing"]
    assert any("model:" in item for item in report["missing"])
    assert "train_parquet" in report["missing"]
    assert "finbert_train_jsonl" in report["missing"]
