from pathlib import Path

import numpy as np

from heavy_lab.hardware import HardwareProfile


def test_finbert_launcher_uses_local_snapshot_and_real_trainer_contract(tmp_path: Path):
    from heavy_lab.training.launch import run_finbert_training

    loaded: dict = {}
    updates: list[tuple[str, int]] = []

    class FakeAdapter:
        def __init__(self, *, model_path, device):
            loaded.update(model_path=Path(model_path), device=device)

        def train_step(self, *, text, label, learning_rate):
            updates.append((str(text), int(label)))
            return 0.2

        def predict_logits(self, examples):
            rows = []
            for item in examples:
                label = int(item["label"])
                row = [-1.0, -1.0, -1.0]
                row[label] = 3.0 if updates else 1.0
                rows.append(row)
            return np.asarray(rows, dtype=float)

        def save_pretrained(self, path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "model.ok").write_text("ok", encoding="utf-8")

        def save_tokenizer(self, path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "tokenizer.ok").write_text("ok", encoding="utf-8")

    train = [
        {"document_id": "a", "timestamp_utc": "2026-01-01T10:00:00Z", "text": "profits rose", "label": 2},
        {"document_id": "b", "timestamp_utc": "2026-01-01T11:00:00Z", "text": "loss widened", "label": 0},
    ]
    validation = [
        {"document_id": "c", "timestamp_utc": "2026-01-02T10:00:00Z", "text": "outlook stable", "label": 1},
        {"document_id": "d", "timestamp_utc": "2026-01-02T11:00:00Z", "text": "guidance raised", "label": 2},
    ]
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

    result = run_finbert_training(
        root=tmp_path,
        train_examples=train,
        validation_examples=validation,
        epochs=1,
        learning_rate=1e-5,
        hardware=hardware,
        adapter_factory=FakeAdapter,
    )

    assert result.status == "COMPLETED"
    assert loaded["model_path"].name == "finbert"
    assert loaded["device"] == "cuda"
    assert updates == [("profits rose", 2), ("loss widened", 0)]
    assert Path(result.checkpoint, "model", "model.ok").is_file()
    assert Path(result.checkpoint, "tokenizer", "tokenizer.ok").is_file()
    assert Path(result.checkpoint, "calibration.json").is_file()
