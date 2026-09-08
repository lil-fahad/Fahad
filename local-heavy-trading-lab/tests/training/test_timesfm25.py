from pathlib import Path


def test_timesfm_uses_raw_past_future_and_forces_lora_when_full_is_unsafe(tmp_path: Path):
    from heavy_lab.runs import RunRegistry
    from heavy_lab.training.base import MemoryProbeResult
    from heavy_lab.training.timesfm25 import TimesFM25Trainer

    calls: list[dict] = []

    class TinyTimesFM:
        def train_step(self, *, past_values, future_values, learning_rate, use_lora):
            calls.append(
                {
                    "past_values": list(past_values),
                    "future_values": list(future_values),
                    "learning_rate": learning_rate,
                    "use_lora": use_lora,
                }
            )
            return 0.25

        def evaluate(self, examples):
            return 0.5 if not calls else 0.2

        def save_adapter(self, path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter.txt").write_text("lora", encoding="utf-8")

    registry = RunRegistry(tmp_path / "runs")
    run = registry.start("timesfm25", {"profile": "smoke"})
    trainer = TimesFM25Trainer(registry, model=TinyTimesFM())
    examples = [
        {"past_values": [100.0, 101.0, 103.0], "future_values": [104.0, 102.0]},
        {"past_values": [200.0, 201.0, 199.0], "future_values": [198.0, 202.0]},
    ]
    preflight = MemoryProbeResult(
        safe_full_finetune=False,
        max_micro_batch_size=1,
        reason="8 GB VRAM below 24 GB full gate",
        peak_memory_gb=5.0,
    )
    plan = trainer.prepare(
        run,
        dataset={"train": examples, "validation": examples[:1]},
        hardware={"device": "cuda", "vram_gb": 8.0},
        preflight=preflight,
        output_root=tmp_path / "trained",
        epochs=1,
        learning_rate=1e-4,
    )
    result = trainer.train(plan)

    assert calls
    assert calls[0]["past_values"] == [100.0, 101.0, 103.0]
    assert calls[0]["future_values"] == [104.0, 102.0]
    assert calls[0]["use_lora"] is True
    assert result.metrics["finetuned_loss"] < result.metrics["baseline_loss"]
    assert Path(result.checkpoint, "adapter", "adapter.txt").is_file()
    assert Path(result.checkpoint, "checkpoint.ok").is_file()
