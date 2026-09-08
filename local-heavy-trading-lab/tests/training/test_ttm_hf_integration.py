from pathlib import Path


def test_ttm_hf_adapter_uses_trainer_for_real_evaluate_and_train(tmp_path: Path):
    from heavy_lab.training.integrations.ttm_hf import TTMHFModelAdapter

    calls = {"train": 0, "evaluate": 0, "saved": None}

    class FakeModel:
        def save_pretrained(self, path):
            calls["saved"] = Path(path)

    class FakeTrainer:
        def __init__(self, *, model, args, train_dataset=None, eval_dataset=None):
            self.model = model
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset

        def evaluate(self, dataset=None):
            calls["evaluate"] += 1
            return {"eval_loss": 0.25}

        def train(self):
            calls["train"] += 1
            return type("Result", (), {"training_loss": 0.125})()

    adapter = TTMHFModelAdapter(
        model_path=tmp_path / "ttm-r2",
        output_root=tmp_path / "trainer",
        model=FakeModel(),
        trainer_factory=FakeTrainer,
        training_args_factory=lambda **kwargs: kwargs,
        batch_size=4,
        precision="fp16",
    )

    assert adapter.evaluate([{"past_values": [1.0], "future_values": [2.0]}]) == 0.25
    assert adapter.train_epoch([{"past_values": [1.0], "future_values": [2.0]}], learning_rate=1e-4) == 0.125
    adapter.save_pretrained(tmp_path / "saved")

    assert calls == {"train": 1, "evaluate": 1, "saved": tmp_path / "saved"}
