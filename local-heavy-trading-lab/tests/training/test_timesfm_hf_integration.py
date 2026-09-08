from pathlib import Path


def test_timesfm_hf_adapter_runs_lora_step_eval_and_save(tmp_path: Path):
    from heavy_lab.training.integrations.timesfm_hf import TimesFMHFModelAdapter

    calls = {"backward": 0, "step": 0, "zero": 0, "saved": None, "lora": 0}

    class FakeLoss:
        def __init__(self, value):
            self.value = float(value)

        def backward(self):
            calls["backward"] += 1

        def item(self):
            return self.value

    class FakeOutput:
        def __init__(self, value):
            self.loss = FakeLoss(value)

    class FakeModel:
        def __init__(self):
            self.training = False

        def train(self):
            self.training = True
            return self

        def eval(self):
            self.training = False
            return self

        def __call__(self, *, past_values, future_values, forecast_context_len):
            assert forecast_context_len == len(past_values[0])
            assert len(future_values[0]) == 2
            return FakeOutput(0.25 if self.training else 0.5)

        def parameters(self):
            return []

        def save_pretrained(self, path):
            calls["saved"] = Path(path)

    class FakeOptimizer:
        def step(self):
            calls["step"] += 1

        def zero_grad(self):
            calls["zero"] += 1

    def fake_lora_factory(model, **kwargs):
        calls["lora"] += 1
        return model

    adapter = TimesFMHFModelAdapter(
        model_path=tmp_path / "timesfm-2.5",
        device="cuda",
        precision="fp16",
        model=FakeModel(),
        tensor_factory=lambda values, **kwargs: [list(values)],
        optimizer_factory=lambda params, lr: FakeOptimizer(),
        lora_factory=fake_lora_factory,
        clip_grad_norm=lambda params, max_norm: None,
        no_grad_context=lambda: _NullContext(),
    )

    train_loss = adapter.train_step(
        past_values=[100.0, 101.0, 102.0, 103.0],
        future_values=[104.0, 105.0],
        learning_rate=1e-4,
        use_lora=True,
    )
    eval_loss = adapter.evaluate(
        [{"past_values": [100.0, 101.0, 102.0, 103.0], "future_values": [104.0, 105.0]}]
    )
    adapter.save_adapter(tmp_path / "adapter")

    assert train_loss == 0.25
    assert eval_loss == 0.5
    assert calls["lora"] == 1
    assert calls["backward"] == 1
    assert calls["step"] == 1
    assert calls["zero"] == 1
    assert calls["saved"] == tmp_path / "adapter"


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False
