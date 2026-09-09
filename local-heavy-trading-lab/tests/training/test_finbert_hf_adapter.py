from pathlib import Path

import numpy as np


def test_finbert_adapter_remaps_native_labels_and_logits(tmp_path: Path):
    from heavy_lab.training.integrations.finbert_hf import FinBERTHFModelAdapter

    calls = {"labels": [], "model_saved": None, "tokenizer_saved": None}

    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def to(self, device):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self.value, dtype=float)

        def item(self):
            return float(self.value)

        def backward(self):
            return None

    class FakeTorch:
        long = "long"

        @staticmethod
        def tensor(value, dtype=None, device=None):
            return FakeTensor(value)

        class no_grad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": FakeTensor([[1, 2, 3]])}

        def save_pretrained(self, path):
            calls["tokenizer_saved"] = Path(path)

    class FakeConfig:
        id2label = {0: "positive", 1: "negative", 2: "neutral"}

    class FakeOutput:
        def __init__(self, *, logits=None, loss=None):
            self.logits = logits
            self.loss = loss

    class FakeModel:
        config = FakeConfig()

        def to(self, device):
            return self

        def train(self):
            return None

        def eval(self):
            return None

        def parameters(self):
            return [object()]

        def __call__(self, **kwargs):
            labels = kwargs.get("labels")
            if labels is not None:
                calls["labels"].append(labels.value)
                return FakeOutput(loss=FakeTensor(0.25), logits=FakeTensor([[10.0, 20.0, 30.0]]))
            return FakeOutput(logits=FakeTensor([[10.0, 20.0, 30.0]]))

        def save_pretrained(self, path):
            calls["model_saved"] = Path(path)

    class FakeOptimizer:
        def zero_grad(self):
            return None

        def step(self):
            return None

    def optimizer_factory(parameters, lr):
        assert lr == 1e-5
        return FakeOptimizer()

    adapter = FinBERTHFModelAdapter(
        model_path=tmp_path / "finbert",
        device="cuda",
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        torch_module=FakeTorch,
        optimizer_factory=optimizer_factory,
    )

    loss = adapter.train_step(text="profits rose", label=2, learning_rate=1e-5)
    assert loss == 0.25
    assert calls["labels"] == [[0]]  # canonical positive=2 -> native Prosus positive=0

    logits = adapter.predict_logits([{"text": "example"}])
    np.testing.assert_allclose(logits, [[20.0, 30.0, 10.0]])  # negative, neutral, positive

    adapter.save_pretrained(tmp_path / "model")
    adapter.save_tokenizer(tmp_path / "tokenizer")
    assert calls["model_saved"] == tmp_path / "model"
    assert calls["tokenizer_saved"] == tmp_path / "tokenizer"
