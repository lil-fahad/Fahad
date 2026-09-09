from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np


CANONICAL_LABELS = ("negative", "neutral", "positive")


class FinBERTHFModelAdapter:
    """Local-only Hugging Face adapter for ProsusAI/FinBERT.

    The lab uses canonical sentiment IDs negative=0, neutral=1, positive=2.
    ProsusAI/FinBERT publishes a different native ID order, so this adapter
    derives the mapping from model.config.id2label and translates both labels
    and logits at the integration boundary.
    """

    def __init__(
        self,
        *,
        model_path: Path,
        device: str,
        tokenizer: Any | None = None,
        model: Any | None = None,
        torch_module: Any | None = None,
        optimizer_factory: Callable[[Any, float], Any] | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists() and (tokenizer is None or model is None):
            raise FileNotFoundError(f"FinBERT local snapshot not found: {self.model_path}")

        if torch_module is None:
            try:
                import torch as torch_module  # type: ignore[no-redef]
            except ImportError as exc:  # pragma: no cover - exercised on local training host
                raise RuntimeError("FinBERT training requires PyTorch; install the finbert-train extra") from exc
        self.torch = torch_module
        self.device = str(device)
        self.max_length = int(max_length)
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

        if tokenizer is None or model is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - exercised on local training host
                raise RuntimeError("FinBERT training requires Transformers; install the finbert-train extra") from exc
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path), local_files_only=True)

        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self._canonical_to_native = self._derive_label_mapping(self.model.config.id2label)
        self._native_columns_for_canonical = [self._canonical_to_native[index] for index in range(3)]
        self._optimizer_factory = optimizer_factory or (
            lambda parameters, lr: self.torch.optim.AdamW(parameters, lr=float(lr))
        )
        self._optimizer: Any | None = None
        self._optimizer_lr: float | None = None

    @staticmethod
    def _derive_label_mapping(id2label: Any) -> dict[int, int]:
        native_by_name: dict[str, int] = {}
        for raw_index, raw_name in dict(id2label).items():
            native_by_name[str(raw_name).strip().lower()] = int(raw_index)
        missing = [name for name in CANONICAL_LABELS if name not in native_by_name]
        if missing:
            raise ValueError(f"FinBERT config is missing sentiment labels: {missing}")
        return {canonical_id: native_by_name[name] for canonical_id, name in enumerate(CANONICAL_LABELS)}

    def _tokenize(self, texts: str | list[str]) -> dict[str, Any]:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {name: value.to(self.device) for name, value in dict(batch).items()}

    def _optimizer_for(self, learning_rate: float):
        learning_rate = float(learning_rate)
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self._optimizer is None or self._optimizer_lr != learning_rate:
            self._optimizer = self._optimizer_factory(self.model.parameters(), learning_rate)
            self._optimizer_lr = learning_rate
        return self._optimizer

    def train_step(self, *, text: str, label: int, learning_rate: float) -> float:
        canonical_label = int(label)
        if canonical_label not in self._canonical_to_native:
            raise ValueError("FinBERT canonical label must be 0, 1, or 2")
        native_label = self._canonical_to_native[canonical_label]

        self.model.train()
        optimizer = self._optimizer_for(learning_rate)
        optimizer.zero_grad()
        batch = self._tokenize(str(text))
        labels = self.torch.tensor([native_label], dtype=self.torch.long, device=self.device)
        output = self.model(**batch, labels=labels)
        loss = output.loss
        loss.backward()
        optimizer.step()
        return float(loss.item())

    def predict_logits(self, examples) -> np.ndarray:
        texts = [str(item["text"]) for item in examples]
        if not texts:
            return np.empty((0, 3), dtype=float)
        self.model.eval()
        batch = self._tokenize(texts)
        with self.torch.no_grad():
            output = self.model(**batch)
        native_logits = np.asarray(output.logits.detach().cpu().numpy(), dtype=float)
        if native_logits.ndim != 2 or native_logits.shape[1] != 3:
            raise ValueError(f"FinBERT expected 3 logits per example, received shape {native_logits.shape}")
        return native_logits[:, self._native_columns_for_canonical]

    def save_pretrained(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)

    def save_tokenizer(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(path)
