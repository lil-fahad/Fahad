from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


class TTMHFModelAdapter:
    """Real IBM TTM adapter backed by Hugging Face Trainer.

    Heavy dependencies are imported lazily so the core lab and CI remain light.
    When ``model``/factories are injected, the adapter is unit-testable without
    installing PyTorch, Transformers, or granite-tsfm.
    """

    def __init__(
        self,
        *,
        model_path: Path,
        output_root: Path,
        model: Any | None = None,
        trainer_factory: Callable[..., Any] | None = None,
        training_args_factory: Callable[..., Any] | None = None,
        batch_size: int = 4,
        precision: str = "fp16",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")

        self.model_path = Path(model_path)
        self.output_root = Path(output_root)
        self.batch_size = int(batch_size)
        self.precision = precision

        if model is None or trainer_factory is None or training_args_factory is None:
            loaded_model, loaded_trainer, loaded_args = self._load_upstream(model)
            model = loaded_model
            trainer_factory = trainer_factory or loaded_trainer
            training_args_factory = training_args_factory or loaded_args

        self.model = model
        self.trainer_factory = trainer_factory
        self.training_args_factory = training_args_factory

    def _load_upstream(self, existing_model: Any | None):
        try:
            from transformers import Trainer, TrainingArguments
            from tsfm_public.models.tinytimemixer.modeling_tinytimemixer import (
                TinyTimeMixerForPrediction,
            )
        except Exception as exc:  # pragma: no cover - exercised on local heavy install
            raise RuntimeError(
                "TTM training dependencies are missing. Install the local extra "
                "with: pip install -e '.[ttm-train]'"
            ) from exc

        model = existing_model
        if model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Local TTM snapshot not found: {self.model_path}")
            model = TinyTimeMixerForPrediction.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
        return model, Trainer, TrainingArguments

    def _args(self, *, mode: str, learning_rate: float | None = None) -> Any:
        self.output_root.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {
            "output_dir": str(self.output_root),
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "report_to": [],
            "remove_unused_columns": True,
            "save_strategy": "no",
            "logging_strategy": "no",
            "fp16": self.precision == "fp16",
            "bf16": self.precision == "bf16",
        }
        if mode == "train":
            kwargs.update(num_train_epochs=1.0, learning_rate=float(learning_rate or 1e-4))
        return self.training_args_factory(**kwargs)

    def evaluate(self, dataset: Any) -> float:
        trainer = self.trainer_factory(
            model=self.model,
            args=self._args(mode="eval"),
            eval_dataset=dataset,
        )
        metrics = trainer.evaluate(dataset)
        if "eval_loss" not in metrics:
            raise RuntimeError("TTM Trainer.evaluate did not return eval_loss")
        return float(metrics["eval_loss"])

    def train_epoch(self, dataset: Any, *, learning_rate: float) -> float:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        trainer = self.trainer_factory(
            model=self.model,
            args=self._args(mode="train", learning_rate=learning_rate),
            train_dataset=dataset,
        )
        result = trainer.train()
        loss = getattr(result, "training_loss", None)
        if loss is None:
            raise RuntimeError("TTM Trainer.train did not return training_loss")
        return float(loss)

    def save_pretrained(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
