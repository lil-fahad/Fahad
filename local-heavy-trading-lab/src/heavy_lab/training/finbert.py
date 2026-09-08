from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from heavy_lab.training.base import TrainPlan, TrainResult, TrainerAdapter
from heavy_lab.training.calibration import fit_temperature, negative_log_likelihood, softmax


class FinBERTTrainer(TrainerAdapter):
    kind = "finbert"

    def __init__(self, registry, *, model: Any) -> None:
        super().__init__(registry)
        self.model = model
        self._datasets: dict[str, dict[str, Any]] = {}
        self._temperature_by_run: dict[str, float] = {}
        self._latest_run_id: str | None = None

    @staticmethod
    def _validate_examples(examples, *, split: str) -> None:
        required = {"document_id", "timestamp_utc", "text", "label"}
        for item in examples:
            missing = required.difference(item)
            if missing:
                raise ValueError(f"FinBERT {split} example missing fields: {sorted(missing)}")
            label = int(item["label"])
            if label not in {0, 1, 2}:
                raise ValueError("FinBERT labels must be 0, 1, or 2")
            pd.Timestamp(item["timestamp_utc"])

    def prepare(
        self,
        run,
        dataset,
        hardware,
        *,
        output_root: Path,
        epochs: int = 1,
        learning_rate: float = 1e-5,
    ) -> TrainPlan:
        if "train" not in dataset or "validation" not in dataset:
            raise ValueError("FinBERT dataset must contain train and validation splits")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        train = list(dataset["train"])
        validation = list(dataset["validation"])
        if not train or not validation:
            raise ValueError("FinBERT train and validation splits must be non-empty")
        self._validate_examples(train, split="train")
        self._validate_examples(validation, split="validation")

        train_ids = {str(item["document_id"]) for item in train}
        validation_ids = {str(item["document_id"]) for item in validation}
        overlap = train_ids.intersection(validation_ids)
        if overlap:
            raise ValueError(f"FinBERT document overlap across train/validation: {sorted(overlap)}")

        train_times = [pd.Timestamp(item["timestamp_utc"]) for item in train]
        validation_times = [pd.Timestamp(item["timestamp_utc"]) for item in validation]
        if max(train_times) >= min(validation_times):
            raise ValueError("FinBERT validation time must be strictly after training time")

        self._datasets[run.run_id] = {"train": train, "validation": validation}
        return TrainPlan(
            run_id=run.run_id,
            model_kind=self.kind,
            config={
                "hardware": dict(hardware),
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
            },
            checkpoint_dir=Path(output_root) / run.run_id / "checkpoint",
        )

    def train(self, plan: TrainPlan) -> TrainResult:
        dataset = self._datasets.get(plan.run_id)
        if dataset is None:
            raise RuntimeError("FinBERT training dataset is not attached to this run")

        validation = dataset["validation"]
        labels = np.asarray([int(item["label"]) for item in validation], dtype=int)
        baseline_logits = np.asarray(self.model.predict_logits(validation), dtype=float)
        baseline_nll = negative_log_likelihood(baseline_logits, labels)

        last_train_loss: float | None = None
        for _ in range(int(plan.config["epochs"])):
            for item in dataset["train"]:
                last_train_loss = float(
                    self.model.train_step(
                        text=item["text"],
                        label=int(item["label"]),
                        learning_rate=float(plan.config["learning_rate"]),
                    )
                )

        finetuned_logits = np.asarray(self.model.predict_logits(validation), dtype=float)
        finetuned_nll = negative_log_likelihood(finetuned_logits, labels)
        temperature = fit_temperature(finetuned_logits, labels)
        calibrated_nll = negative_log_likelihood(finetuned_logits, labels, temperature=temperature)
        self._temperature_by_run[plan.run_id] = temperature
        self._latest_run_id = plan.run_id

        plan.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(plan.checkpoint_dir / "model")
        self.model.save_tokenizer(plan.checkpoint_dir / "tokenizer")
        (plan.checkpoint_dir / "label_map.json").write_text(
            json.dumps({"0": "negative", "1": "neutral", "2": "positive"}, indent=2),
            encoding="utf-8",
        )
        (plan.checkpoint_dir / "calibration.json").write_text(
            json.dumps({"method": "temperature", "temperature": temperature}, indent=2),
            encoding="utf-8",
        )
        (plan.checkpoint_dir / "checkpoint.ok").write_text("ok", encoding="utf-8")
        self.registry.mark_checkpoint(plan.run_id, plan.checkpoint_dir)
        self.registry.set_status(plan.run_id, "COMPLETED")

        metrics = {
            "baseline_nll": baseline_nll,
            "finetuned_nll": finetuned_nll,
            "calibrated_nll": calibrated_nll,
            "calibration_temperature": temperature,
        }
        if last_train_loss is not None:
            metrics["last_train_loss"] = last_train_loss
        return TrainResult(
            run_id=plan.run_id,
            status="COMPLETED",
            checkpoint=str(plan.checkpoint_dir.resolve()),
            metrics=metrics,
        )

    def calibrated_probabilities(self, examples, *, run_id: str | None = None) -> np.ndarray:
        selected = run_id or self._latest_run_id
        if selected is None or selected not in self._temperature_by_run:
            raise RuntimeError("FinBERT calibration is not fitted yet")
        logits = np.asarray(self.model.predict_logits(examples), dtype=float)
        return softmax(logits, temperature=self._temperature_by_run[selected])

    def resume(self, run_id: str) -> TrainResult:
        run = self.registry.resume(run_id)
        return TrainResult(
            run_id=run.run_id,
            status=run.status,
            checkpoint=run.latest_checkpoint,
            metrics={},
        )
