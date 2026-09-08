from __future__ import annotations

from pathlib import Path
from typing import Any

from heavy_lab.training.base import MemoryProbeResult, TrainPlan, TrainResult, TrainerAdapter


class TimesFM25Trainer(TrainerAdapter):
    kind = "timesfm25"

    def __init__(self, registry, *, model: Any) -> None:
        super().__init__(registry)
        self.model = model
        self._datasets: dict[str, dict[str, Any]] = {}

    def prepare(
        self,
        run,
        dataset,
        hardware,
        *,
        preflight: MemoryProbeResult,
        output_root: Path,
        epochs: int = 1,
        learning_rate: float = 1e-4,
    ) -> TrainPlan:
        if "train" not in dataset or "validation" not in dataset:
            raise ValueError("TimesFM dataset must contain train and validation splits")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if preflight.max_micro_batch_size <= 0:
            raise RuntimeError(f"TimesFM memory preflight rejected training: {preflight.reason}")

        use_lora = not preflight.safe_full_finetune
        self._datasets[run.run_id] = {
            "train": dataset["train"],
            "validation": dataset["validation"],
        }
        return TrainPlan(
            run_id=run.run_id,
            model_kind=self.kind,
            config={
                "hardware": dict(hardware),
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
                "use_lora": use_lora,
                "micro_batch_size": int(preflight.max_micro_batch_size),
                "preflight_reason": preflight.reason,
            },
            checkpoint_dir=Path(output_root) / run.run_id / "checkpoint",
        )

    def train(self, plan: TrainPlan) -> TrainResult:
        dataset = self._datasets.get(plan.run_id)
        if dataset is None:
            raise RuntimeError("TimesFM training dataset is not attached to this run")

        validation = dataset["validation"]
        baseline_loss = float(self.model.evaluate(validation))
        last_train_loss: float | None = None
        for _ in range(int(plan.config["epochs"])):
            for example in dataset["train"]:
                if "past_values" not in example or "future_values" not in example:
                    raise ValueError("TimesFM examples require past_values and future_values")
                last_train_loss = float(
                    self.model.train_step(
                        past_values=example["past_values"],
                        future_values=example["future_values"],
                        learning_rate=float(plan.config["learning_rate"]),
                        use_lora=bool(plan.config["use_lora"]),
                    )
                )

        finetuned_loss = float(self.model.evaluate(validation))
        plan.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if plan.config["use_lora"]:
            self.model.save_adapter(plan.checkpoint_dir / "adapter")
        else:
            save_full = getattr(self.model, "save_pretrained", None)
            if save_full is None:
                raise RuntimeError("Full TimesFM fine-tune requires save_pretrained support")
            save_full(plan.checkpoint_dir / "model")
        (plan.checkpoint_dir / "checkpoint.ok").write_text("ok", encoding="utf-8")
        self.registry.mark_checkpoint(plan.run_id, plan.checkpoint_dir)
        self.registry.set_status(plan.run_id, "COMPLETED")

        metrics = {
            "baseline_loss": baseline_loss,
            "finetuned_loss": finetuned_loss,
        }
        if last_train_loss is not None:
            metrics["last_train_loss"] = last_train_loss
        return TrainResult(
            run_id=plan.run_id,
            status="COMPLETED",
            checkpoint=str(plan.checkpoint_dir.resolve()),
            metrics=metrics,
        )

    def resume(self, run_id: str) -> TrainResult:
        run = self.registry.resume(run_id)
        return TrainResult(
            run_id=run.run_id,
            status=run.status,
            checkpoint=run.latest_checkpoint,
            metrics={},
        )
