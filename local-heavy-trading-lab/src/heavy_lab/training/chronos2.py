from __future__ import annotations

from pathlib import Path
from typing import Any

from heavy_lab.training.base import MemoryProbeResult, TrainPlan, TrainResult, TrainerAdapter


class Chronos2Trainer(TrainerAdapter):
    kind = "chronos2"

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
            raise ValueError("Chronos-2 dataset must contain train and validation splits")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if preflight.max_micro_batch_size <= 0:
            raise RuntimeError(f"Chronos-2 memory preflight rejected training: {preflight.reason}")

        self._datasets[run.run_id] = {
            "train": list(dataset["train"]),
            "validation": list(dataset["validation"]),
        }
        return TrainPlan(
            run_id=run.run_id,
            model_kind=self.kind,
            config={
                "hardware": dict(hardware),
                "num_steps": int(epochs),
                "learning_rate": float(learning_rate),
                "micro_batch_size": int(preflight.max_micro_batch_size),
                "preflight_reason": preflight.reason,
                "finetune_mode": "full" if preflight.safe_full_finetune else "lora",
            },
            checkpoint_dir=Path(output_root) / run.run_id / "checkpoint",
        )

    @staticmethod
    def _prepare_inputs(windows: list[dict[str, Any]]) -> tuple[list[list[float]], int]:
        if not windows:
            raise ValueError("Chronos-2 split is empty")
        prediction_lengths = {len(window.get("target", [])) for window in windows}
        if 0 in prediction_lengths or len(prediction_lengths) != 1:
            raise ValueError("Chronos-2 targets must be non-empty and share one prediction length")
        prediction_length = prediction_lengths.pop()
        inputs: list[list[float]] = []
        for window in windows:
            if "context" not in window or "target" not in window:
                raise ValueError("Chronos-2 windows require context and target")
            context = [float(value) for value in window["context"]]
            target = [float(value) for value in window["target"]]
            if not context:
                raise ValueError("Chronos-2 context must be non-empty")
            inputs.append(context + target)
        return inputs, prediction_length

    def train(self, plan: TrainPlan) -> TrainResult:
        dataset = self._datasets.get(plan.run_id)
        if dataset is None:
            raise RuntimeError("Chronos-2 training dataset is not attached to this run")

        train_inputs, prediction_length = self._prepare_inputs(dataset["train"])
        validation_inputs, validation_prediction_length = self._prepare_inputs(dataset["validation"])
        if validation_prediction_length != prediction_length:
            raise ValueError("Chronos-2 train and validation prediction lengths must match")

        plan.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer_output = plan.checkpoint_dir / "trainer"
        finetuned = self.model.fit(
            train_inputs,
            prediction_length=prediction_length,
            validation_inputs=validation_inputs,
            finetune_mode=str(plan.config["finetune_mode"]),
            learning_rate=float(plan.config["learning_rate"]),
            num_steps=int(plan.config["num_steps"]),
            batch_size=int(plan.config["micro_batch_size"]),
            output_dir=trainer_output,
            finetuned_ckpt_name="native-fit",
        )
        model_dir = plan.checkpoint_dir / "model"
        finetuned.save_pretrained(model_dir)
        (plan.checkpoint_dir / "checkpoint.ok").write_text("ok", encoding="utf-8")
        self.registry.mark_checkpoint(plan.run_id, plan.checkpoint_dir)
        self.registry.set_status(plan.run_id, "COMPLETED")

        return TrainResult(
            run_id=plan.run_id,
            status="COMPLETED",
            checkpoint=str(plan.checkpoint_dir.resolve()),
            metrics={},
        )

    def resume(self, run_id: str) -> TrainResult:
        run = self.registry.resume(run_id)
        return TrainResult(
            run_id=run.run_id,
            status=run.status,
            checkpoint=run.latest_checkpoint,
            metrics={},
        )
