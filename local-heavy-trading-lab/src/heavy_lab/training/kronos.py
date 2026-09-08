from __future__ import annotations

from pathlib import Path
from typing import Any

from heavy_lab.training.base import MemoryProbeResult, TrainPlan, TrainResult, TrainerAdapter


def _pair_revision(manifest: dict) -> str | None:
    revision = manifest.get("requested_revision")
    return str(revision) if revision is not None else None


class KronosTrainer(TrainerAdapter):
    kind = "kronos"

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
        model_manifest: dict,
        tokenizer_manifest: dict,
        output_root: Path,
        epochs: int = 1,
        learning_rate: float = 1e-4,
    ) -> TrainPlan:
        if "train" not in dataset or "validation" not in dataset:
            raise ValueError("Kronos dataset must contain train and validation splits")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if preflight.max_micro_batch_size <= 0:
            raise RuntimeError(f"Kronos memory preflight rejected training: {preflight.reason}")

        model_revision = _pair_revision(model_manifest)
        tokenizer_revision = _pair_revision(tokenizer_manifest)
        if model_revision != tokenizer_revision:
            raise ValueError(
                "Kronos model/tokenizer requested revision mismatch: "
                f"model={model_revision!r} tokenizer={tokenizer_revision!r}"
            )

        for split_name in ("train", "validation"):
            for window in dataset[split_name]:
                ohlcv = window.get("ohlcv")
                if not isinstance(ohlcv, list) or not ohlcv:
                    raise ValueError("Kronos windows require non-empty ohlcv rows")
                if any(not isinstance(row, (list, tuple)) or len(row) != 5 for row in ohlcv):
                    raise ValueError("Kronos OHLCV rows must be ordered [open, high, low, close, volume]")

        gradient_checkpointing = not preflight.safe_full_finetune
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
                "gradient_checkpointing": gradient_checkpointing,
                "micro_batch_size": int(preflight.max_micro_batch_size),
                "mode": "native-full" if preflight.safe_full_finetune else "native-low-vram",
                "model_requested_revision": model_revision,
                "tokenizer_requested_revision": tokenizer_revision,
                "model_resolved_revision": model_manifest.get("resolved_revision"),
                "tokenizer_resolved_revision": tokenizer_manifest.get("resolved_revision"),
            },
            checkpoint_dir=Path(output_root) / run.run_id / "checkpoint",
        )

    def train(self, plan: TrainPlan) -> TrainResult:
        dataset = self._datasets.get(plan.run_id)
        if dataset is None:
            raise RuntimeError("Kronos training dataset is not attached to this run")

        baseline_loss = float(self.model.evaluate(dataset["validation"]))
        last_train_loss: float | None = None
        for _ in range(int(plan.config["epochs"])):
            for window in dataset["train"]:
                last_train_loss = float(
                    self.model.train_step(
                        ohlcv=window["ohlcv"],
                        learning_rate=float(plan.config["learning_rate"]),
                        gradient_checkpointing=bool(plan.config["gradient_checkpointing"]),
                    )
                )

        finetuned_loss = float(self.model.evaluate(dataset["validation"]))
        plan.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(plan.checkpoint_dir / "model")
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
