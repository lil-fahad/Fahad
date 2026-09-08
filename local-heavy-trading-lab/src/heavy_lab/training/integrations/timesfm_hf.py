from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


class TimesFMHFModelAdapter:
    """TimesFM 2.5 training adapter using Transformers + PEFT LoRA.

    Heavy imports are lazy so the core package and normal CI do not require
    PyTorch/Transformers/PEFT. TimesFM's internal RevIN remains responsible for
    normalization; values are passed in their original scale.
    """

    def __init__(
        self,
        *,
        model_path: Path,
        device: str,
        precision: str,
        model: Any | None = None,
        tensor_factory: Callable[..., Any] | None = None,
        optimizer_factory: Callable[[Any, float], Any] | None = None,
        lora_factory: Callable[..., Any] | None = None,
        clip_grad_norm: Callable[[Any, float], Any] | None = None,
        no_grad_context: Callable[[], Any] | None = None,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
    ) -> None:
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if lora_r <= 0 or lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")

        self.model_path = Path(model_path)
        self.device = str(device)
        self.precision = precision
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self._lora_applied = False
        self._optimizer = None
        self._optimizer_lr: float | None = None

        if any(
            item is None
            for item in (model, tensor_factory, optimizer_factory, lora_factory, clip_grad_norm, no_grad_context)
        ):
            upstream = self._load_upstream(model)
            model = model or upstream["model"]
            tensor_factory = tensor_factory or upstream["tensor_factory"]
            optimizer_factory = optimizer_factory or upstream["optimizer_factory"]
            lora_factory = lora_factory or upstream["lora_factory"]
            clip_grad_norm = clip_grad_norm or upstream["clip_grad_norm"]
            no_grad_context = no_grad_context or upstream["no_grad_context"]

        self.model = model
        self.tensor_factory = tensor_factory
        self.optimizer_factory = optimizer_factory
        self.lora_factory = lora_factory
        self.clip_grad_norm = clip_grad_norm
        self.no_grad_context = no_grad_context

    def _load_upstream(self, existing_model: Any | None) -> dict[str, Any]:
        try:
            import torch
            from peft import LoraConfig, get_peft_model
            from transformers import TimesFm2_5ModelForPrediction
        except Exception as exc:  # pragma: no cover - local heavy environment
            raise RuntimeError(
                "TimesFM training dependencies are missing. Install with: "
                "pip install -e '.[timesfm-train]'"
            ) from exc

        dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]

        model = existing_model
        if model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Local TimesFM snapshot not found: {self.model_path}")
            model = TimesFm2_5ModelForPrediction.from_pretrained(
                str(self.model_path),
                torch_dtype=dtype,
                local_files_only=True,
            )
            to_method = getattr(model, "to", None)
            if callable(to_method):
                model = to_method(self.device)

        def tensor_factory(values, **_kwargs):
            return torch.tensor(values, dtype=torch.float32, device=self.device).unsqueeze(0)

        def optimizer_factory(params, lr: float):
            return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

        def lora_factory(base_model, **kwargs):
            config = LoraConfig(
                r=int(kwargs["r"]),
                lora_alpha=int(kwargs["lora_alpha"]),
                target_modules="all-linear",
                lora_dropout=float(kwargs["lora_dropout"]),
                bias="none",
            )
            return get_peft_model(base_model, config)

        return {
            "model": model,
            "tensor_factory": tensor_factory,
            "optimizer_factory": optimizer_factory,
            "lora_factory": lora_factory,
            "clip_grad_norm": torch.nn.utils.clip_grad_norm_,
            "no_grad_context": torch.no_grad,
        }

    def _ensure_lora(self) -> None:
        if self._lora_applied:
            return
        self.model = self.lora_factory(
            self.model,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules="all-linear",
            bias="none",
        )
        self._lora_applied = True
        self._optimizer = None
        self._optimizer_lr = None

    def _ensure_optimizer(self, learning_rate: float) -> Any:
        if self._optimizer is None:
            self._optimizer = self.optimizer_factory(self.model.parameters(), learning_rate)
            self._optimizer_lr = float(learning_rate)
        elif self._optimizer_lr != float(learning_rate):
            for group in getattr(self._optimizer, "param_groups", []):
                group["lr"] = float(learning_rate)
            self._optimizer_lr = float(learning_rate)
        return self._optimizer

    def _tensor(self, values: Any) -> Any:
        return self.tensor_factory(list(values), device=self.device, dtype="float32")

    def train_step(
        self,
        *,
        past_values: Any,
        future_values: Any,
        learning_rate: float,
        use_lora: bool,
    ) -> float:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if use_lora:
            self._ensure_lora()
        model_train = getattr(self.model, "train", None)
        if callable(model_train):
            model_train()

        past = self._tensor(past_values)
        future = self._tensor(future_values)
        optimizer = self._ensure_optimizer(learning_rate)
        output = self.model(
            past_values=past,
            future_values=future,
            forecast_context_len=len(past_values),
        )
        loss = getattr(output, "loss", None)
        if loss is None:
            raise RuntimeError("TimesFM forward pass did not return loss")
        loss.backward()
        self.clip_grad_norm(self.model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        return float(loss.item())

    def evaluate(self, examples: Any) -> float:
        if not examples:
            raise ValueError("TimesFM evaluation examples are empty")
        model_eval = getattr(self.model, "eval", None)
        if callable(model_eval):
            model_eval()
        losses: list[float] = []
        with self.no_grad_context():
            for example in examples:
                past_values = example["past_values"]
                future_values = example["future_values"]
                output = self.model(
                    past_values=self._tensor(past_values),
                    future_values=self._tensor(future_values),
                    forecast_context_len=len(past_values),
                )
                loss = getattr(output, "loss", None)
                if loss is None:
                    raise RuntimeError("TimesFM evaluation forward pass did not return loss")
                losses.append(float(loss.item()))
        return sum(losses) / len(losses)

    def save_adapter(self, path: Path) -> None:
        if not self._lora_applied:
            raise RuntimeError("LoRA adapter has not been initialized")
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)

    def save_pretrained(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
