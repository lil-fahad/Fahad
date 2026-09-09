from __future__ import annotations

from pathlib import Path
from typing import Any


def load_chronos2_pipeline(
    *,
    model_path: Path,
    device: str,
    precision: str,
    pipeline_cls: Any | None = None,
    torch_module: Any | None = None,
):
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"Chronos-2 local snapshot not found: {path}. Run `lab download-models --model chronos2` first."
        )

    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for Chronos-2 training") from exc

    if pipeline_cls is None:
        try:
            from chronos import BaseChronosPipeline as pipeline_cls
        except ImportError as exc:
            raise RuntimeError(
                "Chronos-2 training dependencies are missing. Install the `chronos-train` extra."
            ) from exc

    precision_key = str(precision).strip().lower()
    dtype_map = {
        "fp16": torch_module.float16,
        "float16": torch_module.float16,
        "bf16": torch_module.bfloat16,
        "bfloat16": torch_module.bfloat16,
        "fp32": torch_module.float32,
        "float32": torch_module.float32,
    }
    if precision_key not in dtype_map:
        raise ValueError(f"Unsupported Chronos-2 precision: {precision}")

    return pipeline_cls.from_pretrained(
        path,
        device_map=str(device),
        torch_dtype=dtype_map[precision_key],
    )
