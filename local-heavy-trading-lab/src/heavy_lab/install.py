from __future__ import annotations

import platform
import sys


PYTORCH_INDEX_ROOT = "https://download.pytorch.org/whl"


def _cuda_index(cuda_version: str | None) -> str:
    if not cuda_version:
        raise ValueError("CUDA device detected but CUDA runtime version is unknown")
    try:
        major_s, minor_s, *_ = cuda_version.split(".") + ["0"]
        version = (int(major_s), int(minor_s))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported CUDA version format: {cuda_version!r}") from exc

    if version >= (12, 8):
        return "cu128"
    if version >= (12, 6):
        return "cu126"
    if version >= (11, 8):
        return "cu118"
    raise ValueError(
        f"CUDA {cuda_version} is below the supported local installer floor (11.8). "
        "Update the NVIDIA driver/CUDA-compatible environment before heavy training."
    )


def resolve_torch_install(
    *,
    device: str,
    cuda_version: str | None,
    os_name: str | None = None,
) -> list[str]:
    """Return an explicit pip command for the detected platform.

    The resolver intentionally does not install bitsandbytes or guess a CUDA
    wheel. macOS/MPS uses the normal PyPI wheel; Linux/Windows use the official
    PyTorch compute-platform index selected from the detected CUDA runtime.
    """

    os_name = os_name or platform.system()
    base = [sys.executable, "-m", "pip", "install", "torch", "torchvision"]

    if os_name == "Darwin" or device == "mps":
        return base
    if device == "cpu":
        return base + ["--index-url", f"{PYTORCH_INDEX_ROOT}/cpu"]
    if device == "cuda":
        return base + ["--index-url", f"{PYTORCH_INDEX_ROOT}/{_cuda_index(cuda_version)}"]
    raise ValueError(f"Unsupported training device: {device!r}")
