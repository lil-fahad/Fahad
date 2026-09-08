from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import platform
import shutil

import psutil


@dataclass(frozen=True)
class HardwareProfile:
    os_name: str
    cpu_count: int
    device: str
    cuda: bool
    mps: bool
    ram_gb: float
    vram_gb: float | None
    disk_free_gb: float
    gpu_name: str | None = None
    cuda_version: str | None = None
    compute_capability: str | None = None
    bf16: bool = False
    fp16: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _gb(value: int | float) -> float:
    return round(float(value) / (1024 ** 3), 3)


def _disk_free_gb(path: Path) -> float:
    probe = Path(path).expanduser().resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return _gb(shutil.disk_usage(probe).free)


def _torch_probe() -> dict:
    result = {
        "cuda": False,
        "mps": False,
        "device": "cpu",
        "vram_gb": None,
        "gpu_name": None,
        "cuda_version": None,
        "compute_capability": None,
        "bf16": False,
        "fp16": False,
    }
    try:
        import torch
    except Exception:
        return result

    try:
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            major, minor = torch.cuda.get_device_capability(idx)
            result.update(
                cuda=True,
                device="cuda",
                vram_gb=_gb(props.total_memory),
                gpu_name=torch.cuda.get_device_name(idx),
                cuda_version=getattr(torch.version, "cuda", None),
                compute_capability=f"{major}.{minor}",
                bf16=bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
                fp16=True,
            )
            return result
    except Exception:
        pass

    try:
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            result.update(mps=True, device="mps", fp16=True)
    except Exception:
        pass
    return result


def detect_hardware(storage_root: Path | str = Path.cwd()) -> HardwareProfile:
    forced = os.getenv("HEAVY_LAB_FORCE_DEVICE", "").strip().lower()
    torch_info = _torch_probe()
    if forced == "cpu":
        torch_info.update(
            cuda=False,
            mps=False,
            device="cpu",
            vram_gb=None,
            gpu_name=None,
            cuda_version=None,
            compute_capability=None,
            bf16=False,
            fp16=False,
        )
    elif forced in {"cuda", "mps"} and torch_info[forced]:
        torch_info["device"] = forced

    return HardwareProfile(
        os_name=platform.system() or "Unknown",
        cpu_count=os.cpu_count() or 1,
        device=str(torch_info["device"]),
        cuda=bool(torch_info["cuda"]),
        mps=bool(torch_info["mps"]),
        ram_gb=_gb(psutil.virtual_memory().total),
        vram_gb=torch_info["vram_gb"],
        disk_free_gb=_disk_free_gb(Path(storage_root)),
        gpu_name=torch_info["gpu_name"],
        cuda_version=torch_info["cuda_version"],
        compute_capability=torch_info["compute_capability"],
        bf16=bool(torch_info["bf16"]),
        fp16=bool(torch_info["fp16"]),
    )


def select_training_profile(profile: HardwareProfile) -> str:
    if profile.cuda:
        vram = float(profile.vram_gb or 0.0)
        if vram >= 24.0:
            return "full-heavy"
        if vram >= 12.0:
            return "lora-heavy"
        return "low-vram"
    if profile.mps:
        return "lora-heavy" if profile.ram_gb >= 24.0 else "low-vram"
    return "cpu-control"
