from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

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


def _blank_probe() -> dict:
    return {
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


def _torch_probe() -> dict:
    result = _blank_probe()
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


def _nvidia_smi_probe() -> dict | None:
    """Probe an NVIDIA GPU without importing PyTorch.

    `nvidia-smi` ships with the NVIDIA driver and is therefore usable during a
    fresh install before a CUDA-enabled PyTorch wheel exists.
    """
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        header = subprocess.run(
            [executable],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        ).stdout
        cuda_match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", header)
        cuda_version = cuda_match.group(1) if cuda_match else None

        query = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        ).stdout.strip().splitlines()[0]
        parts = [part.strip() for part in query.split(",")]
        gpu_name = parts[0] if parts else None
        memory_mib = float(parts[1]) if len(parts) > 1 else 0.0
        compute_capability = parts[2] if len(parts) > 2 and parts[2] else None
        cap_major = 0
        try:
            cap_major = int(float(compute_capability or "0"))
        except ValueError:
            pass
        return {
            "cuda": True,
            "mps": False,
            "device": "cuda",
            "vram_gb": round(memory_mib / 1024.0, 3),
            "gpu_name": gpu_name,
            "cuda_version": cuda_version,
            "compute_capability": compute_capability,
            "bf16": cap_major >= 8,
            "fp16": True,
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _pre_torch_apple_probe() -> dict | None:
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        result = _blank_probe()
        result.update(mps=True, device="mps", fp16=True, gpu_name="Apple Silicon")
        return result
    return None


def detect_hardware(storage_root: Path | str = Path.cwd()) -> HardwareProfile:
    forced = os.getenv("HEAVY_LAB_FORCE_DEVICE", "").strip().lower()
    torch_info = _torch_probe()

    if forced != "cpu" and torch_info["device"] == "cpu":
        nvidia_info = _nvidia_smi_probe()
        if nvidia_info is not None:
            torch_info = nvidia_info
        else:
            apple_info = _pre_torch_apple_probe()
            if apple_info is not None:
                torch_info = apple_info

    if forced == "cpu":
        torch_info = _blank_probe()
    elif forced in {"cuda", "mps"} and torch_info.get(forced):
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
