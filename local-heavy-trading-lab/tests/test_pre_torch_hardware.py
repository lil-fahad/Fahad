from pathlib import Path


def test_nvidia_is_detected_before_torch_is_installed(monkeypatch, tmp_path: Path):
    import heavy_lab.hardware as hw

    monkeypatch.delenv("HEAVY_LAB_FORCE_DEVICE", raising=False)
    monkeypatch.setattr(hw, "_torch_probe", lambda: {
        "cuda": False, "mps": False, "device": "cpu", "vram_gb": None,
        "gpu_name": None, "cuda_version": None, "compute_capability": None,
        "bf16": False, "fp16": False,
    })
    monkeypatch.setattr(hw, "_nvidia_smi_probe", lambda: {
        "cuda": True,
        "mps": False,
        "device": "cuda",
        "vram_gb": 24.0,
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "cuda_version": "12.8",
        "compute_capability": "8.9",
        "bf16": True,
        "fp16": True,
    })

    profile = hw.detect_hardware(tmp_path)
    assert profile.device == "cuda"
    assert profile.gpu_name == "NVIDIA GeForce RTX 4090"
    assert profile.vram_gb == 24.0
    assert profile.cuda_version == "12.8"
    assert hw.select_training_profile(profile) == "full-heavy"


def test_apple_silicon_is_marked_mps_candidate_before_torch(monkeypatch, tmp_path: Path):
    import heavy_lab.hardware as hw

    monkeypatch.delenv("HEAVY_LAB_FORCE_DEVICE", raising=False)
    monkeypatch.setattr(hw, "_torch_probe", lambda: {
        "cuda": False, "mps": False, "device": "cpu", "vram_gb": None,
        "gpu_name": None, "cuda_version": None, "compute_capability": None,
        "bf16": False, "fp16": False,
    })
    monkeypatch.setattr(hw, "_nvidia_smi_probe", lambda: None)
    monkeypatch.setattr(hw.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hw.platform, "machine", lambda: "arm64")

    profile = hw.detect_hardware(tmp_path)
    assert profile.device == "mps"
    assert profile.mps is True
