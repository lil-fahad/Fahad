from heavy_lab.hardware import HardwareProfile, detect_hardware, select_training_profile


def test_cpu_profile_never_claims_cuda(monkeypatch, tmp_path):
    monkeypatch.setenv("HEAVY_LAB_FORCE_DEVICE", "cpu")
    p = detect_hardware(tmp_path)
    assert p.device == "cpu"
    assert p.cuda is False
    assert p.mps is False


def test_profile_requires_lora_below_24gb_vram():
    p = HardwareProfile(
        os_name="Windows",
        cpu_count=16,
        device="cuda",
        cuda=True,
        mps=False,
        ram_gb=64.0,
        vram_gb=12.0,
        disk_free_gb=500.0,
        gpu_name="RTX",
        cuda_version="12.8",
        compute_capability="8.9",
        bf16=True,
        fp16=True,
    )
    assert select_training_profile(p) == "lora-heavy"


def test_profile_allows_full_heavy_at_24gb_vram():
    p = HardwareProfile(
        os_name="Linux",
        cpu_count=32,
        device="cuda",
        cuda=True,
        mps=False,
        ram_gb=128.0,
        vram_gb=24.0,
        disk_free_gb=1000.0,
        gpu_name="RTX 4090",
        cuda_version="12.8",
        compute_capability="8.9",
        bf16=True,
        fp16=True,
    )
    assert select_training_profile(p) == "full-heavy"
