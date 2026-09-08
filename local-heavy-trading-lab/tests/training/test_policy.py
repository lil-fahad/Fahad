from heavy_lab.hardware import HardwareProfile


def _rtx3070ti():
    return HardwareProfile(
        os_name="Windows",
        cpu_count=16,
        device="cuda",
        cuda=True,
        mps=False,
        ram_gb=32.0,
        vram_gb=8.0,
        disk_free_gb=500.0,
        gpu_name="NVIDIA GeForce RTX 3070 Ti",
        cuda_version="12.8",
        compute_capability="8.6",
        bf16=True,
        fp16=True,
    )


def test_8gb_gpu_uses_safe_real_training_modes():
    from heavy_lab.training.policy import plan_training

    hw = _rtx3070ti()
    assert plan_training("ttm", hw).mode == "full"
    assert plan_training("timesfm25", hw).mode == "lora"
    assert plan_training("chronos2", hw).mode == "native-low-vram"
    assert plan_training("kronos", hw).mode == "native-low-vram"
    assert plan_training("finbert", hw).mode == "full"


def test_heavy_transformers_use_gradient_accumulation_on_8gb():
    from heavy_lab.training.policy import plan_training

    hw = _rtx3070ti()
    plan = plan_training("timesfm25", hw)
    assert plan.micro_batch_size == 1
    assert plan.gradient_accumulation_steps >= 8
    assert plan.gradient_checkpointing is True
