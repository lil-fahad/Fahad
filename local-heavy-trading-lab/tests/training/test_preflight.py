from heavy_lab.hardware import HardwareProfile


def test_memory_probe_is_conservative_on_eight_gb_gpu():
    from heavy_lab.training.preflight import probe_training_memory

    hardware = HardwareProfile(
        os_name="Windows",
        cpu_count=16,
        device="cuda",
        cuda=True,
        mps=False,
        ram_gb=32.0,
        vram_gb=8.0,
        disk_free_gb=500.0,
        gpu_name="RTX 3070 Ti",
        cuda_version="12.8",
        compute_capability="8.6",
        bf16=True,
        fp16=True,
    )

    def trial(batch_size: int) -> float:
        if batch_size > 1:
            raise MemoryError("simulated oom")
        return 5.5

    result = probe_training_memory(
        hardware,
        model_name="timesfm25",
        candidate_batch_sizes=(1, 2, 4),
        trial=trial,
    )

    assert result.max_micro_batch_size == 1
    assert result.safe_full_finetune is False
    assert result.peak_memory_gb == 5.5
    assert "24" in result.reason
