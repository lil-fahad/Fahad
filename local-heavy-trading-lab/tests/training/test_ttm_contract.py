from heavy_lab.hardware import HardwareProfile


def test_ttm_policy_keeps_full_finetune_on_8gb_cuda():
    from heavy_lab.training.policy import plan_training

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
        bf16=False,
        fp16=True,
    )
    strategy = plan_training("ttm", hardware)
    assert strategy.mode == "full"
    assert strategy.precision == "fp16"


def test_ttm_windows_never_cross_forecast_origin():
    from heavy_lab.training.windows import make_supervised_windows

    values = list(range(20))
    windows = make_supervised_windows(values, context_length=5, prediction_length=2, stride=2)
    assert windows
    for context, target in windows:
        assert len(context) == 5
        assert len(target) == 2
        assert context[-1] < target[0]
