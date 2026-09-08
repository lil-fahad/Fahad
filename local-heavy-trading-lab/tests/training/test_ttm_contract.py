def test_ttm_profile_uses_decoder_only_on_low_vram():
    from heavy_lab.training.ttm import resolve_ttm_strategy

    strategy = resolve_ttm_strategy(device="cuda", vram_gb=8.0)
    assert strategy.freeze_backbone is True
    assert strategy.precision == "fp16"
    assert strategy.gradient_accumulation_steps >= 1


def test_ttm_profile_allows_full_finetune_on_large_gpu():
    from heavy_lab.training.ttm import resolve_ttm_strategy

    strategy = resolve_ttm_strategy(device="cuda", vram_gb=24.0)
    assert strategy.freeze_backbone is False


def test_ttm_windows_never_cross_forecast_origin():
    from heavy_lab.training.windows import make_supervised_windows

    values = list(range(20))
    windows = make_supervised_windows(values, context_length=5, prediction_length=2, stride=2)
    assert windows
    for context, target in windows:
        assert len(context) == 5
        assert len(target) == 2
        assert context[-1] < target[0]
