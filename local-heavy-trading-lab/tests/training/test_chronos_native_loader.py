from pathlib import Path


def test_chronos_loader_uses_local_path_device_and_precision(tmp_path: Path):
    from heavy_lab.training.integrations.chronos2_native import load_chronos2_pipeline

    model_path = tmp_path / "chronos-2"
    model_path.mkdir()
    calls = []

    class FakeTorch:
        float16 = "fp16-dtype"
        bfloat16 = "bf16-dtype"
        float32 = "fp32-dtype"

    class FakeBasePipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((Path(path), kwargs))
            return "pipeline"

    result = load_chronos2_pipeline(
        model_path=model_path,
        device="cuda",
        precision="fp16",
        pipeline_cls=FakeBasePipeline,
        torch_module=FakeTorch,
    )

    assert result == "pipeline"
    assert calls == [
        (
            model_path,
            {"device_map": "cuda", "torch_dtype": "fp16-dtype"},
        )
    ]
