from heavy_lab.install import resolve_torch_install


def test_cpu_never_uses_cuda_index():
    cmd = resolve_torch_install(device="cpu", cuda_version=None, os_name="Linux")
    assert "https://download.pytorch.org/whl/cpu" in cmd
    assert not any("/cu" in part for part in cmd)


def test_cuda_128_uses_official_cu128_index():
    cmd = resolve_torch_install(device="cuda", cuda_version="12.8", os_name="Windows")
    assert "https://download.pytorch.org/whl/cu128" in cmd


def test_macos_mps_uses_standard_pypi_wheel():
    cmd = resolve_torch_install(device="mps", cuda_version=None, os_name="Darwin")
    assert "--index-url" not in cmd
    assert "torch" in cmd
