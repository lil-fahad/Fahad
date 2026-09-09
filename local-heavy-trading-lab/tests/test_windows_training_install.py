from pathlib import Path
import tomllib


def test_windows_installer_has_training_mode_and_installs_after_torch():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "install_windows.ps1").read_text(encoding="utf-8")

    assert "[switch]$Training" in script
    assert "install_torch.py" in script
    assert "ttm-train" in script
    assert "timesfm-train" in script
    assert "chronos-train" in script
    assert "finbert-train" in script
    assert "kronos-train" in script
    assert script.index("install_torch.py") < script.index("ttm-train")


def test_kronos_training_extra_avoids_old_huggingface_hub_pin():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    deps = project["project"]["optional-dependencies"]["kronos-train"]
    joined = " ".join(deps).lower()

    assert "einops" in joined
    assert "safetensors" in joined
    assert "pyyaml" in joined
    assert "huggingface_hub==0.33.1" not in joined
