from pathlib import Path


def test_windows_max_training_script_can_self_bootstrap_and_download_models():
    script = Path("scripts/start_training_windows.ps1").read_text(encoding="utf-8")

    assert "[switch]$Bootstrap" in script
    assert "[switch]$DownloadModels" in script

    bootstrap_pos = script.index("install_windows.ps1")
    venv_check_pos = script.index('if (-not (Test-Path $Py))')
    assert bootstrap_pos < venv_check_pos

    assert "-Training" in script
    assert "-DownloadModels" in script
