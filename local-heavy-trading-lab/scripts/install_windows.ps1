param(
    [switch]$DownloadModels,
    [string]$Root = (Get-Location).Path
)

$ErrorActionPreference = "Stop"
Set-Location $Root

function Resolve-Python312 {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -c "import sys; assert sys.version_info >= (3,12)" 2>$null
        if ($LASTEXITCODE -eq 0) { return @("py", "-3.12") }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        & python -c "import sys; assert sys.version_info >= (3,12)"
        if ($LASTEXITCODE -eq 0) { return @("python") }
    }
    throw "Python 3.12+ is required. Install Python 3.12 or newer, then run this script again."
}

$Python = Resolve-Python312
if (-not (Test-Path ".venv")) {
    if ($Python.Count -eq 2) { & $Python[0] $Python[1] -m venv .venv }
    else { & $Python[0] -m venv .venv }
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install -e ".[dev]"
& $VenvPython scripts\install_torch.py
& $VenvPython -m heavy_lab.cli doctor --json

if ($DownloadModels) {
    & $VenvPython -m heavy_lab.cli download-models --all --root $Root
}

Write-Host "Local Heavy Trading Lab installed successfully."
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Then run: lab doctor --json"
