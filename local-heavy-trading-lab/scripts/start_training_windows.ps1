param(
    [Parameter(Mandatory = $true)]
    [string]$TrainParquet,
    [Parameter(Mandatory = $true)]
    [string]$ValidationParquet,
    [Parameter(Mandatory = $true)]
    [string]$FinBERTTrainJsonl,
    [Parameter(Mandatory = $true)]
    [string]$FinBERTValidationJsonl,
    [ValidateSet("smoke", "max")]
    [string]$Profile = "smoke",
    [string]$Root = (Get-Location).Path
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path $Root).Path
Set-Location $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "Training environment not found. Run .\scripts\install_windows.ps1 -Training -DownloadModels first."
}

$TrainParquet = (Resolve-Path $TrainParquet).Path
$ValidationParquet = (Resolve-Path $ValidationParquet).Path
$FinBERTTrainJsonl = (Resolve-Path $FinBERTTrainJsonl).Path
$FinBERTValidationJsonl = (Resolve-Path $FinBERTValidationJsonl).Path

$ReadinessJson = & $Py -m heavy_lab.cli training-ready `
    --root $Root `
    --train-parquet $TrainParquet `
    --validation-parquet $ValidationParquet `
    --finbert-train-jsonl $FinBERTTrainJsonl `
    --finbert-validation-jsonl $FinBERTValidationJsonl `
    --json
if ($LASTEXITCODE -ne 0) {
    throw "training-ready command failed before training started."
}
$Readiness = $ReadinessJson | ConvertFrom-Json
if (-not $Readiness.ready) {
    $Missing = ($Readiness.missing -join ", ")
    throw "Training is not ready. Missing or invalid requirements: $Missing"
}

if ($Profile -eq "smoke") {
    $TtmEpochs = 1
    $TtmContext = 64
    $TtmPrediction = 12
    $TimesFmEpochs = 1
    $TimesFmContext = 128
    $TimesFmPrediction = 24
    $ChronosSteps = 50
    $ChronosContext = 128
    $ChronosPrediction = 24
    $KronosTokenizerEpochs = 1
    $KronosPredictorEpochs = 1
    $FinBERTEpochs = 1
} else {
    # RTX 3070 Ti / 8 GB profile: models are still run one at a time.
    # TimesFM remains LoRA and Chronos/Kronos remain low-VRAM by policy.
    $TtmEpochs = 10
    $TtmContext = 512
    $TtmPrediction = 96
    $TimesFmEpochs = 5
    $TimesFmContext = 512
    $TimesFmPrediction = 96
    $ChronosSteps = 3000
    $ChronosContext = 512
    $ChronosPrediction = 96
    $KronosTokenizerEpochs = 3
    $KronosPredictorEpochs = 5
    $FinBERTEpochs = 5
}

function Clear-CudaCache {
    & $Py -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA cleanup failed. Stop before starting the next heavy model."
    }
}

Write-Host "Training readiness: READY"
Write-Host "Profile: $Profile"
Write-Host "GPU: $($Readiness.hardware.gpu_name) | VRAM: $($Readiness.hardware.vram_gb) GB"
Write-Host "Models will run sequentially to stay inside the 8 GB VRAM budget."

Write-Host "[1/5] Training TTM control..."
& $Py -m heavy_lab.cli train-ttm `
    --train-parquet $TrainParquet `
    --validation-parquet $ValidationParquet `
    --root $Root `
    --context-length $TtmContext `
    --prediction-length $TtmPrediction `
    --epochs $TtmEpochs
if ($LASTEXITCODE -ne 0) { throw "TTM training failed." }
Clear-CudaCache

Write-Host "[2/5] Training TimesFM 2.5 (LoRA on 8 GB VRAM)..."
& $Py -m heavy_lab.cli train-timesfm25 `
    --train-parquet $TrainParquet `
    --validation-parquet $ValidationParquet `
    --root $Root `
    --context-length $TimesFmContext `
    --prediction-length $TimesFmPrediction `
    --epochs $TimesFmEpochs `
    --micro-batch-size 1
if ($LASTEXITCODE -ne 0) { throw "TimesFM 2.5 training failed." }
Clear-CudaCache

Write-Host "[3/5] Training Chronos-2 (native low-VRAM)..."
& $Py -m heavy_lab.cli train-chronos2 `
    --train-parquet $TrainParquet `
    --validation-parquet $ValidationParquet `
    --root $Root `
    --context-length $ChronosContext `
    --prediction-length $ChronosPrediction `
    --steps $ChronosSteps `
    --micro-batch-size 1
if ($LASTEXITCODE -ne 0) { throw "Chronos-2 training failed." }
Clear-CudaCache

Write-Host "[4/5] Training Kronos from pinned official source..."
& $Py -m heavy_lab.cli train-kronos `
    $TrainParquet `
    $ValidationParquet `
    --root $Root `
    --lookback-window 64 `
    --predict-window 16 `
    --tokenizer-epochs $KronosTokenizerEpochs `
    --predictor-epochs $KronosPredictorEpochs
if ($LASTEXITCODE -ne 0) { throw "Kronos training failed." }
Clear-CudaCache

Write-Host "[5/5] Training FinBERT..."
& $Py -m heavy_lab.cli train-finbert `
    --train-jsonl $FinBERTTrainJsonl `
    --validation-jsonl $FinBERTValidationJsonl `
    --root $Root `
    --epochs $FinBERTEpochs
if ($LASTEXITCODE -ne 0) { throw "FinBERT training failed." }
Clear-CudaCache

Write-Host "All requested training stages completed for profile '$Profile'."
Write-Host "Review runs/, checkpoints/, and artifacts/ before benchmarking or promotion."
