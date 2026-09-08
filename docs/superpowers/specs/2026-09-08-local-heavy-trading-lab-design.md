# Local Heavy Trading Lab — Design Specification

Date: 2026-09-08
Status: Proposed for implementation after user review
Repository: lil-fahad/Fahad
Branch: local-heavy-trading-lab

## 1. Objective

Rebuild the options/market prediction project as a local-first AI research and training system that runs on the user's computer instead of Railway. The system must download and retain the full pretrained model snapshots locally, perform real fine-tuning/adaptation where the upstream model supports it, benchmark every model with strict time-aware out-of-sample evaluation, resume interrupted training, and keep all trading execution disabled while the research stack is being trained and validated.

The initial market scope is SPY and SPX, with optional expansion to VIX, QQQ, ES, rates, and historical options-surface data when a reliable source is configured.

## 2. Safety and Execution Boundary

This project is a research and PAPER-trading laboratory. No broker order submission, autonomous live trading, naked option selling, or live-money execution will be implemented in the training phase.

A model may not influence a PAPER trading decision until it passes the promotion gates defined in this specification. A model may never be labeled "accurate" or "profitable" solely from training loss or in-sample results.

## 3. Architecture

The project is split into isolated subsystems:

1. `hardware/` — detects CPU, RAM, NVIDIA CUDA, GPU model/VRAM, Apple MPS/unified memory, precision support, and available disk.
2. `models/` — local Hugging Face model snapshots and immutable manifests.
3. `data/` — raw market data, normalized canonical data, feature tables, and split manifests.
4. `training/` — one trainer per model family plus shared checkpoint/resume infrastructure.
5. `evaluation/` — purged walk-forward evaluation, baselines, calibration, PnL simulation, and statistical confidence.
6. `ensemble/` — trains a meta-model only on out-of-fold predictions from the base models.
7. `artifacts/` — checkpoints, adapters, metrics, reports, plots, and promotion records.
8. `paper/` — optional local PAPER simulator that consumes only promoted models.
9. `cli/` — one command surface for doctor, download, prepare, train, benchmark, report, and resume.

The model trainers communicate only through versioned artifact contracts. Training code does not import broker/execution code.

## 4. Model Set and Real Training Modes

### 4.1 Amazon Chronos-2

Model: `amazon/chronos-2` (~119.5M parameters, Apache-2.0).

Training path:
- Download the full snapshot locally.
- Use the official `chronos-forecasting` training/fine-tuning interface.
- Prefer native `Chronos2Pipeline.fit(...)` fine-tuning for target-domain adaptation.
- Support validation inputs, early stopping/evaluation checkpoints where available, and covariate-aware training.
- Save fine-tuned model artifacts separately from the immutable base snapshot.

Chronos-2 supports actual fine-tuning in the current upstream package; this is not inference-only adaptation.

### 4.2 Google TimesFM 2.5

Model: `google/timesfm-2.5-200m-transformers` (~231.3M parameters; model snapshot ~925 MB, Apache-2.0).

Training path:
- Download the full Transformers snapshot locally.
- Fine-tune with PyTorch + Transformers + PEFT.
- Default adaptation is LoRA because it reduces trainable parameters and VRAM requirements while preserving the base weights.
- Permit full fine-tuning only when hardware and a preflight memory test prove it is safe.
- Support gradient accumulation, mixed precision, gradient clipping, checkpoint/resume, and early stopping.
- Do not externally normalize TimesFM inputs when using the model's native normalization behavior.

TimesFM 2.5 has an upstream LoRA fine-tuning example using Transformers/PEFT and a normal training loss when future values are supplied.

### 4.3 Kronos

Primary model: `NeoQuasar/Kronos-base` (~102.3M parameters) with `NeoQuasar/Kronos-Tokenizer-base`. Optional comparison models: Kronos-small and Kronos-mini.

Training path:
- Download model and tokenizer snapshots locally.
- Train/fine-tune on OHLCV K-line windows using the official Kronos training path.
- Preserve the tokenizer/model pairing in the manifest.
- Train on multiple market regimes and symbols rather than only a single short SPY sample.
- Keep a zero-shot baseline for every fine-tuned checkpoint.

Kronos is specifically designed for financial candlesticks and its published model card includes fine-tuning on user data.

### 4.4 IBM Granite TinyTimeMixer (TTM R2/R2.1)

Model: `ibm-granite/granite-timeseries-ttm-r2` (~805K parameters for the base variant).

Training path:
- Download appropriate minutely/hourly/daily revisions based on the dataset frequency.
- Run zero-shot baseline first.
- Fine-tune using channel-independent and channel-mixing modes where appropriate.
- Support exogenous/control variables.
- Because TTM is small, full/few-shot fine-tuning is allowed on far weaker hardware than the heavy models.

TTM serves as an important low-compute control: a heavy model is not promoted merely because it has more parameters.

### 4.5 FinBERT

Model: `ProsusAI/finbert`.

Training path:
- Download tokenizer/config/weights locally.
- Use the pretrained financial sentiment classifier as the starting point.
- Fine-tune only on labeled financial text datasets with explicit provenance.
- Maintain a clean evaluation set separated by source and time.
- Output calibrated sentiment probabilities; FinBERT never predicts price by itself.

FinBERT votes are auxiliary features for the ensemble and market-regime layer.

### 4.6 Optional TimesFM 3.0 Research Track

TimesFM 3.0 may be added as a research-only experiment after the core system is stable. Its current pretrained-weight license is more restrictive than TimesFM 2.5, so it must live behind a separate license gate and must not silently replace the Apache-2.0 TimesFM 2.5 path.

## 5. Local Model Storage

All heavy snapshots must be downloaded once and retained locally. The downloader will use Hugging Face snapshot semantics and produce a manifest containing:

- model ID
- revision/commit SHA
- file list
- file sizes
- local path
- license metadata when available
- download timestamp
- optional SHA256 for critical files

Recommended layout:

```text
local-heavy-trading-lab/
  models/
    base/
      chronos-2/
      timesfm-2.5/
      kronos-base/
      kronos-tokenizer-base/
      ttm-r2/
      finbert/
    trained/
      chronos-2/
      timesfm-2.5/
      kronos/
      ttm/
      finbert/
  checkpoints/
  data/
    raw/
    canonical/
    features/
    splits/
  artifacts/
    metrics/
    reports/
    predictions/
    promotion/
```

The application must support an alternate storage root on a second SSD because model/checkpoint storage can grow large. The preflight doctor will warn if less than 100 GB free disk is available for a full research run.

## 6. Hardware Detection and Maximum-Compute Policy

The command `lab doctor` determines the actual machine capabilities before installation/training.

### NVIDIA CUDA

If NVIDIA is present:
- detect GPU model, CUDA compute capability, VRAM, driver, CUDA runtime, bf16/fp16 support
- enable TF32 where appropriate
- prefer bf16 on supported Ampere-or-newer devices, otherwise fp16
- use mixed precision, gradient accumulation, pinned memory, persistent DataLoader workers, and gradient checkpointing for heavy models
- use Flash Attention only when the installed model/library/hardware combination is verified compatible
- auto-tune batch size using a small dry run before the main job

Suggested training profiles:
- >=24 GB VRAM: allow full fine-tuning experiments for supported models plus LoRA controls
- 12–23 GB VRAM: LoRA/PEFT by default for TimesFM and other heavy Transformers; native Chronos fine-tune with conservative batch sizing
- 8–11 GB VRAM: low-rank adaptation, gradient checkpointing, small microbatches, CPU offload only when it improves stability
- <8 GB VRAM: heavy models remain available for inference/limited adaptation; TTM and smaller/adapted paths become the primary trainable models

### Apple Silicon

Use MPS when supported. Do not assume CUDA-only packages such as bitsandbytes are available. Use unified-memory-aware batch sizing and fall back per-model when unsupported operators are encountered.

### CPU-only

The system still installs and downloads all requested model snapshots. Full heavy training is not attempted blindly. TTM/FinBERT and small adaptation jobs may train on CPU; the system reports when a requested full run is computationally impractical rather than crashing the machine.

## 7. Data Pipeline

Canonical market rows use UTC timestamps and explicit exchange/session metadata.

Initial resolutions:
- 1 minute
- 5 minute
- 15 minute
- 1 hour
- daily

Initial target series:
- SPY OHLCV
- SPX OHLC where obtainable
- VIX/regime data
- risk-free/short-rate proxy

Optional later inputs:
- QQQ
- ES futures
- breadth/volume indicators
- options IV surface, skew, term structure, OI, volume, Greeks when a reliable historical options source is configured
- financial news and filing text for FinBERT

Raw data is immutable. All feature engineering writes new versioned datasets.

## 8. Leakage Prevention

This is mandatory and enforced in code/tests:

- no random train/test shuffling across time
- chronological splits only
- purged walk-forward folds
- embargo around fold boundaries where overlapping horizons exist
- scalers/normalizers fit only on each training fold
- no future news, future IV, revised economic data, or post-event metadata in historical features
- model-selection decisions use validation folds, never the final locked test period
- ensemble/meta-model trains only on out-of-fold base-model predictions

Any dataset failing timestamp/provenance checks is rejected.

## 9. Training Orchestrator

Main commands:

```text
lab doctor
lab download-models --all
lab ingest --profile spy-spx
lab prepare --profile intraday-v1
lab train chronos2 --profile max
lab train timesfm25 --profile max
lab train kronos --profile max
lab train ttm --profile max
lab train finbert --profile max
lab train-all --profile max
lab benchmark --all
lab train-ensemble
lab report
lab resume <run-id>
```

Every run receives an immutable run ID and writes:
- config snapshot
- hardware snapshot
- dataset manifest
- git commit
- random seeds
- training curves
- checkpoint paths
- evaluation metrics
- failure reason if interrupted

Training must be resumable after reboot or power loss from the latest valid checkpoint.

## 10. Evaluation Protocol

Each forecasting model is measured before and after fine-tuning against at least:

- last-value/random-walk baseline
- drift baseline
- moving-average/EMA baseline
- TTM baseline/control

Metrics include:
- MAE
- RMSE
- normalized error / skill versus last-value baseline
- directional accuracy
- precision/recall for thresholded up/down moves
- calibration/Brier-type metrics for probabilistic outputs where applicable
- simulated net PnL after configurable slippage and transaction costs
- Sharpe/Sortino where statistically meaningful
- max drawdown
- profit factor
- turnover
- performance by volatility/regime/time-of-day

Confidence intervals are reported. Small samples are never presented as definitive accuracy.

## 11. Model Promotion Gate

All models begin in `CANDIDATE` state.

A model can become `SHADOW` after it completes the full benchmark without leakage or integrity failures.

A model can become `PAPER_ELIGIBLE` only when all of the following are satisfied on locked out-of-sample folds:
- positive skill versus the required baseline
- positive net PnL after configured costs for the target strategy simulation
- no catastrophic regime-specific failure hidden by aggregate metrics
- minimum sample requirement met (default >=250 independent forecast origins for the promoted symbol/horizon; higher when available)
- directional result is accompanied by a confidence interval and is not promoted solely from point accuracy
- checkpoint/data/code manifests are reproducible

The ensemble receives the same gate. Base models do not receive arbitrary fixed production weights.

## 12. Ensemble Training

The ensemble is trained only after base-model out-of-fold predictions exist.

Inputs may include:
- Chronos forecast distribution/features
- TimesFM point/quantile forecast features
- Kronos OHLCV forecast features
- TTM forecasts
- FinBERT calibrated sentiment
- technical/regime features

The first meta-model is deliberately simple and auditable (regularized logistic/linear model or gradient-boosted model). A neural meta-model is added only if it proves superior on locked walk-forward evaluation.

Ensemble weights are learned from out-of-fold data; the system does not hard-code a model as superior merely because a previous 12-window experiment favored it.

## 13. Testing

Tests cover:
- hardware detection on mocked CPU/CUDA/MPS systems
- model manifest integrity
- resume/checkpoint recovery
- deterministic split construction
- leakage guards
- training smoke tests with tiny synthetic fixtures
- real-data parser tests
- model adapter contracts
- benchmark math
- promotion gate behavior
- PAPER execution boundary

Heavy-model integration tests are tagged separately so normal unit tests do not redownload hundreds of megabytes.

## 14. Installation Strategy

The repository ships:
- Windows PowerShell bootstrap
- Linux/macOS shell bootstrap
- optional WSL2/NVIDIA path for Windows users
- Python environment lock files
- hardware-specific PyTorch installation resolver

The installer never guesses a CUDA wheel. It detects the system first and then installs the compatible PyTorch stack.

Secrets/API keys live only in a local `.env` excluded from Git. Hugging Face tokens are optional for public models unless a model/provider requires authentication.

## 15. Railway Retirement

Once this written specification is approved:
1. capture the current Railway project/service identifiers for audit only
2. delete the dedicated `fahad-options-ai-v6` Railway project and its services/volumes
3. do not delete the GitHub repository or historical branches
4. preserve no Railway secrets in the new local repository

Railway is not part of the new runtime architecture.

## 16. Implementation Order

1. local project skeleton and CLI
2. hardware doctor and disk preflight
3. model downloader/manifests
4. dataset ingestion/canonical format
5. time-aware split/leakage engine
6. TTM training path (fast control)
7. TimesFM 2.5 LoRA/full-capability trainer
8. Chronos-2 native fine-tuning trainer
9. Kronos fine-tuning trainer
10. FinBERT training/calibration path
11. unified benchmark engine
12. out-of-fold ensemble training
13. promotion registry
14. local PAPER simulator integration
15. end-to-end reproducibility and recovery tests

## 17. Success Criteria

The first milestone is complete only when:
- the target computer passes `lab doctor`
- all selected base model snapshots are stored locally with manifests
- at least TTM, TimesFM 2.5, Chronos-2, Kronos-base, and FinBERT each complete their intended real training/adaptation path or produce a hardware-specific documented block instead of silently falling back
- walk-forward benchmark reports are generated from real SPY/SPX data
- no leakage tests fail
- at least one full interrupted-training resume test succeeds
- no model is promoted based on in-sample metrics
- live-money trading remains impossible in the training build

## 18. Source Capability Notes

Verified before design freeze:
- Amazon Chronos-2 upstream tests expose `Chronos2Pipeline.fit(...)` for fine-tuning.
- Google TimesFM 2.5 has an official repository LoRA fine-tuning example using Transformers + PEFT.
- Kronos-base publishes a user-data fine-tuning path and is trained specifically on financial K-lines.
- IBM TTM R2 explicitly supports zero-shot and fine-tuned forecasting, including channel-mixing and exogenous variables.
- FinBERT is a pretrained financial sentiment text classifier and is treated as a text model, not a standalone price predictor.

This design intentionally chooses actual supported adaptation methods instead of pretending every pretrained model has the same training interface.
