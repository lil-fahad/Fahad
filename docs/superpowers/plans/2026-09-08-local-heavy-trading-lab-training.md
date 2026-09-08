# Local Heavy Trading Lab — Heavy Model Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement real train/adaptation paths for TTM, TimesFM 2.5, Chronos-2, Kronos, and FinBERT with hardware-aware maximum-compute settings, checkpoint/resume, and reproducible artifacts.

**Architecture:** Every model implements the same `TrainerAdapter` contract while keeping its native training API. A shared orchestrator selects precision/batch strategy from the hardware profile, records run metadata, and refuses unsafe full fine-tuning when preflight memory fails.

**Tech Stack:** PyTorch, Transformers, PEFT, chronos-forecasting, granite-tsfm, Kronos upstream package/code, scikit-learn calibration, Accelerate where appropriate.

**Spec:** `docs/superpowers/specs/2026-09-08-local-heavy-trading-lab-design.md`

## Global Constraints

- Actual fine-tuning/adaptation only; inference-only paths must be labeled as such.
- Checkpoint/resume is mandatory for every trainable model.
- Mixed precision is hardware-derived; no guessed CUDA wheel or dtype.
- Full heavy fine-tuning requires a successful memory dry run.
- TimesFM native normalization is preserved.
- FinBERT is sentiment-only and never treated as a standalone price predictor.

---

### Task 1: Shared trainer contract and preflight

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/training/base.py`
- Create: `local-heavy-trading-lab/src/heavy_lab/training/preflight.py`
- Test: `local-heavy-trading-lab/tests/training/test_base.py`

**Interfaces:**
- Produces: `TrainerAdapter.prepare(run, dataset, hardware) -> TrainPlan`
- Produces: `TrainerAdapter.train(plan) -> TrainResult`
- Produces: `TrainerAdapter.resume(run_id) -> TrainResult`

- [ ] Write a fake trainer test proving run metadata and latest checkpoint are recorded.
- [ ] Verify RED.
- [ ] Implement shared dataclasses `TrainPlan`, `TrainResult`, `MemoryProbeResult` and conservative batch auto-tuning by tiny forward/backward dry run.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: add shared heavy trainer contract`.

### Task 2: TTM full/few-shot control trainer

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/training/ttm.py`
- Test: `local-heavy-trading-lab/tests/training/test_ttm.py`

**Interfaces:**
- CLI: `lab train ttm --profile max`

- [ ] Write smoke test using a tiny mocked TTM model and chronological train/validation batches.
- [ ] Verify RED.
- [ ] Implement zero-shot baseline first, then fine-tune path using the selected TTM revision and channel mode.
- [ ] Save base metrics, fine-tuned checkpoint, config, and predictions separately.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: add TTM fine tuning control`.

### Task 3: TimesFM 2.5 LoRA and guarded full fine-tuning

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/training/timesfm25.py`
- Test: `local-heavy-trading-lab/tests/training/test_timesfm25.py`

**Interfaces:**
- CLI: `lab train timesfm25 --profile max`
- Produces: PEFT adapter under `models/trained/timesfm-2.5/<run-id>/adapter/`

- [ ] Write test proving `past_values` and `future_values` are supplied and no external scaler is applied.
- [ ] Verify RED.
- [ ] Implement LoRA config against supported projection modules discovered from the loaded model; persist discovered target module names in run config.
- [ ] Add gradient accumulation, clipping, mixed precision, early stopping, and checkpoint resume.
- [ ] Permit full fine-tune only when `MemoryProbeResult.safe_full_finetune is True`; otherwise automatically use LoRA and record the reason.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: add TimesFM 2.5 real LoRA training`.

### Task 4: Chronos-2 native fine-tuning

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/training/chronos2.py`
- Test: `local-heavy-trading-lab/tests/training/test_chronos2.py`

**Interfaces:**
- CLI: `lab train chronos2 --profile max`

- [ ] Write test against a fake `Chronos2Pipeline` asserting the native fit/fine-tune API receives training and validation data separately.
- [ ] Verify RED.
- [ ] Implement native Chronos-2 fine-tuning using the upstream API, with optional covariates when available.
- [ ] Save fine-tuned model separately from immutable base snapshot and preserve the base revision in the manifest.
- [ ] Add resume via latest valid checkpoint/artifact supported by the upstream fit path.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: add Chronos 2 native fine tuning`.

### Task 5: Kronos K-line fine-tuning

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/training/kronos.py`
- Test: `local-heavy-trading-lab/tests/training/test_kronos.py`

**Interfaces:**
- CLI: `lab train kronos --profile max`

- [ ] Write test ensuring OHLCV window ordering and tokenizer/model manifest pairing are enforced.
- [ ] Verify RED.
- [ ] Implement official-style K-line dataset windows across multiple symbols/regimes; refuse mismatched tokenizer/model revisions.
- [ ] Train/fine-tune and preserve a zero-shot baseline for the identical test folds.
- [ ] Save checkpoints and resume state.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: add Kronos financial K line fine tuning`.

### Task 6: FinBERT sentiment fine-tuning and calibration

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/training/finbert.py`
- Create: `local-heavy-trading-lab/src/heavy_lab/training/calibration.py`
- Test: `local-heavy-trading-lab/tests/training/test_finbert.py`

**Interfaces:**
- CLI: `lab train finbert --profile max`

- [ ] Write test proving text train/eval are separated by source/time and output is calibrated probabilities summing to 1.
- [ ] Verify RED.
- [ ] Implement Transformers classifier fine-tune on provenance-tagged labeled financial text only.
- [ ] Fit temperature/isotonic calibration on validation predictions, never the locked test set.
- [ ] Save tokenizer, model, label map, calibration object, and metrics.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: add FinBERT training and calibration`.

### Task 7: Unified train-all orchestrator

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/training/orchestrator.py`
- Modify: `local-heavy-trading-lab/src/heavy_lab/cli.py`
- Test: `local-heavy-trading-lab/tests/training/test_orchestrator.py`

**Interfaces:**
- CLI: `lab train-all --profile max`

- [ ] Write a fake-adapter test proving models train sequentially by default and each run can resume independently.
- [ ] Verify RED.
- [ ] Implement sequential heavy training to avoid VRAM contention; permit explicit parallelism only for safe low-memory jobs.
- [ ] On failure, mark that model run failed while preserving completed model artifacts.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: add resumable heavy training orchestrator`.

## Training Acceptance Gate

Run:

```bash
python -m pytest tests/training -q
lab train ttm --profile smoke
lab train timesfm25 --profile smoke
lab train chronos2 --profile smoke
lab train kronos --profile smoke
lab train finbert --profile smoke
```

Acceptance requires every trainer to execute a real optimizer/update path in smoke mode or explicitly emit a hardware/dependency block. No trainer may silently fall back to inference-only behavior.
