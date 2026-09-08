# Local Heavy Trading Lab — Evaluation and Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build leakage-safe benchmark, confidence intervals, trading-cost simulation, out-of-fold ensemble training, and an auditable model promotion registry.

**Architecture:** Evaluation consumes immutable prediction artifacts and split manifests, never trainer internals. Base-model out-of-fold predictions are joined by timestamp/symbol/horizon and passed to a simple regularized meta-model first. Promotion state is an explicit registry transition based on locked out-of-sample gates.

**Tech Stack:** numpy, pandas, scikit-learn, scipy, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-local-heavy-trading-lab-design.md`

## Global Constraints

- Baselines include last value/random walk, drift, EMA, and TTM control.
- Minimum default promotion sample is 250 independent forecast origins.
- Small samples must show confidence intervals and cannot be labeled definitive.
- Ensemble trains only on out-of-fold base-model predictions.
- Final locked test period is excluded from model selection and meta-model training.

---

### Task 1: Forecast metrics and baseline skill

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/evaluation/metrics.py`
- Create: `local-heavy-trading-lab/src/heavy_lab/evaluation/baselines.py`
- Test: `local-heavy-trading-lab/tests/evaluation/test_metrics.py`

**Interfaces:**
- Produces: `score_forecast(actual, predicted, reference) -> ForecastMetrics`

- [ ] Write exact numeric tests for MAE, RMSE, directional accuracy, and `skill_vs_last = 1 - mae_model / mae_last`.
- [ ] Verify RED.
- [ ] Implement metrics with NaN/zero-baseline guards.
- [ ] Implement last-value, drift, EMA baselines using past data only.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: add forecast metrics and baselines`.

### Task 2: Confidence intervals and classification calibration

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/evaluation/confidence.py`
- Test: `local-heavy-trading-lab/tests/evaluation/test_confidence.py`

**Interfaces:**
- Produces: `wilson_interval(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]`
- Produces: `brier_score(y_true, probabilities) -> float`

- [ ] Write known-value Wilson interval tests and Brier-score tests.
- [ ] Verify RED.
- [ ] Implement numerically stable confidence helpers.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: add benchmark confidence intervals`.

### Task 3: Cost-aware PnL simulator

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/evaluation/pnl.py`
- Test: `local-heavy-trading-lab/tests/evaluation/test_pnl.py`

**Interfaces:**
- Produces: `simulate_signals(predictions, costs: CostModel) -> PnLMetrics`

- [ ] Test that zero edge with positive costs yields negative net PnL.
- [ ] Test max drawdown, turnover, profit factor on a deterministic equity curve.
- [ ] Verify RED.
- [ ] Implement configurable spread/slippage/commission assumptions and reject negative costs.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: add cost aware signal simulation`.

### Task 4: Unified walk-forward benchmark runner

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/evaluation/benchmark.py`
- Modify: `local-heavy-trading-lab/src/heavy_lab/cli.py`
- Test: `local-heavy-trading-lab/tests/evaluation/test_benchmark.py`

**Interfaces:**
- CLI: `lab benchmark --all`
- Produces: `BenchmarkReport`

- [ ] Write fixture test with two folds and two models; ensure only test rows are scored.
- [ ] Verify RED.
- [ ] Implement report sections by symbol, horizon, regime, time-of-day, model, and baseline.
- [ ] Persist prediction rows and JSON metrics separately.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: add unified walk forward benchmark`.

### Task 5: Out-of-fold ensemble dataset and meta-model

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/ensemble/dataset.py`
- Create: `local-heavy-trading-lab/src/heavy_lab/ensemble/train.py`
- Test: `local-heavy-trading-lab/tests/ensemble/test_ensemble.py`

**Interfaces:**
- CLI: `lab train-ensemble`
- Produces: `MetaModelArtifact`

- [ ] Write test that an in-sample/base-training prediction row is rejected from the ensemble dataset.
- [ ] Verify RED.
- [ ] Join OOF predictions by immutable keys `(timestamp_utc, symbol, horizon, fold_id)`.
- [ ] Train regularized logistic regression for direction and regularized linear regression for return magnitude as initial auditable meta-models.
- [ ] Save feature names, coefficients, scaler, calibration, training rows hash, and metrics.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: add out of fold meta ensemble`.

### Task 6: Promotion registry

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/evaluation/promotion.py`
- Test: `local-heavy-trading-lab/tests/evaluation/test_promotion.py`

**Interfaces:**
- Produces: states `CANDIDATE`, `SHADOW`, `PAPER_ELIGIBLE`
- Produces: `evaluate_promotion(report, manifest, min_samples=250) -> PromotionDecision`

- [ ] Write tests proving a 12-sample 90% directional model cannot become PAPER_ELIGIBLE.
- [ ] Write tests proving negative net PnL blocks promotion despite positive MAE skill.
- [ ] Verify RED.
- [ ] Implement all spec gates and machine-readable rejection reasons.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: add auditable model promotion registry`.

### Task 7: Research report generation

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/report.py`
- Test: `local-heavy-trading-lab/tests/test_report.py`

**Interfaces:**
- CLI: `lab report`

- [ ] Test report contains hardware, data revision, model revision, sample count, confidence interval, costs, and promotion state.
- [ ] Verify RED.
- [ ] Generate Markdown + JSON report and standalone matplotlib PNG plots with no interactive dependency.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: add reproducible research reports`.

## Evaluation Acceptance Gate

Run:

```bash
python -m pytest tests/evaluation tests/ensemble -q
lab benchmark --all --dry-run
lab train-ensemble --dry-run
lab report --dry-run
```

Acceptance requires every reported model metric to be traceable to immutable prediction rows and a split manifest, and no model with fewer than 250 independent origins may become PAPER_ELIGIBLE by default.
