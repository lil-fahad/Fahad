# Local Heavy Trading Lab — Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build immutable real-market ingestion, canonical storage, deterministic features, and leakage-safe purged walk-forward splits for SPY/SPX research.

**Architecture:** Raw downloads are immutable; normalization creates versioned canonical Parquet datasets; feature generation is deterministic; split manifests are generated from timestamps and forecast horizons only. No trainer is allowed to build its own ad-hoc split.

**Tech Stack:** Python 3.12, pandas, pyarrow, numpy, httpx, pydantic, pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-local-heavy-trading-lab-design.md`

## Global Constraints

- UTC timestamps only in canonical tables.
- No random train/test shuffling.
- Raw data is immutable and provenance-tagged.
- Scalers and feature transforms are fitted per training fold only.
- Overlapping forecast horizons require purge/embargo at fold boundaries.
- Final locked test period is never used for model selection.

---

### Task 1: Canonical market schema and provenance

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/data/schema.py`
- Create: `local-heavy-trading-lab/src/heavy_lab/data/provenance.py`
- Test: `local-heavy-trading-lab/tests/data/test_schema.py`

**Interfaces:**
- Produces: `CanonicalBar`
- Produces: `DatasetManifest`
- Produces: `validate_canonical_frame(df) -> None`

- [ ] Write tests asserting required columns: `timestamp_utc`, `symbol`, `open`, `high`, `low`, `close`, `volume`, `session`, `source`, `source_revision`.
- [ ] Verify RED with `python -m pytest tests/data/test_schema.py -q`.
- [ ] Implement dtype/range/monotonicity validation and duplicate timestamp rejection.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: add canonical market data schema`.

### Task 2: Immutable SPY/SPX ingestion adapters

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/data/providers/base.py`
- Create: `local-heavy-trading-lab/src/heavy_lab/data/providers/nasdaq.py`
- Create: `local-heavy-trading-lab/src/heavy_lab/data/ingest.py`
- Test: `local-heavy-trading-lab/tests/data/test_ingest.py`

**Interfaces:**
- Produces: `MarketProvider.fetch(symbol: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame`
- CLI: `lab ingest --profile spy-spx`

- [ ] Write fixture-driven parser tests from captured provider JSON; tests must not depend on live internet.
- [ ] Verify RED.
- [ ] Implement retry/timeouts, source timestamp preservation, immutable raw file naming `{source}/{symbol}/{interval}/{sha256}.json`.
- [ ] Normalize into canonical Parquet without modifying raw payloads.
- [ ] Verify parser tests and manifest checks PASS.
- [ ] Commit with `feat: add immutable SPY SPX ingestion`.

### Task 3: Deterministic features and targets

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/data/features.py`
- Create: `local-heavy-trading-lab/src/heavy_lab/data/targets.py`
- Test: `local-heavy-trading-lab/tests/data/test_features.py`

**Interfaces:**
- Produces: `build_features(df, profile: str) -> pd.DataFrame`
- Produces: `build_targets(df, horizons: list[int]) -> pd.DataFrame`

- [ ] Test that feature at time `t` is unchanged if future rows after `t` are modified.
- [ ] Verify RED.
- [ ] Implement returns, ATR-like volatility, realized volatility, EMA distance, volume z-score using past-only windows.
- [ ] Implement future-return targets in a separate table so target columns never leak into model inputs.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: add deterministic past-only features`.

### Task 4: Purged walk-forward split engine

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/data/splits.py`
- Test: `local-heavy-trading-lab/tests/data/test_splits.py`

**Interfaces:**
- Produces: `build_walk_forward_splits(index, train_size, val_size, test_size, horizon, embargo) -> list[Fold]`
- CLI: `lab prepare --profile intraday-v1`

- [ ] Write a test with overlapping labels proving the last `horizon` rows of train are purged before validation.
- [ ] Write a test proving embargo rows are excluded after validation/test boundaries.
- [ ] Verify RED.
- [ ] Implement deterministic chronological folds and JSON split manifests containing exact timestamp boundaries and row hashes.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: add purged walk forward splits`.

### Task 5: Fold-local transforms

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/data/transforms.py`
- Test: `local-heavy-trading-lab/tests/data/test_transforms.py`

**Interfaces:**
- Produces: `FoldTransformer.fit(train_df) -> FoldTransformer`
- Produces: `transform(df) -> pd.DataFrame`

- [ ] Test that validation outliers do not change train-fitted scaler parameters.
- [ ] Verify RED.
- [ ] Implement robust/standard scaling selected by profile, persisting fitted parameters per fold.
- [ ] Ensure TimesFM path can opt out of external normalization.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: enforce fold local transforms`.

## Data Acceptance Gate

Run:

```bash
python -m pytest tests/data -q
lab ingest --profile spy-spx --dry-run
lab prepare --profile intraday-v1 --dry-run
```

Acceptance requires deterministic split manifests, no future-feature leakage, immutable raw inputs, and parser tests independent of live provider availability.
