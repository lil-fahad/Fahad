# Options Model Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Chronos-2 + TimesFM 2.5 + FinBERT as an optional, fail-safe ensemble for SPY/SPX options-paper decisions.

**Architecture:** A focused `model_ensemble.py` module lazy-loads optional heavyweight models and returns normalized directional votes. The existing options-paper engine consumes one ensemble result while retaining all pricing, P/L, persistence and safety behavior.

**Tech Stack:** Python 3.12, asyncio, SQLite, httpx; optional PyTorch, Transformers, Chronos Forecasting.

**Spec:** `docs/superpowers/specs/2026-09-08-options-model-ensemble-design.md`

## Global Constraints
- Paper options only; no broker order endpoint.
- SPY and SPX only.
- Heavy model dependencies and weights are optional and never bundled in the core ZIP.
- Missing/broken models fall back to the existing technical signal.
- Existing configuration files must keep loading unchanged.

---

### Task 1: Pure ensemble policy and model adapters
**Files:** create `telegram_bridge/model_ensemble.py`; test `tests/test_model_ensemble.py`.
**Interfaces:** produce `ModelVote`, `EnsembleResult`, `combine_votes`, `OptionsModelEnsemble`.
- [ ] Write tests for agreement, disagreement and no-model fallback.
- [ ] Run tests and confirm RED because the module does not exist.
- [ ] Implement pure policy and lazy optional adapters.
- [ ] Re-run focused tests to GREEN.

### Task 2: Options engine integration and persistence
**Files:** modify `telegram_bridge/options_paper.py`, `telegram_bridge/storage.py`; test `tests/test_model_ensemble.py` and `tests/test_options_paper.py`.
**Interfaces:** `OptionsPaperEngine(..., model_ensemble=None)` and persisted `option_model_signals`.
- [ ] Add failing tests showing an injected ensemble can change entry direction and its components are persisted.
- [ ] Confirm RED.
- [ ] Wire ensemble evaluation through `asyncio.to_thread` and isolate failures.
- [ ] Confirm GREEN and old options tests remain green.

### Task 3: Configuration, Telegram status and downloader
**Files:** modify `telegram_bridge/config.py`, `telegram_bridge/commands.py`, `telegram_bridge/cli.py`; create `requirements-models.txt`, `MODELS_AR.md`; tests `tests/test_model_ensemble.py`, `tests/test_commands.py`.
**Interfaces:** `/optionsmodels`, CLI `configure-options-models`, CLI `download-options-models`.
- [ ] Add failing tests for backward-compatible config and owner-only status command.
- [ ] Confirm RED.
- [ ] Implement configuration/status/downloader and docs.
- [ ] Confirm GREEN.

### Task 4: Release verification and packaging
**Files:** update `README.md`, `README_AR.md`, `VALIDATION.md`, `pyproject.toml`.
- [ ] Run all tests available in the build environment and record the known MCP dependency limitation separately.
- [ ] Run `compileall` and `git diff --check`/equivalent whitespace scan.
- [ ] Build a clean v6 ZIP excluding state, secrets, virtualenvs, caches and VCS metadata.
- [ ] Extract the produced ZIP and rerun focused options/model tests against that exact artifact.
