# Local Heavy Trading Lab — PAPER Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local PAPER-only runtime that consumes only PAPER_ELIGIBLE artifacts and preserves a hard architectural barrier against live-money execution.

**Architecture:** The PAPER runtime loads promoted model manifests and predictions through a read-only inference interface. There is no broker SDK, order endpoint, credential schema, or live execution code in the training package. A local SQLite ledger stores simulated positions and PnL.

**Tech Stack:** Python 3.12, SQLite, pydantic, pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-local-heavy-trading-lab-design.md`

## Global Constraints

- No broker SDK dependencies.
- No real order endpoint or API key fields.
- Only `PAPER_ELIGIBLE` model artifacts may influence simulated positions.
- Candidate and shadow models may log predictions but cannot alter PAPER positions.
- Runtime must remain local-first and resumable.

---

### Task 1: Hard execution boundary

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/paper/boundary.py`
- Test: `local-heavy-trading-lab/tests/paper/test_boundary.py`

**Interfaces:**
- Produces: `assert_paper_only_environment() -> None`

- [ ] Write test scanning declared dependencies/modules for common broker clients and failing if any are present.
- [ ] Verify RED.
- [ ] Implement explicit forbidden-import list and config validation rejecting keys such as `broker_api_key`, `broker_secret`, `live_trading`, `order_endpoint`.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: enforce paper only execution boundary`.

### Task 2: Promotion-aware inference registry

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/paper/models.py`
- Test: `local-heavy-trading-lab/tests/paper/test_models.py`

**Interfaces:**
- Produces: `load_paper_models(registry_path: Path) -> list[PaperModel]`

- [ ] Write test proving `SHADOW` artifact is visible for logging but excluded from active model list.
- [ ] Verify RED.
- [ ] Implement manifest hash verification and promotion-state filtering.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: load only promoted paper models`.

### Task 3: Local simulated ledger

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/paper/ledger.py`
- Test: `local-heavy-trading-lab/tests/paper/test_ledger.py`

**Interfaces:**
- Produces: `PaperLedger.open_position(...)`, `mark(...)`, `close_position(...)`

- [ ] Write deterministic tests for opening, marking, closing, realized/unrealized PnL, and restart persistence.
- [ ] Verify RED.
- [ ] Implement SQLite WAL mode with transactions and idempotent event IDs.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: add persistent local paper ledger`.

### Task 4: PAPER strategy simulator

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/paper/engine.py`
- Test: `local-heavy-trading-lab/tests/paper/test_engine.py`

**Interfaces:**
- CLI: `lab paper run`
- Produces: `PaperDecision`

- [ ] Write test proving a candidate/shadow disagreement cannot change a position decision made by PAPER_ELIGIBLE models.
- [ ] Verify RED.
- [ ] Implement signal aggregation, max-risk config, configurable costs, and no naked short options.
- [ ] Add deterministic replay mode from historical prediction artifacts.
- [ ] Run tests and verify PASS.
- [ ] Commit with `feat: add local paper strategy engine`.

### Task 5: End-to-end reproducibility and recovery

**Files:**
- Create: `local-heavy-trading-lab/tests/e2e/test_recovery.py`
- Create: `local-heavy-trading-lab/tests/e2e/test_no_live_execution.py`

**Interfaces:**
- Validates complete `doctor -> prepare -> train smoke -> benchmark -> promote -> paper replay` path.

- [ ] Simulate interrupted training and restart from latest checkpoint.
- [ ] Simulate process restart with an open PAPER position and verify ledger recovery.
- [ ] Scan source tree and dependency lock for broker/live execution surfaces.
- [ ] Run `python -m pytest tests/e2e -q` and verify PASS.
- [ ] Commit with `test: verify local lab recovery and paper boundary`.

## PAPER Acceptance Gate

Run:

```bash
python -m pytest tests/paper tests/e2e -q
lab paper run --replay artifacts/predictions/example.parquet --dry-run
```

Acceptance requires zero broker/live-trading surface, persistent simulated accounting, and exclusive use of PAPER_ELIGIBLE artifacts for actionable PAPER decisions.
