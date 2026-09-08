# Local Heavy Trading Lab — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the local-first project skeleton, hardware doctor, model snapshot downloader, resumable run registry, and platform installers required before any heavy model training.

**Architecture:** The project lives under `local-heavy-trading-lab/` and is isolated from the legacy Railway code. A Typer CLI calls small service modules; immutable manifests are written as JSON; all heavy model files stay outside Git. Hardware detection is explicit and produces a machine profile consumed by every later trainer.

**Tech Stack:** Python 3.12, Typer, Pydantic 2, psutil, platformdirs, huggingface_hub, PyTorch (resolved after hardware detection), pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-local-heavy-trading-lab-design.md`

## Global Constraints

- Training build must contain no broker order submission or live-money execution.
- Python version is `>=3.12`.
- Heavy model snapshots are downloaded once and stored under configurable local storage, never committed to Git.
- Default full-run disk warning threshold is 100 GB free.
- Hardware selection must detect CUDA/MPS/CPU before installing the final PyTorch build; never guess a CUDA wheel.
- All runs receive immutable run IDs and resumable state.
- Secrets live only in local `.env`, excluded from Git.

---

### Task 1: Create isolated package, CLI, and filesystem contract

**Files:**
- Create: `local-heavy-trading-lab/pyproject.toml`
- Create: `local-heavy-trading-lab/.gitignore`
- Create: `local-heavy-trading-lab/src/heavy_lab/__init__.py`
- Create: `local-heavy-trading-lab/src/heavy_lab/cli.py`
- Create: `local-heavy-trading-lab/src/heavy_lab/paths.py`
- Test: `local-heavy-trading-lab/tests/test_paths.py`

**Interfaces:**
- Produces: `LabPaths.from_root(root: Path) -> LabPaths`
- Produces: CLI command `lab doctor`

- [ ] **Step 1: Write the failing path-layout test**

```python
from pathlib import Path
from heavy_lab.paths import LabPaths


def test_lab_paths_are_isolated(tmp_path: Path):
    p = LabPaths.from_root(tmp_path)
    assert p.models_base == tmp_path / "models" / "base"
    assert p.models_trained == tmp_path / "models" / "trained"
    assert p.checkpoints == tmp_path / "checkpoints"
    assert p.data_raw == tmp_path / "data" / "raw"
    assert p.artifacts == tmp_path / "artifacts"
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_paths.py -q`
Expected: import failure because `heavy_lab.paths` does not exist.

- [ ] **Step 3: Implement the package and `LabPaths`**

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class LabPaths:
    root: Path
    models_base: Path
    models_trained: Path
    checkpoints: Path
    data_raw: Path
    data_canonical: Path
    data_features: Path
    data_splits: Path
    artifacts: Path

    @classmethod
    def from_root(cls, root: Path) -> "LabPaths":
        root = Path(root).expanduser().resolve()
        return cls(
            root=root,
            models_base=root / "models" / "base",
            models_trained=root / "models" / "trained",
            checkpoints=root / "checkpoints",
            data_raw=root / "data" / "raw",
            data_canonical=root / "data" / "canonical",
            data_features=root / "data" / "features",
            data_splits=root / "data" / "splits",
            artifacts=root / "artifacts",
        )
```

- [ ] **Step 4: Add `pyproject.toml` and CLI entry point**

Use project name `fahad-heavy-trading-lab`, Python `>=3.12`, dependencies `typer`, `pydantic`, `psutil`, `platformdirs`, `huggingface_hub`, and dev dependency `pytest`. Expose `lab = "heavy_lab.cli:app"`.

- [ ] **Step 5: Run unit tests**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add local-heavy-trading-lab
git commit -m "feat: scaffold local heavy trading lab"
```

### Task 2: Hardware doctor and maximum-compute profile

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/hardware.py`
- Create: `local-heavy-trading-lab/src/heavy_lab/doctor.py`
- Test: `local-heavy-trading-lab/tests/test_hardware.py`

**Interfaces:**
- Produces: `HardwareProfile.detect() -> HardwareProfile`
- Produces: `select_training_profile(profile: HardwareProfile) -> str`
- CLI: `lab doctor --json`

- [ ] **Step 1: Write mocked CPU/CUDA/MPS tests**

```python
def test_cpu_profile_never_claims_cuda(monkeypatch):
    from heavy_lab.hardware import detect_hardware
    monkeypatch.setenv("HEAVY_LAB_FORCE_DEVICE", "cpu")
    p = detect_hardware()
    assert p.device == "cpu"
    assert p.cuda is False


def test_profile_requires_lora_below_24gb_vram():
    from heavy_lab.hardware import HardwareProfile, select_training_profile
    p = HardwareProfile(device="cuda", cuda=True, mps=False, ram_gb=64, vram_gb=12, disk_free_gb=500)
    assert select_training_profile(p) == "lora-heavy"
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_hardware.py -q`
Expected: FAIL because the API does not exist.

- [ ] **Step 3: Implement detection**

Detect OS, CPU count, RAM, disk free, NVIDIA GPU name/VRAM/compute capability when `torch.cuda` is available, MPS support when available, bf16/fp16 capability, and a conservative profile: `full-heavy`, `lora-heavy`, `low-vram`, or `cpu-control`.

- [ ] **Step 4: Add disk preflight**

If `disk_free_gb < 100`, return warning code `LOW_DISK_FOR_FULL_RUN`; do not abort `lab doctor`.

- [ ] **Step 5: Run tests and CLI smoke**

Run: `python -m pytest tests/test_hardware.py -q && lab doctor --json`
Expected: PASS and valid JSON.

- [ ] **Step 6: Commit**

```bash
git add local-heavy-trading-lab/src/heavy_lab/hardware.py local-heavy-trading-lab/src/heavy_lab/doctor.py local-heavy-trading-lab/tests/test_hardware.py
git commit -m "feat: add hardware doctor and compute profiles"
```

### Task 3: Model catalog, immutable manifests, and full snapshot downloader

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/models/catalog.py`
- Create: `local-heavy-trading-lab/src/heavy_lab/models/download.py`
- Create: `local-heavy-trading-lab/src/heavy_lab/models/manifest.py`
- Test: `local-heavy-trading-lab/tests/test_model_manifests.py`

**Interfaces:**
- Produces: `MODEL_CATALOG: dict[str, ModelSpec]`
- Produces: `download_model(name: str, paths: LabPaths, revision: str | None = None) -> ModelManifest`
- CLI: `lab download-models --all`

- [ ] **Step 1: Write catalog test**

```python
def test_required_heavy_models_are_cataloged():
    from heavy_lab.models.catalog import MODEL_CATALOG
    assert MODEL_CATALOG["chronos2"].repo_id == "amazon/chronos-2"
    assert MODEL_CATALOG["timesfm25"].repo_id == "google/timesfm-2.5-200m-transformers"
    assert MODEL_CATALOG["kronos"].repo_id == "NeoQuasar/Kronos-base"
    assert MODEL_CATALOG["kronos_tokenizer"].repo_id == "NeoQuasar/Kronos-Tokenizer-base"
    assert MODEL_CATALOG["ttm"].repo_id == "ibm-granite/granite-timeseries-ttm-r2"
    assert MODEL_CATALOG["finbert"].repo_id == "ProsusAI/finbert"
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_model_manifests.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement downloader with `snapshot_download`**

Use `huggingface_hub.snapshot_download(repo_id=..., revision=..., local_dir=..., local_dir_use_symlinks=False)` and never write tokens to the manifest.

- [ ] **Step 4: Write manifest atomically**

Manifest fields: model name, repo ID, resolved revision/commit SHA, local path, download UTC timestamp, total bytes, file list with relative path/size, and license metadata if available. Write to temporary file then `os.replace`.

- [ ] **Step 5: Run tests with mocked HF calls**

Run: `python -m pytest tests/test_model_manifests.py -q`
Expected: PASS without downloading model weights.

- [ ] **Step 6: Commit**

```bash
git add local-heavy-trading-lab/src/heavy_lab/models local-heavy-trading-lab/tests/test_model_manifests.py
git commit -m "feat: add immutable model snapshot downloader"
```

### Task 4: Run registry, checkpoints, and crash-safe resume

**Files:**
- Create: `local-heavy-trading-lab/src/heavy_lab/runs.py`
- Create: `local-heavy-trading-lab/src/heavy_lab/checkpoints.py`
- Test: `local-heavy-trading-lab/tests/test_resume.py`

**Interfaces:**
- Produces: `RunRegistry.start(kind: str, config: dict) -> RunRecord`
- Produces: `RunRegistry.mark_checkpoint(run_id: str, checkpoint: Path) -> None`
- Produces: `RunRegistry.resume(run_id: str) -> RunRecord`

- [ ] **Step 1: Write interruption/resume test**

```python
def test_resume_returns_latest_valid_checkpoint(tmp_path):
    from heavy_lab.runs import RunRegistry
    reg = RunRegistry(tmp_path)
    run = reg.start("timesfm25", {"seed": 7})
    ckpt = tmp_path / "checkpoint-20"
    ckpt.mkdir()
    reg.mark_checkpoint(run.run_id, ckpt)
    resumed = reg.resume(run.run_id)
    assert resumed.latest_checkpoint == str(ckpt)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_resume.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement atomic run-state writes**

Each run stores `config.json`, `hardware.json`, `state.json`, `git.json`, and `events.jsonl`. State transitions: `CREATED -> RUNNING -> INTERRUPTED|FAILED|COMPLETED`.

- [ ] **Step 4: Add corruption guard**

A checkpoint is valid only if the directory exists and contains a trainer-specific completion marker such as `trainer_state.json` or `checkpoint.ok`.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_resume.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add local-heavy-trading-lab/src/heavy_lab/runs.py local-heavy-trading-lab/src/heavy_lab/checkpoints.py local-heavy-trading-lab/tests/test_resume.py
git commit -m "feat: add resumable training run registry"
```

### Task 5: Cross-platform bootstrap installers

**Files:**
- Create: `local-heavy-trading-lab/scripts/install_windows.ps1`
- Create: `local-heavy-trading-lab/scripts/install_unix.sh`
- Create: `local-heavy-trading-lab/scripts/install_torch.py`
- Test: `local-heavy-trading-lab/tests/test_torch_resolver.py`

**Interfaces:**
- Produces: `resolve_torch_install(profile: HardwareProfile) -> list[str]`

- [ ] **Step 1: Write resolver tests**

```python
def test_cpu_never_uses_cuda_index():
    from heavy_lab.install import resolve_torch_install
    cmd = resolve_torch_install(device="cpu", cuda_version=None)
    assert all("cu12" not in part for part in cmd)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_torch_resolver.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement bootstrap behavior**

Install Python project first without PyTorch, run hardware probe, choose official CPU/CUDA wheel source based on detected platform, then install model extras. On Apple Silicon use the normal macOS PyTorch wheels; do not install bitsandbytes automatically.

- [ ] **Step 4: Add `--dry-run` to installer resolver**

`python scripts/install_torch.py --dry-run` must print the exact pip command without executing it.

- [ ] **Step 5: Run unit tests and syntax checks**

Run: `python -m pytest -q && python scripts/install_torch.py --dry-run`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add local-heavy-trading-lab/scripts local-heavy-trading-lab/tests/test_torch_resolver.py
git commit -m "feat: add cross-platform local installers"
```

## Foundation Acceptance Gate

Run from `local-heavy-trading-lab/`:

```bash
python -m pytest -q
lab doctor --json
lab download-models --help
python scripts/install_torch.py --dry-run
```

Acceptance requires all tests passing, valid doctor JSON, no secrets in Git, and no broker/live-execution dependency anywhere in the package.
