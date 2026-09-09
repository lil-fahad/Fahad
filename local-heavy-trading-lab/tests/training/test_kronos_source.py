import json
from pathlib import Path


def test_ensure_kronos_source_clones_exact_revision_once(tmp_path: Path):
    from heavy_lab.training.integrations.kronos_official import (
        KRONOS_SOURCE_REPOSITORY,
        KRONOS_SOURCE_REVISION,
        ensure_kronos_source,
    )

    vendor_root = tmp_path / "vendor"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:2] == ["git", "clone"]:
            target = Path(command[-1])
            (target / "finetune_csv").mkdir(parents=True)
            (target / "finetune_csv" / "train_sequential.py").write_text("# fixture", encoding="utf-8")

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    source = ensure_kronos_source(vendor_root=vendor_root, run_command=fake_run)
    assert source == vendor_root.resolve() / f"kronos-{KRONOS_SOURCE_REVISION[:12]}"
    assert calls[0][0] == ["git", "clone", "--filter=blob:none", KRONOS_SOURCE_REPOSITORY, str(source)]
    assert calls[1][0] == ["git", "-C", str(source), "checkout", "--detach", KRONOS_SOURCE_REVISION]
    marker = json.loads((source / ".source.json").read_text(encoding="utf-8"))
    assert marker == {"repository": KRONOS_SOURCE_REPOSITORY, "revision": KRONOS_SOURCE_REVISION}

    call_count = len(calls)
    assert ensure_kronos_source(vendor_root=vendor_root, run_command=fake_run) == source
    assert len(calls) == call_count
