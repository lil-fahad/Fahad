from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class ModelFile:
    path: str
    size_bytes: int


@dataclass(frozen=True)
class ModelManifest:
    name: str
    repo_id: str
    requested_revision: str | None
    resolved_revision: str
    local_path: str
    license: str | None
    downloaded_at_utc: str
    total_bytes: int
    files: tuple[ModelFile, ...]
    manifest_path: str

    def as_dict(self) -> dict:
        data = asdict(self)
        data["files"] = [asdict(item) for item in self.files]
        return data


def inventory_files(root: Path) -> tuple[ModelFile, ...]:
    root = Path(root)
    items: list[ModelFile] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"):
        items.append(ModelFile(path=str(path.relative_to(root)).replace(os.sep, "/"), size_bytes=path.stat().st_size))
    return tuple(items)


def build_manifest(
    *,
    name: str,
    repo_id: str,
    requested_revision: str | None,
    resolved_revision: str,
    local_path: Path,
    license_name: str | None,
) -> ModelManifest:
    local_path = Path(local_path).resolve()
    files = inventory_files(local_path)
    manifest_path = local_path / "manifest.json"
    return ModelManifest(
        name=name,
        repo_id=repo_id,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        local_path=str(local_path),
        license=license_name,
        downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
        total_bytes=sum(item.size_bytes for item in files),
        files=files,
        manifest_path=str(manifest_path),
    )


def write_manifest_atomic(manifest: ModelManifest) -> Path:
    target = Path(manifest.manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)
    return target
