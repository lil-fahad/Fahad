from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from heavy_lab.models.catalog import MODEL_CATALOG
from heavy_lab.models.manifest import ModelManifest, build_manifest, write_manifest_atomic
from heavy_lab.paths import LabPaths


def _license_from_info(info) -> str | None:
    card = getattr(info, "cardData", None)
    if isinstance(card, dict):
        value = card.get("license")
        return str(value) if value else None
    if card is not None:
        value = getattr(card, "license", None)
        return str(value) if value else None
    return None


def download_model(name: str, paths: LabPaths, revision: str | None = None) -> ModelManifest:
    if name not in MODEL_CATALOG:
        raise KeyError(f"Unknown model: {name}")
    spec = MODEL_CATALOG[name]
    paths.models_base.mkdir(parents=True, exist_ok=True)
    destination = paths.models_base / spec.local_name
    destination.mkdir(parents=True, exist_ok=True)

    info = HfApi().model_info(spec.repo_id, revision=revision)
    resolved_revision = str(getattr(info, "sha", None) or revision or "unknown")
    local_path = Path(
        snapshot_download(
            repo_id=spec.repo_id,
            revision=resolved_revision if resolved_revision != "unknown" else revision,
            local_dir=destination,
        )
    )
    manifest = build_manifest(
        name=name,
        repo_id=spec.repo_id,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        local_path=local_path,
        license_name=_license_from_info(info),
    )
    write_manifest_atomic(manifest)
    return manifest


def download_models(names: Iterable[str], paths: LabPaths) -> list[ModelManifest]:
    return [download_model(name, paths) for name in names]


def download_all(paths: LabPaths) -> list[ModelManifest]:
    return download_models(MODEL_CATALOG.keys(), paths)
