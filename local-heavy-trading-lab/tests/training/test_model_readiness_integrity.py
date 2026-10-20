import json

from heavy_lab.models.manifest import validate_local_model_manifest


def test_manifest_rejects_missing_snapshot_file(tmp_path):
    root = tmp_path / "ttm"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({
        "name": "ttm",
        "resolved_revision": "abc123",
        "files": [{"path": "model.safetensors", "size_bytes": 1024}],
        "total_bytes": 1024
    }), encoding="utf-8")
    assert validate_local_model_manifest(root, "ttm") is False


def test_manifest_accepts_matching_local_files(tmp_path):
    root = tmp_path / "ttm"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "manifest.json").write_text(json.dumps({
        "name": "ttm",
        "resolved_revision": "abc123",
        "files": [{"path": "model.safetensors", "size_bytes": 7}],
        "total_bytes": 7
    }), encoding="utf-8")
    assert validate_local_model_manifest(root, "ttm") is True


def test_manifest_rejects_path_escape(tmp_path):
    root = tmp_path / "ttm"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({
        "name": "ttm",
        "resolved_revision": "abc123",
        "files": [{"path": "../outside.bin", "size_bytes": 5}],
        "total_bytes": 5
    }), encoding="utf-8")
    assert validate_local_model_manifest(root, "ttm") is False
