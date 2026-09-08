from pathlib import Path


def test_required_heavy_models_are_cataloged():
    from heavy_lab.models.catalog import MODEL_CATALOG

    assert MODEL_CATALOG["chronos2"].repo_id == "amazon/chronos-2"
    assert MODEL_CATALOG["timesfm25"].repo_id == "google/timesfm-2.5-200m-transformers"
    assert MODEL_CATALOG["kronos"].repo_id == "NeoQuasar/Kronos-base"
    assert MODEL_CATALOG["kronos_tokenizer"].repo_id == "NeoQuasar/Kronos-Tokenizer-base"
    assert MODEL_CATALOG["ttm"].repo_id == "ibm-granite/granite-timeseries-ttm-r2"
    assert MODEL_CATALOG["finbert"].repo_id == "ProsusAI/finbert"


def test_download_model_writes_secret_free_manifest(monkeypatch, tmp_path: Path):
    from heavy_lab.models import download as mod
    from heavy_lab.paths import LabPaths

    snapshot = tmp_path / "fake-snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model":"fake"}', encoding="utf-8")
    (snapshot / "weights.safetensors").write_bytes(b"weights")

    def fake_snapshot_download(**kwargs):
        assert kwargs["repo_id"] == "amazon/chronos-2"
        assert "token" not in kwargs
        return str(snapshot)

    class Info:
        sha = "abc123"
        cardData = {"license": "apache-2.0"}

    class FakeApi:
        def model_info(self, repo_id, revision=None):
            assert repo_id == "amazon/chronos-2"
            return Info()

    monkeypatch.setattr(mod, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(mod, "HfApi", FakeApi)

    paths = LabPaths.from_root(tmp_path / "lab")
    manifest = mod.download_model("chronos2", paths)

    assert manifest.repo_id == "amazon/chronos-2"
    assert manifest.resolved_revision == "abc123"
    assert manifest.license == "apache-2.0"
    assert manifest.total_bytes > 0
    assert all("token" not in item.path.lower() for item in manifest.files)
    assert Path(manifest.manifest_path).exists()
