from pathlib import Path

from heavy_lab.paths import LabPaths


def test_lab_paths_are_isolated(tmp_path: Path):
    p = LabPaths.from_root(tmp_path)
    assert p.models_base == tmp_path / "models" / "base"
    assert p.models_trained == tmp_path / "models" / "trained"
    assert p.checkpoints == tmp_path / "checkpoints"
    assert p.data_raw == tmp_path / "data" / "raw"
    assert p.data_canonical == tmp_path / "data" / "canonical"
    assert p.data_features == tmp_path / "data" / "features"
    assert p.data_splits == tmp_path / "data" / "splits"
    assert p.artifacts == tmp_path / "artifacts"
