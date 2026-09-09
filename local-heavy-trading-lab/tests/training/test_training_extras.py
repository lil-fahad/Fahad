from pathlib import Path
import tomllib


def _optional_dependencies() -> dict[str, list[str]]:
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    return data["project"]["optional-dependencies"]


def test_finbert_training_extra_installs_transformers_stack():
    extras = _optional_dependencies()
    assert "finbert-train" in extras
    requirements = extras["finbert-train"]
    assert any(item.startswith("transformers") for item in requirements)
    assert any(item.startswith("accelerate") for item in requirements)
