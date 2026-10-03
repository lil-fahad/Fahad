from __future__ import annotations

from pathlib import Path
from typing import Any


def run_training_campaign(
    *,
    root: Path,
    manifest_path: Path,
    profile: str | None = None,
) -> dict[str, Any]:
    """Run the manifest-driven local heavy-training campaign.

    The full execution behavior is added behind tests in the next TDD step.
    """
    raise NotImplementedError("manifest-driven campaign execution is not implemented yet")
