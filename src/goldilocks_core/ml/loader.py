"""Convenience loaders for goldilocks ML models.

All public functions return None on any failure; callers fall back to heuristic.
Model search order is documented per function.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from goldilocks_core.ml.kpoints.predictor import KSpacingPredictor
    from goldilocks_core.ml.magnetic import MagneticClassifier


def try_load_kpoints_predictor(
    confidence_level: float = 0.95,
) -> "KSpacingPredictor | None":
    """Load the k-spacing predictor (CGCNN + QRF from HuggingFace).

    Returns None (→ heuristic fallback) if required files are missing or
    dependencies (dscribe, torch_geometric) are not installed.
    Downloads the QRF model from HuggingFace on first use.
    """
    try:
        from goldilocks_core.ml.kpoints.predictor import KSpacingPredictor

        return KSpacingPredictor(confidence_level=confidence_level)
    except Exception:
        return None


def try_load_magnetic_classifier(
    backbone_path: Path | str | None = None,
    device: str = "cpu",
) -> "MagneticClassifier | None":
    """Load the magnetic property classifier.

    Returns None (→ heuristic fallback) when the backbone is unavailable or
    the optional ``goldilocks[magnetic]`` dependencies are not installed.

    Backbone search order:
    1. *backbone_path* argument
    2. ``GOLDILOCKS_MACE_BACKBONE`` environment variable
    3. ``~/.goldilocks/models/mace_matpes_pbe_baseline_run-3.model``
    """
    try:
        if backbone_path is None:
            backbone_path = os.environ.get("GOLDILOCKS_MACE_BACKBONE")
        if backbone_path is None:
            default = (
                Path.home() / ".goldilocks" / "models"
                / "mace_matpes_pbe_baseline_run-3.model"
            )
            if default.exists():
                backbone_path = default
        if backbone_path is None:
            return None

        backbone_path = Path(backbone_path)
        if not backbone_path.exists():
            return None

        from goldilocks_core.ml.magnetic import MagneticClassifier

        return MagneticClassifier(backbone_path=backbone_path, device=device)
    except Exception:
        return None
