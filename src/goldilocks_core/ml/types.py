"""Types for goldilocks_core.ml."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class StructureFeatureVector:
    """Named numerical feature vector extracted from a structure.

    Stores feature values together with their column names so feature
    ordering remains explicit during inference.
    """

    values: np.ndarray
    feature_names: list[str]
