"""Shared type definitions for goldilocks_core."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

PathLike = str | Path
StructureInput = Structure | PathLike


@dataclass(slots=True)
class StructureFeatureVector:
    """Named numerical feature vector extracted from a structure.

    Stores feature values together with their column names so feature
    ordering remains explicit during inference.
    """

    values: np.ndarray
    feature_names: list[str]


@dataclass(frozen=True)
class KMeshEntry:
    """One indexed k-mesh entry produced from a structure scan."""

    k_index: int
    mesh: tuple[int, int, int]
    k_distance_interval: tuple[float, float]
    k_line_density_interval: tuple[float, float] | None
    k_pra: float
    n_reduced_kpoints: int
