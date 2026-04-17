"""Feature extraction utilities for machine-learning models."""

from __future__ import annotations

import numpy as np
from pymatgen.core import Structure

from goldilocks_core.helpers.types import StructureFeatureVector


def extract_c_features(structure: Structure) -> StructureFeatureVector:
    """Extract composition-based features from a structure."""
    raise NotImplementedError("C features are not implemented yet.")


def extract_s_features(structure: Structure) -> StructureFeatureVector:
    """Extract structure-based features from a structure."""
    raise NotImplementedError("S features are not implemented yet.")


def extract_l_features(structure: Structure) -> StructureFeatureVector:
    """Extract lattice-based features from a structure."""
    lattice = structure.lattice

    feature_names = [
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
        "volume",
    ]

    values = np.array(
        [
            lattice.a,
            lattice.b,
            lattice.c,
            lattice.alpha,
            lattice.beta,
            lattice.gamma,
            lattice.volume,
        ],
        dtype=float,
    )

    return StructureFeatureVector(
        values=values,
        feature_names=feature_names,
    )


def extract_r_features(structure: Structure) -> StructureFeatureVector:
    """Extract reciprocal-lattice-based features from a structure."""
    raise NotImplementedError("R features are not implemented yet.")


def extract_cslr_features(structure: Structure) -> StructureFeatureVector:
    """Extract CSLR-style features from a structure."""
    return extract_l_features(structure)
