"""K-point recommendation utilities."""

from __future__ import annotations

from pymatgen.core import Structure

from goldilocks_core.helpers.types import AccuracyLevel, KPointsAdvice, ModelSpec


def advise_kpoints(
    structure: Structure,
    spec: ModelSpec,
    accuracy_level: AccuracyLevel = "standard",
) -> KPointsAdvice:
    """Advise k-point settings for a structure."""
    raise NotImplementedError("K-point advice is not implemented yet.")
