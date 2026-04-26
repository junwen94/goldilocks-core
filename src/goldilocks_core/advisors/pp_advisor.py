"""Pseudopotential recommendation utilities."""

from __future__ import annotations

from pymatgen.core import Structure

from goldilocks_core.pseudo.pp_metadata import PseudoMetadata
from goldilocks_core.pseudo.pp_policy import PseudoPolicy


def advise_pseudos(
    structure: Structure,
    metadata_list: list[PseudoMetadata],
    policy: PseudoPolicy,
) -> dict[str, list[PseudoMetadata]]:
    """Advise pseudopotential candidates for each element in a structure.

    Parameters
    ----------
    structure
        Structure whose elements need pseudopotential candidates.
    metadata_list
        Parsed pseudopotential metadata available for selection.
    policy
        Policy constraints used to filter allowed pseudopotentials.

    Returns
    -------
    dict[str, list[PseudoMetadata]]
        Candidate pseudopotentials grouped by element symbol.
    """
    raise NotImplementedError
