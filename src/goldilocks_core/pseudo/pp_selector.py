"""Pseudopotential selection utilities."""

from __future__ import annotations

from pymatgen.core import Structure

from goldilocks_core.pseudo.pp_metadata import PseudoMetadata
from goldilocks_core.pseudo.pp_policy import PseudoPolicy, apply_pseudo_policy
from goldilocks_core.pseudo.pp_registry import (
    filter_by_element,
    filter_by_functional,
    filter_by_pseudo_type,
    filter_by_relativistic,
)


def select_pseudos(
    metadata_list: list[PseudoMetadata],
    *,
    element: str | None = None,
    functional: str | None = None,
    pseudo_type: str | None = None,
    relativistic: str | None = None,
) -> list[PseudoMetadata]:
    """Select pseudopotentials matching optional filter criteria."""
    selected = metadata_list

    if element is not None:
        selected = filter_by_element(selected, element)
    if functional is not None:
        selected = filter_by_functional(selected, functional)
    if pseudo_type is not None:
        selected = filter_by_pseudo_type(selected, pseudo_type)
    if relativistic is not None:
        selected = filter_by_relativistic(selected, relativistic)

    return selected


def group_pseudos_by_element(
    structure: Structure,
    metadata_list: list[PseudoMetadata],
) -> dict[str, list[PseudoMetadata]]:
    """Group available pseudopotentials by element for a structure."""
    elements = sorted({site.specie.symbol for site in structure})

    return {element: filter_by_element(metadata_list, element) for element in elements}


def select_pp_candidates_for_structure(
    structure: Structure,
    metadata_list: list[PseudoMetadata],
    policy: PseudoPolicy,
) -> dict[str, list[PseudoMetadata]]:
    """Select candidate pseudopotentials for each element in a structure."""
    grouped = group_pseudos_by_element(structure, metadata_list)

    return {
        element: apply_pseudo_policy(candidates, policy)
        for element, candidates in grouped.items()
    }
