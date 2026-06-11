"""Tests for advise/vdw.py — VdwDecision logic."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.advise.vdw import advise_vdw
from goldilocks_core.analyse.structure import StructureAnalysis
from goldilocks_core.intent import CalculationIntent, ParameterHints

_STRUCTURE = Structure(
    lattice=Lattice.cubic(2.87),
    species=["Fe"],
    coords=[[0, 0, 0]],
)


def _intent(hints: dict[str, Any] | None = None) -> CalculationIntent:
    return CalculationIntent(structure=_STRUCTURE, hints=ParameterHints.from_dict(hints or {}))


_BASE_3D_ANALYSIS = StructureAnalysis(
    elements=["Fe"],
    n_atoms=1,
    n_species=1,
    contains_transition_metals=True,
    contains_lanthanides=False,
    contains_actinides=False,
    contains_heavy_elements=False,
    heavy_elements=[],
    space_group_number=229,
    space_group_symbol="Im-3m",
    crystal_system="cubic",
    point_group="m-3m",
    has_inversion_symmetry=True,
    n_symmetry_operations=96,
    metallicity="likely_metallic",
    metallicity_source="heuristic",
    metallicity_confidence=None,
    has_d_electrons=True,
    has_f_electrons=False,
    total_electrons=26,
    magnetic_prediction=None,
    magnetic_confidence=None,
    magnetic_source=None,
    magnetic_elements=["Fe"],
    soc_relevant=False,
    pbc=(True, True, True),
    dimensionality="3d",
    system_type="bulk",
    has_vacuum=False,
    is_slab=False,
    is_primitive=True,
    is_noncentrosymmetric=False,
    is_polar=False,
    has_partial_occupancy=False,
    disordered_sites=[],
    warnings=[],
)


def _analysis(**overrides: Any) -> StructureAnalysis:
    return dataclasses.replace(_BASE_3D_ANALYSIS, **overrides)


# ---------------------------------------------------------------------------
# Heuristic path
# ---------------------------------------------------------------------------

def test_3d_bulk_no_vdw() -> None:
    dec = advise_vdw(_BASE_3D_ANALYSIS, _intent())
    assert dec.use_vdw is False
    assert dec.method is None
    assert dec.provenance == "heuristic"


def test_2d_slab_gets_d3bj() -> None:
    ana = _analysis(
        dimensionality="2d",
        system_type="slab",
        is_slab=True,
        pbc=(True, True, False),
    )
    dec = advise_vdw(ana, _intent())
    assert dec.use_vdw is True
    assert dec.method == "d3bj"
    assert dec.provenance == "heuristic"


def test_has_vacuum_triggers_vdw() -> None:
    ana = _analysis(has_vacuum=True)
    dec = advise_vdw(ana, _intent())
    assert dec.use_vdw is True
    assert dec.method == "d3bj"


# ---------------------------------------------------------------------------
# User hint overrides
# ---------------------------------------------------------------------------

def test_hint_use_vdw_false_disables() -> None:
    ana = _analysis(dimensionality="2d", system_type="slab", is_slab=True)
    dec = advise_vdw(ana, _intent({"use_vdw": False}))
    assert dec.use_vdw is False
    assert dec.method is None
    assert dec.provenance == "user_hint"


def test_hint_use_vdw_true_on_bulk() -> None:
    dec = advise_vdw(_BASE_3D_ANALYSIS, _intent({"use_vdw": True}))
    assert dec.use_vdw is True
    assert dec.method == "d3bj"
    assert dec.provenance == "user_hint"


def test_hint_vdw_method_d3() -> None:
    dec = advise_vdw(_BASE_3D_ANALYSIS, _intent({"use_vdw": True, "vdw_method": "d3"}))
    assert dec.method == "d3"


def test_hint_vdw_method_ts() -> None:
    dec = advise_vdw(_BASE_3D_ANALYSIS, _intent({"use_vdw": True, "vdw_method": "ts"}))
    assert dec.method == "ts"


def test_hint_vdw_method_mbd() -> None:
    dec = advise_vdw(_BASE_3D_ANALYSIS, _intent({"use_vdw": True, "vdw_method": "mbd"}))
    assert dec.method == "mbd"


def test_hint_invalid_vdw_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown vdw_method"):
        advise_vdw(_BASE_3D_ANALYSIS, _intent({"use_vdw": True, "vdw_method": "lda-d3"}))
