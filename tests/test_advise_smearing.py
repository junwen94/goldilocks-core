"""Tests for advise/smearing.py — SmearingDecision logic."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.advise.smearing import advise_smearing
from goldilocks_core.advise.types import Protocol
from goldilocks_core.analyse.structure import StructureAnalysis
from goldilocks_core.intent import CalculationIntent, ParameterHints

_PROTOCOL_BALANCED = Protocol(
    name="balanced",
    smearing_width_ry=0.01,
    smearing_width_ev=0.136,
    k_distance=0.3,
)

_STRUCTURE = Structure(
    lattice=Lattice.cubic(5.43),
    species=["Si", "Si"],
    coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
)


def _intent(hints: dict[str, Any] | None = None) -> CalculationIntent:
    return CalculationIntent(structure=_STRUCTURE, hints=ParameterHints.from_dict(hints or {}))


_BASE_ANALYSIS = StructureAnalysis(
    elements=["Si"],
    n_atoms=2,
    n_species=1,
    contains_transition_metals=False,
    contains_lanthanides=False,
    contains_actinides=False,
    contains_heavy_elements=False,
    heavy_elements=[],
    space_group_number=227,
    space_group_symbol="Fd-3m",
    crystal_system="cubic",
    point_group="m-3m",
    has_inversion_symmetry=True,
    n_symmetry_operations=192,
    metallicity="insulating",
    metallicity_source="heuristic",
    metallicity_confidence=None,
    has_d_electrons=False,
    has_f_electrons=False,
    total_electrons=28,
    magnetic_prediction=None,
    magnetic_confidence=None,
    magnetic_source=None,
    magnetic_elements=[],
    soc_relevant=False,
    pbc=(True, True, True),
    dimensionality="3d",
    system_type="bulk",
    has_vacuum=False,
    is_slab=False,
    is_primitive=False,
    is_noncentrosymmetric=False,
    is_polar=False,
    has_partial_occupancy=False,
    disordered_sites=[],
    warnings=[],
)


def _analysis(**overrides: Any) -> StructureAnalysis:
    return dataclasses.replace(_BASE_ANALYSIS, **overrides)


# ---------------------------------------------------------------------------
# Metallic path — smearing always on
# ---------------------------------------------------------------------------

def test_metallic_uses_smearing() -> None:
    ana = _analysis(metallicity="metallic")
    dec = advise_smearing(ana, _intent(), _PROTOCOL_BALANCED)
    assert dec.use_smearing is True
    assert dec.method == "marzari_vanderbilt"
    assert dec.width_ev is not None and dec.width_ev > 0
    assert dec.provenance == "heuristic"


def test_likely_metallic_uses_smearing() -> None:
    ana = _analysis(metallicity="likely_metallic")
    dec = advise_smearing(ana, _intent(), _PROTOCOL_BALANCED)
    assert dec.use_smearing is True


def test_unknown_metallicity_uses_smearing() -> None:
    ana = _analysis(metallicity="unknown")
    dec = advise_smearing(ana, _intent(), _PROTOCOL_BALANCED)
    assert dec.use_smearing is True


# ---------------------------------------------------------------------------
# Insulating path — fixed occupations
# ---------------------------------------------------------------------------

def test_insulating_uses_fixed() -> None:
    dec = advise_smearing(_BASE_ANALYSIS, _intent(), _PROTOCOL_BALANCED)
    assert dec.use_smearing is False
    assert dec.method is None
    assert dec.width_ev is None
    assert dec.provenance == "heuristic"


def test_likely_insulating_uses_fixed() -> None:
    ana = _analysis(metallicity="likely_insulating")
    dec = advise_smearing(ana, _intent(), _PROTOCOL_BALANCED)
    assert dec.use_smearing is False


# ---------------------------------------------------------------------------
# Guardrail: metallic structures cannot be overridden to fixed
# ---------------------------------------------------------------------------

def test_metallic_hint_invalid_method_raises() -> None:
    ana = _analysis(metallicity="metallic")
    with pytest.raises(ValueError, match="Unknown smearing method"):
        advise_smearing(ana, _intent({"smearing_method": "not_a_method"}), _PROTOCOL_BALANCED)


# ---------------------------------------------------------------------------
# User hint overrides
# ---------------------------------------------------------------------------

def test_hint_smearing_method_on_insulator() -> None:
    dec = advise_smearing(
        _BASE_ANALYSIS,
        _intent({"smearing_method": "fermi_dirac"}),
        _PROTOCOL_BALANCED,
    )
    assert dec.use_smearing is True
    assert dec.method == "fermi_dirac"
    assert dec.provenance == "user_hint"


def test_hint_smearing_width_overrides_protocol() -> None:
    ana = _analysis(metallicity="metallic")
    dec = advise_smearing(ana, _intent({"smearing_width_ev": 0.05}), _PROTOCOL_BALANCED)
    assert dec.width_ev == pytest.approx(0.05)
