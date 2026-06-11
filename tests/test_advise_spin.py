"""Tests for advise/spin.py — SpinDecision logic."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.advise.spin import advise_spin
from goldilocks_core.analyse.structure import StructureAnalysis
from goldilocks_core.intent import CalculationIntent, ParameterHints


def _intent(hints: dict[str, Any] | None = None) -> CalculationIntent:
    structure = Structure(
        lattice=Lattice.cubic(2.87),
        species=["Fe"],
        coords=[[0, 0, 0]],
    )
    return CalculationIntent(structure=structure, hints=ParameterHints.from_dict(hints or {}))


_BASE_ANALYSIS = StructureAnalysis(
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
    return dataclasses.replace(_BASE_ANALYSIS, **overrides)


# ---------------------------------------------------------------------------
# Heuristic path
# ---------------------------------------------------------------------------

def test_no_magnetic_elements_gives_non_magnetic() -> None:
    ana = _analysis(magnetic_elements=[], elements=["Si"], contains_transition_metals=False)
    dec = advise_spin(ana, _intent())
    assert dec.treatment == "non_magnetic"
    assert dec.initial_magnetization is None
    assert dec.provenance == "heuristic"


def test_fe_collinear_no_soc() -> None:
    dec = advise_spin(_BASE_ANALYSIS, _intent())
    assert dec.treatment == "collinear"
    assert dec.initial_magnetization is not None
    assert "Fe" in dec.initial_magnetization
    assert dec.initial_magnetization["Fe"] > 0
    assert dec.provenance == "heuristic"


def test_fe_bi_soc_upgrades_to_noncolin_soc() -> None:
    ana = _analysis(
        elements=["Fe", "Bi"],
        magnetic_elements=["Fe"],
        soc_relevant=True,
        heavy_elements=["Bi"],
        contains_heavy_elements=True,
    )
    dec = advise_spin(ana, _intent())
    assert dec.treatment == "non_collinear_soc"
    assert dec.angle1 is not None
    assert dec.angle2 is not None
    assert dec.provenance == "heuristic"


# ---------------------------------------------------------------------------
# User hint overrides
# ---------------------------------------------------------------------------

def test_user_hint_forces_non_magnetic() -> None:
    dec = advise_spin(_BASE_ANALYSIS, _intent({"spin_treatment": "non_magnetic"}))
    assert dec.treatment == "non_magnetic"
    assert dec.initial_magnetization is None
    assert dec.provenance == "user_hint"


def test_user_hint_forces_collinear() -> None:
    ana = _analysis(magnetic_elements=[], elements=["Si"])
    dec = advise_spin(ana, _intent({"spin_treatment": "collinear"}))
    assert dec.treatment == "collinear"
    assert dec.provenance == "user_hint"


def test_user_hint_invalid_treatment_raises() -> None:
    with pytest.raises(ValueError, match="Unknown spin treatment"):
        advise_spin(_BASE_ANALYSIS, _intent({"spin_treatment": "ferromagnetic"}))


def test_user_hint_initial_magnetization_dict() -> None:
    dec = advise_spin(_BASE_ANALYSIS, _intent({"initial_magnetization": {"Fe": 2.0}}))
    assert dec.initial_magnetization is not None
    assert dec.initial_magnetization["Fe"] == pytest.approx(2.0)


def test_user_hint_initial_magnetization_string() -> None:
    dec = advise_spin(_BASE_ANALYSIS, _intent({"initial_magnetization": "Fe:2.5"}))
    assert dec.initial_magnetization is not None
    assert dec.initial_magnetization["Fe"] == pytest.approx(2.5)


def test_user_hint_initial_magnetization_multi_element_string() -> None:
    ana = _analysis(
        elements=["Fe", "Ni"],
        magnetic_elements=["Fe", "Ni"],
        n_species=2,
    )
    dec = advise_spin(
        ana, _intent({"initial_magnetization": "Fe:3.0,Ni:1.5"})
    )
    assert dec.initial_magnetization is not None
    assert dec.initial_magnetization["Fe"] == pytest.approx(3.0)
    assert dec.initial_magnetization["Ni"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# ML prediction path
# ---------------------------------------------------------------------------

def test_ml_collinear_respected() -> None:
    ana = _analysis(magnetic_prediction="collinear", magnetic_confidence=0.9, magnetic_source="ml")
    dec = advise_spin(ana, _intent())
    assert dec.treatment == "collinear"
    assert dec.provenance == "ML"


def test_ml_non_magnetic_respected() -> None:
    ana = _analysis(
        magnetic_prediction="non_magnetic",
        magnetic_confidence=0.85,
        magnetic_source="ml",
    )
    dec = advise_spin(ana, _intent())
    assert dec.treatment == "non_magnetic"
    assert dec.provenance == "ML"
