"""Tests for aiida/convert.py — pure Python conversion (no aiida-core needed)."""

from __future__ import annotations

from pymatgen.core import Lattice, Structure

from goldilocks_core.advise.pipeline import advise
from goldilocks_core.aiida.convert import (
    kpoints_to_mesh_dict,
    pseudos_to_upf_dict,
    qe_params_to_input_dict,
)
from goldilocks_core.analyse.structure import analyze_structure
from goldilocks_core.intent import CalculationIntent
from goldilocks_core.select.qe import build_qe_parameter_set

_FE_STRUCTURE = Structure(
    lattice=Lattice.cubic(2.87), species=["Fe"], coords=[[0, 0, 0]]
)
_SI_STRUCTURE = Structure(
    lattice=Lattice.cubic(5.43),
    species=["Si", "Si"],
    coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
)


def _make_params(structure, task="scf", accuracy="balanced"):
    analysis = analyze_structure(structure)
    intent = CalculationIntent(
        structure=structure, task=task, accuracy=accuracy,
        pseudo_family="PseudoDojo/0.4/PBEsol/SR/standard/upf",
    )
    return build_qe_parameter_set(advise(analysis, intent)), intent


class TestQeParamsToInputDict:
    def test_control_has_calculation_key(self) -> None:
        params, intent = _make_params(_SI_STRUCTURE)
        d = qe_params_to_input_dict(params, intent)
        assert d["CONTROL"]["calculation"] == "scf"

    def test_system_has_ecutwfc(self) -> None:
        params, intent = _make_params(_SI_STRUCTURE)
        d = qe_params_to_input_dict(params, intent)
        assert "ecutwfc" in d["SYSTEM"]
        assert d["SYSTEM"]["ecutwfc"] > 0

    def test_system_has_ecutrho(self) -> None:
        params, intent = _make_params(_SI_STRUCTURE)
        d = qe_params_to_input_dict(params, intent)
        assert d["SYSTEM"]["ecutrho"] >= d["SYSTEM"]["ecutwfc"]

    def test_metallic_has_smearing(self) -> None:
        params, intent = _make_params(_FE_STRUCTURE)
        d = qe_params_to_input_dict(params, intent)
        assert d["SYSTEM"]["occupations"] == "smearing"
        assert "degauss" in d["SYSTEM"]

    def test_si_has_occupations_key(self) -> None:
        # Si diamond gets metallicity="unknown" from the heuristic → smearing (safe default)
        params, intent = _make_params(_SI_STRUCTURE)
        d = qe_params_to_input_dict(params, intent)
        assert "occupations" in d["SYSTEM"]

    def test_fe_has_nspin2(self) -> None:
        params, intent = _make_params(_FE_STRUCTURE)
        d = qe_params_to_input_dict(params, intent)
        assert d["SYSTEM"].get("nspin") == 2

    def test_electrons_namelist_present(self) -> None:
        params, intent = _make_params(_SI_STRUCTURE)
        d = qe_params_to_input_dict(params, intent)
        assert "ELECTRONS" in d

    def test_relax_task(self) -> None:
        params, intent = _make_params(_SI_STRUCTURE, task="relax")
        d = qe_params_to_input_dict(params, intent)
        assert d["CONTROL"]["calculation"] == "relax"


class TestKpointsToMeshDict:
    def test_returns_mesh_and_offset(self) -> None:
        params, _ = _make_params(_SI_STRUCTURE)
        d = kpoints_to_mesh_dict(params)
        assert "mesh" in d and "offset" in d
        assert len(d["mesh"]) == 3
        assert len(d["offset"]) == 3

    def test_mesh_values_positive(self) -> None:
        params, _ = _make_params(_SI_STRUCTURE)
        d = kpoints_to_mesh_dict(params)
        assert all(v > 0 for v in d["mesh"])


class TestPseudosToUpfDict:
    def test_returns_element_to_filename_map(self) -> None:
        params, _ = _make_params(_FE_STRUCTURE)
        d = pseudos_to_upf_dict(params)
        assert "Fe" in d
        assert isinstance(d["Fe"], str)
