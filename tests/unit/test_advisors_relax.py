from __future__ import annotations

from goldilocks_core.advisors.convergence import convergence
from goldilocks_core.advisors.relax import (
    RelaxHumanInput,
    RelaxLlmInput,
    RelaxOptions,
    VcRelaxOptions,
    relax_settings,
)
from goldilocks_core.analysis.geometry import GeometryFacts
from goldilocks_core.analysis.symmetry import SymmetryFacts
from goldilocks_core.resolution import Blocked, Provenance, Resolved, Unavailable

_CONVERGENCE = convergence(nat=4)


def _symmetry(crystal_system: str):
    return Resolved(
        SymmetryFacts(
            space_group_symbol="P6/mmm",
            space_group_number=191,
            crystal_system=crystal_system,
        ),
        Provenance(source="heuristic"),
    )


def _geometry(dimensionality: str):
    return Resolved(
        GeometryFacts(
            dimensionality=dimensionality, low_dimensional=dimensionality != "3d"
        ),
        Provenance(source="heuristic"),
    )


def test_plain_relax_uses_qe_defaults() -> None:
    state = relax_settings("relax", _CONVERGENCE)

    assert state.ok
    assert type(state.value) is RelaxOptions
    assert state.value.ion_dynamics == "bfgs"
    assert state.value.forc_conv_thr == 1.0e-3
    assert state.value.nstep == 50
    assert state.value.trust_radius_max == 0.8
    assert state.value.trust_radius_min == 1.0e-3
    assert state.value.trust_radius_ini == 0.5
    assert state.value.remove_rigid_rot is False
    assert state.source == "heuristic"


def test_etot_conv_thr_is_inherited_from_convergence_not_recomputed() -> None:
    tight_convergence = convergence(nat=200)

    state = relax_settings("relax", tight_convergence)

    assert state.value.etot_conv_thr == tight_convergence.value.etot_conv_thr
    assert state.value.etot_conv_thr == 200 * 1e-5


def test_blocked_convergence_blocks_relax_settings() -> None:
    state = relax_settings("relax", Blocked(by="functional undecided"))

    assert isinstance(state, Blocked)
    assert state.root_cause() == "functional undecided"


def test_unavailable_convergence_blocks_relax_settings() -> None:
    state = relax_settings("relax", Unavailable(reason="synthetic"))

    assert isinstance(state, Blocked)
    assert state.root_cause() == "synthetic"


def test_vc_relax_returns_vc_relax_options_with_its_own_defaults() -> None:
    state = relax_settings("vc-relax", _CONVERGENCE)

    assert type(state.value) is VcRelaxOptions
    assert state.value.cell_dofree == "all"
    assert state.value.press == 0.0
    assert state.value.press_conv_thr == 0.5
    assert state.value.cell_factor == 2.0


def test_vc_relax_options_cell_dynamics_mirrors_ion_dynamics() -> None:
    state = relax_settings(
        "vc-relax", _CONVERGENCE, human=RelaxHumanInput(ion_dynamics="damp")
    )

    assert state.value.ion_dynamics == "damp"
    assert state.value.cell_dynamics == "damp"


def test_human_can_override_ion_dynamics_and_stays_human_sourced() -> None:
    state = relax_settings(
        "relax", _CONVERGENCE, human=RelaxHumanInput(ion_dynamics="fire")
    )

    assert state.value.ion_dynamics == "fire"
    assert state.source == "human"


def test_human_can_override_a_single_field_others_keep_their_default() -> None:
    state = relax_settings("relax", _CONVERGENCE, human=RelaxHumanInput(nstep=200))

    assert state.value.nstep == 200
    assert state.value.forc_conv_thr == 1.0e-3
    assert state.source == "human"


def test_llm_can_adjust_ion_dynamics_but_source_reflects_llm() -> None:
    state = relax_settings(
        "relax", _CONVERGENCE, llm=RelaxLlmInput(ion_dynamics="damp")
    )

    assert state.value.ion_dynamics == "damp"
    assert state.source == "llm"


def test_no_overrides_gives_heuristic_source() -> None:
    state = relax_settings("relax", _CONVERGENCE)

    assert state.source == "heuristic"


def test_hexagonal_2d_defaults_cell_dofree_to_ibrav_2dxy() -> None:
    state = relax_settings(
        "vc-relax",
        _CONVERGENCE,
        symmetry=_symmetry("hexagonal"),
        geometry=_geometry("2d"),
    )

    assert state.value.cell_dofree == "ibrav+2Dxy"
    assert state.value.warnings == ()


def test_non_hexagonal_2d_uses_bare_2dxy_with_a_warning() -> None:
    state = relax_settings(
        "vc-relax",
        _CONVERGENCE,
        symmetry=_symmetry("orthorhombic"),
        geometry=_geometry("2d"),
    )

    assert state.value.cell_dofree == "2Dxy"
    assert len(state.value.warnings) == 1
    assert state.value.warnings[0].code == "relax.cell_dofree_non_hexagonal_2d"


def test_non_2d_structure_keeps_cell_dofree_all_even_if_hexagonal() -> None:
    state = relax_settings(
        "vc-relax",
        _CONVERGENCE,
        symmetry=_symmetry("hexagonal"),
        geometry=_geometry("3d"),
    )

    assert state.value.cell_dofree == "all"
    assert state.value.warnings == ()


def test_missing_geometry_falls_back_to_cell_dofree_all() -> None:
    state = relax_settings("vc-relax", _CONVERGENCE)

    assert state.value.cell_dofree == "all"


def test_explicit_human_cell_dofree_skips_the_hexagonal_heuristic() -> None:
    state = relax_settings(
        "vc-relax",
        _CONVERGENCE,
        symmetry=_symmetry("hexagonal"),
        geometry=_geometry("2d"),
        human=RelaxHumanInput(cell_dofree="volume"),
    )

    assert state.value.cell_dofree == "volume"
    assert state.value.warnings == ()


def test_explicit_llm_cell_dofree_skips_the_hexagonal_heuristic() -> None:
    state = relax_settings(
        "vc-relax",
        _CONVERGENCE,
        symmetry=_symmetry("orthorhombic"),
        geometry=_geometry("2d"),
        llm=RelaxLlmInput(cell_dofree="shape"),
    )

    assert state.value.cell_dofree == "shape"
    assert state.value.warnings == ()
