from __future__ import annotations

from goldilocks_core.advisors.convergence import (
    ConvergenceHumanInput,
    ConvergenceLlmInput,
    convergence,
)
from goldilocks_core.analysis.geometry import GeometryFacts
from goldilocks_core.resolution import Provenance, Resolved, Unavailable


def test_conv_thr_scales_with_system_size() -> None:
    state = convergence(nat=10)

    assert state.ok
    assert state.value.conv_thr == 10 * 0.2e-9
    assert state.value.etot_conv_thr == 10 * 1e-5


def test_defaults_match_v1s_flat_constants() -> None:
    state = convergence(nat=4)

    assert state.value.mixing_beta == 0.4
    assert state.value.electron_maxstep == 80


def test_2d_geometry_switches_mixing_mode_to_local_tf() -> None:
    geometry = Resolved(
        GeometryFacts(dimensionality="2d", low_dimensional=True),
        Provenance(source="heuristic"),
    )

    state = convergence(nat=6, geometry=geometry)

    assert state.value.mixing_mode == "local-TF"


def test_molecule_geometry_also_switches_mixing_mode() -> None:
    geometry = Resolved(
        GeometryFacts(dimensionality="molecule", low_dimensional=True),
        Provenance(source="heuristic"),
    )

    state = convergence(nat=3, geometry=geometry)

    assert state.value.mixing_mode == "local-TF"


def test_1d_geometry_does_not_switch_mixing_mode() -> None:
    """F26 names only 2d/molecule; 1d is deliberately not extended to."""
    geometry = Resolved(
        GeometryFacts(dimensionality="1d", low_dimensional=True),
        Provenance(source="heuristic"),
    )

    state = convergence(nat=6, geometry=geometry)

    assert state.value.mixing_mode == "plain"


def test_unavailable_geometry_keeps_plain_mixing() -> None:
    state = convergence(nat=6, geometry=Unavailable(reason="synthetic failure"))

    assert state.value.mixing_mode == "plain"


def test_active_hubbard_sets_mixing_fixed_ns() -> None:
    needs_correlation = Resolved(True, Provenance(source="heuristic"))

    state = convergence(nat=5, needs_correlation=needs_correlation)

    assert state.value.mixing_fixed_ns == 50


def test_inactive_hubbard_leaves_mixing_fixed_ns_unset() -> None:
    needs_correlation = Resolved(False, Provenance(source="heuristic"))

    state = convergence(nat=5, needs_correlation=needs_correlation)

    assert state.value.mixing_fixed_ns is None


def test_unavailable_needs_correlation_does_not_block_or_guess() -> None:
    state = convergence(nat=5, needs_correlation=Unavailable(reason="synthetic"))

    assert state.ok
    assert state.value.mixing_fixed_ns is None


def test_human_conv_thr_override_is_used_directly_not_scaled() -> None:
    state = convergence(nat=100, human=ConvergenceHumanInput(conv_thr=1e-8))

    assert state.value.conv_thr == 1e-8
    assert state.source == "human"


def test_human_can_override_a_single_field_others_keep_their_default() -> None:
    state = convergence(nat=10, human=ConvergenceHumanInput(mixing_beta=0.2))

    assert state.value.mixing_beta == 0.2
    assert state.value.conv_thr == 10 * 0.2e-9
    assert state.source == "human"


def test_llm_can_adjust_mixing_knobs_but_not_conv_thr() -> None:
    state = convergence(
        nat=10, llm=ConvergenceLlmInput(mixing_fixed_ns=80, electron_maxstep=120)
    )

    assert state.value.mixing_fixed_ns == 80
    assert state.value.electron_maxstep == 120
    assert state.value.conv_thr == 10 * 0.2e-9
    assert state.source == "llm"


def test_no_overrides_gives_heuristic_source() -> None:
    state = convergence(nat=10)

    assert state.source == "heuristic"
