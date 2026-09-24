from __future__ import annotations

from pymatgen.core import Lattice, Structure

from goldilocks_core.advisors.k_sampling import (
    KSamplingHumanInput,
    KSamplingLlmInput,
    k_sampling,
)
from goldilocks_core.analysis.is_metal import is_metal
from goldilocks_core.kmesh import build_gamma_kmesh_entries, k_distance_to_mesh
from goldilocks_core.resolution import Blocked, Provenance, Resolved, Unavailable

_IRON = Structure(Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]])
_SILICON = Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])


def test_metal_gets_the_denser_default_k_distance() -> None:
    state = k_sampling(is_metal(_IRON), _IRON)

    assert state.ok
    assert state.value.k_distance == 0.15
    assert state.value.mesh == k_distance_to_mesh(_IRON, 0.15)
    assert state.value.shift == (0, 0, 0)


def test_human_shift_applies_on_the_pure_heuristic_default_path() -> None:
    """Regression for #34 (v2 epic 9, #9): a human-supplied shift with no
    matching k_grid/k_distance override used to be silently dropped in
    the pure-heuristic branches -- K_POINTS was always Gamma-centered
    regardless of the requested shift."""
    state = k_sampling(
        is_metal(_IRON), _IRON, human=KSamplingHumanInput(shift=(1, 1, 1))
    )

    assert state.ok
    assert state.value.shift == (1, 1, 1)
    assert state.value.k_distance == 0.15


def test_confirmed_non_metal_gets_the_coarser_default_k_distance() -> None:
    state = k_sampling(Resolved("non_metal", Provenance(source="heuristic")), _SILICON)

    assert state.value.k_distance == 0.30
    assert state.value.mesh == k_distance_to_mesh(_SILICON, 0.30)


def test_unavailable_metallicity_leans_toward_the_metal_default() -> None:
    """Unlike occupations.py, uncertain metallicity here leans toward the
    denser (metal) k_distance -- an under-sampled undetected metal risks
    a silently wrong answer, an over-sampled insulator only costs more."""
    state = k_sampling(
        Unavailable(reason="composition alone does not confirm"), _SILICON
    )

    assert state.ok
    assert state.value.k_distance == 0.15


def test_blocked_metallicity_propagates_as_blocked() -> None:
    state = k_sampling(Blocked(by="synthetic failure"), _IRON)

    assert not state.ok
    assert state.status == "blocked"
    assert state.root_cause() == "synthetic failure"


def test_human_k_grid_wins_outright() -> None:
    state = k_sampling(
        is_metal(_IRON), _IRON, human=KSamplingHumanInput(k_grid=(4, 4, 4))
    )

    assert state.value.mesh == (4, 4, 4)
    assert state.value.k_distance is None
    assert state.source == "human"


def test_human_k_grid_with_a_shift() -> None:
    state = k_sampling(
        is_metal(_IRON),
        _IRON,
        human=KSamplingHumanInput(k_grid=(4, 4, 4), shift=(1, 1, 1)),
    )

    assert state.value.shift == (1, 1, 1)


def test_human_k_grid_and_k_distance_together_warns_and_grid_wins() -> None:
    state = k_sampling(
        is_metal(_IRON),
        _IRON,
        human=KSamplingHumanInput(k_grid=(4, 4, 4), k_distance=0.2),
    )

    assert state.value.mesh == (4, 4, 4)
    assert any("k_grid wins" in warning.message for warning in state.value.warnings)


def test_human_k_distance_without_a_grid_is_converted_to_a_mesh() -> None:
    state = k_sampling(
        is_metal(_IRON), _IRON, human=KSamplingHumanInput(k_distance=0.25)
    )

    assert state.value.mesh == k_distance_to_mesh(_IRON, 0.25)
    assert state.value.k_distance == 0.25
    assert state.source == "human"


def test_llm_k_distance_is_honored_when_no_human_override_exists() -> None:
    state = k_sampling(is_metal(_IRON), _IRON, llm=KSamplingLlmInput(k_distance=0.2))

    assert state.value.k_distance == 0.2
    assert state.source == "llm"


def test_ml_k_distance_used_when_qrf95_is_installed_and_no_override(
    real_assets,
) -> None:
    """v2 epic 11 (#11) follow-up, #92: QRF95's own PSDI record always
    carried a real model.json, it just was never registered as one of
    ``[defaults.kpoints]``'s asset files -- this exercises the actual,
    installed model, not a monkeypatched stand-in."""
    state = k_sampling(is_metal(_IRON), _IRON)

    assert state.ok
    assert state.source == "ml"
    assert state.value.k_distance is not None
    assert state.value.mesh == k_distance_to_mesh(_IRON, state.value.k_distance)
    assert state.value.shift == (0, 0, 0)


def test_ml_k_distance_respects_a_human_shift(real_assets) -> None:
    """Regression for the same class of bug as #34 (see the pure
    -heuristic-path regression above): the ml branch used to hardcode
    ``shift=None`` regardless of what the human asked for."""
    state = k_sampling(
        is_metal(_IRON), _IRON, human=KSamplingHumanInput(shift=(1, 1, 1))
    )

    assert state.source == "ml"
    assert state.value.shift == (1, 1, 1)


def test_human_k_distance_wins_over_ml(real_assets) -> None:
    state = k_sampling(
        is_metal(_IRON), _IRON, human=KSamplingHumanInput(k_distance=0.25)
    )

    assert state.source == "human"
    assert state.value.k_distance == 0.25


def test_human_k_index_resolves_to_that_rung_s_own_mesh() -> None:
    entries = build_gamma_kmesh_entries(_IRON)

    state = k_sampling(is_metal(_IRON), _IRON, human=KSamplingHumanInput(k_index=3))

    assert state.ok
    assert state.value.mesh == entries[2].mesh
    assert state.value.k_distance is None
    assert state.value.k_index == 3
    assert state.value.k_distance_interval == entries[2].k_distance_interval
    assert state.source == "human"


def test_heuristic_default_k_index_matches_the_ladder() -> None:
    """Not every resolved mesh lands on a rung -- this cross-checks
    whatever `k_index` the heuristic path attached against an
    independently built ladder, rather than assuming 0.15 A^-1 happens to
    land on one for this particular fixture (it may not)."""
    state = k_sampling(is_metal(_IRON), _IRON)

    entries = build_gamma_kmesh_entries(_IRON)
    expected = next(
        (entry.kindex for entry in entries if entry.mesh == state.value.mesh), None
    )
    assert state.value.k_index == expected


def test_human_k_distance_landing_on_a_rung_reverse_derives_k_index() -> None:
    entries = build_gamma_kmesh_entries(_IRON)
    third_rung = entries[2]
    low, high = third_rung.k_distance_interval
    midpoint = (low + high) / 2

    state = k_sampling(
        is_metal(_IRON), _IRON, human=KSamplingHumanInput(k_distance=midpoint)
    )

    assert state.ok
    assert state.value.mesh == third_rung.mesh
    assert state.value.k_index == 3
    assert state.value.k_distance_interval == third_rung.k_distance_interval


def test_human_k_grid_matching_a_ladder_rung_reverse_derives_k_index() -> None:
    entries = build_gamma_kmesh_entries(_IRON)
    third_rung_mesh = entries[2].mesh

    state = k_sampling(
        is_metal(_IRON), _IRON, human=KSamplingHumanInput(k_grid=third_rung_mesh)
    )

    assert state.ok
    assert state.value.k_index == 3
    assert state.value.k_distance_interval == entries[2].k_distance_interval


def test_human_k_grid_off_the_ladder_leaves_k_index_none() -> None:
    """Iron is cubic, so its ladder only ever contains (n, n, n) meshes --
    an anisotropic grid can never land on a rung."""
    state = k_sampling(
        is_metal(_IRON), _IRON, human=KSamplingHumanInput(k_grid=(1, 2, 1))
    )

    assert state.ok
    assert state.value.mesh == (1, 2, 1)
    assert state.value.k_index is None
    assert state.value.k_distance_interval is None


def test_human_k_index_with_a_shift() -> None:
    state = k_sampling(
        is_metal(_IRON),
        _IRON,
        human=KSamplingHumanInput(k_index=3, shift=(1, 1, 1)),
    )

    assert state.value.shift == (1, 1, 1)


def test_human_k_index_beyond_the_ladder_is_blocked() -> None:
    entries = build_gamma_kmesh_entries(_IRON)

    state = k_sampling(
        is_metal(_IRON),
        _IRON,
        human=KSamplingHumanInput(k_index=len(entries) + 1000),
    )

    assert not state.ok
    assert state.status == "blocked"
    assert "exceeds this structure's own ladder length" in state.root_cause()


def test_human_k_grid_wins_over_k_index_and_k_distance_together() -> None:
    state = k_sampling(
        is_metal(_IRON),
        _IRON,
        human=KSamplingHumanInput(k_grid=(4, 4, 4), k_index=3, k_distance=0.2),
    )

    assert state.value.mesh == (4, 4, 4)
    assert any("k_grid wins" in warning.message for warning in state.value.warnings)


def test_human_k_index_wins_over_k_distance_when_no_grid_is_set() -> None:
    entries = build_gamma_kmesh_entries(_IRON)

    state = k_sampling(
        is_metal(_IRON),
        _IRON,
        human=KSamplingHumanInput(k_index=3, k_distance=0.2),
    )

    assert state.value.mesh == entries[2].mesh
    assert any("k_index wins" in warning.message for warning in state.value.warnings)
