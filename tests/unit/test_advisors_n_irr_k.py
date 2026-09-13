from __future__ import annotations

from pymatgen.core import Lattice, Structure

from goldilocks_core.advisors.k_sampling import KSamplingDecision
from goldilocks_core.advisors.n_irr_k import NIrrKHumanInput, n_irr_k
from goldilocks_core.resolution import Blocked, Provenance, Resolved, Unavailable

_DIAMOND_SILICON = Structure.from_spacegroup(
    "Fd-3m", Lattice.cubic(5.43), ["Si"], [[0.0, 0.0, 0.0]]
)


def _k_sampling(mesh: tuple[int, int, int]) -> Resolved[KSamplingDecision]:
    return Resolved(
        KSamplingDecision(mesh=mesh, shift=(0, 0, 0), k_distance=0.2),
        Provenance(source="heuristic"),
    )


def test_reduces_a_mesh_by_symmetry() -> None:
    """Cross-checked against kmesh.py's own ladder: diamond silicon at
    (4, 4, 4) reduces to 10 irreducible k-points
    (test_kmesh.py::test_n_reduced_kpoints_is_the_irreducible_count_not_the_full_mesh)."""
    state = n_irr_k(_DIAMOND_SILICON, _k_sampling((4, 4, 4)))

    assert state.ok
    assert state.value == 10


def test_gamma_only_mesh_has_exactly_one_irreducible_kpoint() -> None:
    state = n_irr_k(_DIAMOND_SILICON, _k_sampling((1, 1, 1)))

    assert state.value == 1


def test_nosym_returns_the_full_mesh_size_unreduced() -> None:
    state = n_irr_k(
        _DIAMOND_SILICON, _k_sampling((4, 4, 4)), human=NIrrKHumanInput(nosym=True)
    )

    assert state.value == 64
    assert state.source == "human"


def test_blocked_k_sampling_propagates_as_blocked() -> None:
    state = n_irr_k(_DIAMOND_SILICON, Blocked(by="synthetic failure"))

    assert not state.ok
    assert state.status == "blocked"
    assert state.root_cause() == "synthetic failure"


def test_unavailable_k_sampling_is_treated_as_blocked_not_a_default() -> None:
    """There is no safe made-up mesh to fall back on if k_sampling itself
    could not resolve -- this must block, not guess."""
    state = n_irr_k(_DIAMOND_SILICON, Unavailable(reason="synthetic failure"))

    assert not state.ok
    assert state.status == "blocked"
    assert state.root_cause() == "synthetic failure"
