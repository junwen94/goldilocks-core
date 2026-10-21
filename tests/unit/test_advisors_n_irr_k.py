from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.advisors.k_sampling import KSamplingDecision
from goldilocks_core.advisors.n_irr_k import NIrrKHumanInput, n_irr_k
from goldilocks_core.resolution import Blocked, Provenance, Resolved, Unavailable

_DIAMOND_SILICON = Structure.from_spacegroup(
    "Fd-3m", Lattice.cubic(5.43), ["Si"], [[0.0, 0.0, 0.0]]
)
_GRAPHITE = Structure.from_spacegroup(
    "P6_3/mmc",
    Lattice.hexagonal(2.46, 6.71),
    ["C", "C"],
    [[0.0, 0.0, 0.25], [1 / 3, 2 / 3, 0.25]],
)
_RUTILE = Structure.from_spacegroup(
    "P4_2/mnm",
    Lattice.tetragonal(4.594, 2.959),
    ["Ti", "O"],
    [[0.0, 0.0, 0.0], [0.305, 0.305, 0.0]],
)


def _k_sampling(mesh: tuple[int, int, int]) -> Resolved[KSamplingDecision]:
    return Resolved(
        KSamplingDecision(mesh=mesh, shift=(0, 0, 0), k_distance=0.2),
        Provenance(source="heuristic"),
    )


@pytest.mark.parametrize(
    ("structure", "mesh", "irreducible"),
    [
        (_DIAMOND_SILICON, (4, 4, 4), 10),  # 64 points, cubic
        (_DIAMOND_SILICON, (5, 5, 5), 10),  # 125 points
        (_GRAPHITE, (14, 14, 5), 72),  # 980 points, hexagonal
        (_RUTILE, (7, 7, 10), 60),  # 490 points, tetragonal
    ],
)
def test_reduces_a_mesh_by_symmetry(
    structure: Structure, mesh: tuple[int, int, int], irreducible: int
) -> None:
    """Real crystals reduce by a large factor. A fallback to the full mesh
    size would report an ordinary-looking integer, which is why there is
    none. n_irr_k is the only place a mesh is reduced; the ladder carries
    no counts (#155)."""
    state = n_irr_k(structure, _k_sampling(mesh))

    assert state.ok
    assert state.value == irreducible


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
