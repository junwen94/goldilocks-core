"""pw.x's dense FFT grid against ph.x's test when a cell is strained (#215).

A vc-relax keeps a cell's rotations but changes its lengths, and its final
scf rebuilds pw.x's grid for the relaxed cell. The strains here keep every
rotation pw.x finds: isotropic ones of +-0.5, 1, 2 and 3%, and eight random
ones each of 1, 2 and 3%, averaged over the point group so that each
rotation maps the strained lattice onto itself. A slab is strained in its
plane only, as cell_dofree='2Dxy' relaxes it. The cells come from
``tests/support.py``, each at the ecutrho of its default pseudopotentials.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Structure
from support import (
    graphite,
    molybdenum_disulfide_monolayer,
    written_alpha_uranium,
    written_bcc_iron,
    written_boron_nitride_monolayer,
    written_gallium_arsenide,
    written_lanthanum_cuprate,
    written_silicon,
    written_sodium_chloride,
)

from goldilocks_core.analysis.geometry import vacuum_axes
from goldilocks_core.pw_fft import (
    dense_fft_grid,
    fragile_axis_pairs,
    grid_is_compatible,
    ph_fft_grid,
)
from goldilocks_core.pw_symmetry import PwRotations, pw_rotations

pytestmark = pytest.mark.physics

_ISOTROPIC = (-0.03, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.03)
_RANDOM_SIZES = (0.01, 0.02, 0.03)
_RANDOM_PER_SIZE = 8


_NOT_FRAGILE = [
    (written_silicon, 192.0),
    (written_gallium_arsenide, 384.0),
    (written_sodium_chloride, 384.0),
    (written_bcc_iron, 424.0),
    (graphite, 360.0),
    (written_boron_nitride_monolayer, 384.0),
    (molybdenum_disulfide_monolayer, 368.0),
]
_FRAGILE = [(written_alpha_uranium, 480.0), (written_lanthanum_cuprate, 720.0)]


def _name(value: object) -> str | None:
    return value.__name__ if callable(value) else None


def _in_plane(structure: Structure) -> np.ndarray:
    """The Cartesian projector a strain of ``structure`` is confined to: the
    xy plane for a slab, written with its vacuum along c and its plane in
    xy, the identity otherwise."""
    if vacuum_axes(structure) == ("c",):
        return np.diag([1.0, 1.0, 0.0])
    return np.eye(3)


def _strains(structure: Structure, rotations: PwRotations) -> list[np.ndarray]:
    """The Cartesian strain tensors of the module docstring, seeded."""
    lattice = structure.lattice.matrix
    cartesian = [lattice.T @ w @ np.linalg.inv(lattice.T) for w in rotations.rotations]
    plane = _in_plane(structure)
    strains = [size * plane for size in _ISOTROPIC]
    generator = np.random.default_rng(215)
    for size in _RANDOM_SIZES:
        for _ in range(_RANDOM_PER_SIZE):
            raw = generator.uniform(-1.0, 1.0, (3, 3))
            raw = plane @ (raw + raw.T) @ plane
            averaged = sum(r @ raw @ r.T for r in cartesian) / len(cartesian)
            strains.append(averaged * size / np.abs(np.linalg.eigvalsh(averaged)).max())
    return strains


def _strained(structure: Structure, strain: np.ndarray) -> Structure:
    """``structure`` with every lattice vector a turned into (1 + strain) a,
    its fractional coordinates kept."""
    return Structure(
        structure.lattice.matrix @ (np.eye(3) + strain).T,
        structure.species,
        structure.frac_coords,
    )


@pytest.mark.parametrize(("build", "ecutrho"), _NOT_FRAGILE, ids=_name)
def test_pw_x_s_grid_passes_on_every_strain_of_a_cell_that_is_not_fragile(
    build, ecutrho: float
) -> None:
    structure = build()
    rotations = pw_rotations(structure)
    assert fragile_axis_pairs(structure, rotations) == ()

    for strain in _strains(structure, rotations):
        strained = _strained(structure, strain)
        kept = pw_rotations(strained)

        assert len(kept) == len(rotations)
        assert grid_is_compatible(kept, dense_fft_grid(strained, ecutrho))


@pytest.mark.parametrize(("build", "ecutrho"), _FRAGILE, ids=_name)
def test_pw_x_s_grid_fails_on_every_isotropic_strain_of_a_fragile_cell(
    build, ecutrho: float
) -> None:
    structure = build()

    for size in _ISOTROPIC:
        strained = _strained(structure, size * np.eye(3))

        assert not grid_is_compatible(
            pw_rotations(strained), dense_fft_grid(strained, ecutrho)
        )


@pytest.mark.parametrize(("build", "ecutrho"), _NOT_FRAGILE + _FRAGILE, ids=_name)
def test_the_grid_to_write_passes_on_every_strain(build, ecutrho: float) -> None:
    structure = build()

    for strain in _strains(structure, pw_rotations(structure)):
        strained = _strained(structure, strain)
        kept = pw_rotations(strained)
        own = dense_fft_grid(strained, ecutrho)
        grid = ph_fft_grid(strained, kept, ecutrho)

        assert grid_is_compatible(kept, grid)
        assert all(length >= least for length, least in zip(grid, own, strict=True))
