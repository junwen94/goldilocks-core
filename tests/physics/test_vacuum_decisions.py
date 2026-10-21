"""Boundary conditions and cell constraints of crystals, slabs and
isolated systems (#126, #132).

Martyna-Tuckerman (MT) replaces the periodic Hartree, local-potential and
Ewald terms with those of an isolated cluster, and pw.x then computes no
stress. It suits only a system with vacuum along all three lattice
vectors. The bond rank cannot tell a molecular crystal from a molecule in
a box: CO2 dry ice ranks 'molecule', and H2O ice and trigonal Te rank '1d',
yet each fills its cell. On SCARF, MT on the CO2 SCF deck shifted the total
energy by -3.1 Ry per cell and flipped the sign of the O forces.

The rank cannot tell a bulk layered crystal from a slab either: graphite
ranks '2d' like graphene. QE's 2D cell constraint (cell_dofree='2Dxy')
holds c and the z components of a and b fixed. On SCARF, graphite under it
stopped after one BFGS step with c unchanged and 16 kbar of stress along z
left; with 'all', c shrank by 7.9%.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pymatgen.core import Lattice, Structure
from support import (
    carbon_chain,
    carbon_dioxide,
    carbon_pair_in_a_box,
    graphene_monolayer,
    graphene_with_vacuum_along_x,
    graphite,
    molybdenum_disulfide,
    phosphorene,
    rhombohedral_bismuth_selenide,
    sodium_chloride,
    water_ice,
)

from goldilocks_core.advisors.boundary import boundary
from goldilocks_core.advisors.convergence import convergence
from goldilocks_core.advisors.relax import relax_settings
from goldilocks_core.analysis.geometry import geometry
from goldilocks_core.resolution import Blocked


def trigonal_tellurium() -> Structure:
    """Te, P3_121, a = 4.457 A, c = 5.929 A, x = 0.2636: helical chains
    (Te-Te 2.84 A) 3.49 A apart, ranked '1d'."""
    return Structure.from_spacegroup(
        "P3_121", Lattice.hexagonal(4.457, 5.929), ["Te"], [[0.2636, 0, 1 / 3]]
    )


def _named(build: Callable[[], Structure]) -> str:
    return build.__name__


@pytest.mark.parametrize(
    "build",
    [
        carbon_dioxide,
        water_ice,
        trigonal_tellurium,
        sodium_chloride,
        phosphorene,
        carbon_chain,
    ],
    ids=_named,
)
def test_a_cell_periodic_along_any_axis_is_not_treated_as_isolated(
    build: Callable[[], Structure],
) -> None:
    """Molecular crystals, a chain crystal, an ionic solid, a monolayer and
    a wire: each is periodic along at least one lattice vector, while MT is
    a correction for a molecule or cluster in a 3D supercell (QE 7.3
    PW/Doc/INPUT_PW.def:1593-1594)."""
    state = boundary(geometry(build()))

    assert state.ok
    assert state.value.assume_isolated == "none"


def test_a_molecule_in_a_box_gets_martyna_tuckerman() -> None:
    state = boundary(geometry(carbon_pair_in_a_box()))

    assert state.value.assume_isolated == "martyna-tuckerman"


@pytest.mark.parametrize(
    ("build", "vacuum_axes"),
    [
        (carbon_dioxide, ()),
        (water_ice, ()),
        (trigonal_tellurium, ()),
        (sodium_chloride, ()),
        (phosphorene, ("c",)),
        (carbon_chain, ("a", "b")),
        (carbon_pair_in_a_box, ("a", "b", "c")),
    ],
    ids=lambda value: value.__name__ if callable(value) else None,
)
def test_vacuum_axes_are_the_lattice_vectors_with_an_empty_slab(
    build: Callable[[], Structure], vacuum_axes: tuple[str, ...]
) -> None:
    """Dense crystals leave no atom-free slab wider than 1.48 A (CO2) or
    2.82 A (NaCl); phosphorene has 20.07 A of vacuum along c."""
    assert geometry(build()).value.vacuum_axes == vacuum_axes


def test_the_carbon_box_vacuum_widths() -> None:
    """MC3D mc3d-6516: 6.1, 6.1 and 8.59 A between the two atoms' planes
    along a, b and c."""
    widths = geometry(carbon_pair_in_a_box()).value.vacuum_widths_angstrom

    assert widths == (6.1, 6.1, 8.59)


def _vc_relax(structure: Structure):
    return relax_settings(
        "vc-relax", convergence(nat=len(structure)), geometry=geometry(structure)
    )


@pytest.mark.parametrize(
    "build",
    [graphite, molybdenum_disulfide, rhombohedral_bismuth_selenide],
    ids=_named,
)
def test_vc_relax_of_a_bulk_layered_crystal_relaxes_the_whole_cell(
    build: Callable[[], Structure],
) -> None:
    """Each ranks '2d' but has no vacuum, so the stacking axis must relax
    too. Under '2Dxy' the rhombohedral Bi2Se3 cell, whose a and b point out
    of the xy plane, also distorted on SCARF: a and b shrank while c stayed
    fixed."""
    state = _vc_relax(build())

    assert state.ok
    assert state.value.cell_dofree == "all"
    assert state.value.warnings == ()


def test_vc_relax_of_a_monolayer_in_the_xy_plane_holds_its_vacuum_fixed() -> None:
    state = _vc_relax(graphene_monolayer())

    assert state.ok
    assert state.value.cell_dofree == "2Dxy"
    assert state.value.warnings == ()


def test_vc_relax_of_a_monolayer_with_its_vacuum_along_x_is_refused() -> None:
    """'2Dxy' would free the vacuum vector a and freeze c, which lies in the
    sheet."""
    state = _vc_relax(graphene_with_vacuum_along_x())

    assert isinstance(state, Blocked)
    assert "cell_dofree" in state.root_cause()
    assert "along a" in state.root_cause()
