"""Dimensionality-driven decisions through the real service pipeline.

``advise()``/``check()``/``generate()`` and their relax variants on Si
cells, with the synthetic SSSP Si table from ``tests/support.py``, so these
run without installed pseudopotentials (#126, #132). The dispersion tests
(#141) and the graphite vc-relax test (#132) read only the advice and the
check, which need no pseudopotentials either.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import SlabGenerator
from support import graphite, sssp_fixture_table_spec

from goldilocks_core.assets.pseudopotentials.importers import sssp_preparer
from goldilocks_core.assets.store import AssetStore
from goldilocks_core.inputs.hpc import load_hpc_profile
from goldilocks_core.service import (
    advise,
    advise_relax,
    check,
    check_relax,
    generate,
    generate_relax,
)
from goldilocks_core.set_overrides import build_overrides

pytestmark = pytest.mark.integration


@pytest.fixture
def store(tmp_path, monkeypatch) -> AssetStore:
    spec, registry_table = sssp_fixture_table_spec(tmp_path / "sources")
    monkeypatch.setattr(
        "goldilocks_core.service._pseudo.load_tables",
        lambda *_args, **_kwargs: {registry_table.id: registry_table},
    )
    store = AssetStore(tmp_path / "store")
    store.install(spec, sssp_preparer(registry_table))
    return store


def silicon_dimer_in_a_box() -> Structure:
    """An Si2 dimer (bond 2.25 A, along [111]) in a 12 A cubic box: 10.7 A
    of vacuum along every lattice vector."""
    offset = 2.25 / (12.0 * 3**0.5)
    return Structure(
        Lattice.cubic(12.0), ["Si", "Si"], [[0, 0, 0], [offset, offset, offset]]
    )


def diamond_silicon() -> Structure:
    return Structure.from_spacegroup(
        "Fd-3m", Lattice.cubic(5.431), ["Si"], [[0.0, 0.0, 0.0]]
    )


def primitive_diamond_silicon() -> Structure:
    """Diamond Si in its 2-atom fcc primitive cell, a = (0, 1/2, 1/2),
    b = (1/2, 0, 1/2) and c = (1/2, 1/2, 0) times 5.431 A: a and b point out
    of the xy plane."""
    half = 5.431 / 2
    return Structure(
        Lattice([[0, half, half], [half, 0, half], [half, half, 0]]),
        ["Si", "Si"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
    )


def silicon_001_slab() -> Structure:
    """A Si(001) slab from pymatgen's SlabGenerator: 8 atoms, 17.65 A of
    vacuum along c, and a, b in the xy plane."""
    slab = SlabGenerator(
        diamond_silicon(), (0, 0, 1), min_slab_size=8, min_vacuum_size=12
    ).get_slab()
    return Structure.from_sites(slab)


def rotated_about_x(structure: Structure, degrees: float) -> Structure:
    """``structure`` turned rigidly about the Cartesian x axis."""
    angle = np.radians(degrees)
    rotation = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )
    return Structure(
        Lattice(structure.lattice.matrix @ rotation.T),
        structure.species,
        structure.frac_coords,
    )


def test_an_isolated_molecule_gets_martyna_tuckerman_and_no_stress(
    store: AssetStore,
) -> None:
    advice = advise(
        silicon_dimer_in_a_box(), hpc=load_hpc_profile("scarf"), store=store
    )
    report = check(advice)
    assert report.ok, report.blocking

    content = generate(advice, report)[0].files["scf.in"]

    assert advice.system.boundary.value.assume_isolated == "martyna-tuckerman"
    assert "assume_isolated  = 'martyna-tuckerman'" in content
    assert "tstress" not in content


def test_vc_relax_of_an_isolated_molecule_is_refused(store: AssetStore) -> None:
    """Under MT, pw.x computes no stress, so a vc-relax would report the
    input cell as converged (#126)."""
    advice = advise_relax(
        silicon_dimer_in_a_box(),
        calculation="vc-relax",
        hpc=load_hpc_profile("scarf"),
        store=store,
    )

    report = check_relax(advice)

    assert len(report.blocking) == 1
    assert "assume_isolated='martyna-tuckerman'" in report.blocking[0]
    assert "task='relax'" in report.blocking[0]


def test_vc_relax_of_a_bulk_crystal_computes_the_stress(store: AssetStore) -> None:
    advice = advise_relax(
        diamond_silicon(),
        calculation="vc-relax",
        hpc=load_hpc_profile("scarf"),
        store=store,
    )
    report = check_relax(advice)
    assert report.ok, report.blocking

    content = generate_relax(advice, report)[0].files["vc-relax.in"]

    assert "tstress          = .true." in content
    assert "assume_isolated" not in content
    assert "cell_dofree      = 'all'" in content


def test_vc_relax_of_a_slab_holds_c_and_the_z_components_fixed(
    store: AssetStore,
) -> None:
    """Si(001) has its vacuum along c and a, b in the xy plane, the frame
    QE's 2D cell constraint assumes (#132)."""
    advice = advise_relax(
        silicon_001_slab(),
        calculation="vc-relax",
        hpc=load_hpc_profile("scarf"),
        store=store,
    )
    report = check_relax(advice)
    assert report.ok, report.blocking

    content = generate_relax(advice, report)[0].files["vc-relax.in"]

    assert "cell_dofree      = '2Dxy'" in content


def test_vc_relax_of_bulk_graphite_relaxes_the_stacking_axis() -> None:
    """#132: graphite ranks '2d' but has no vacuum. Under '2Dxy', pw.x kept
    c fixed and reported convergence with 16 kbar of stress along z."""
    advice = advise_relax(
        graphite(), calculation="vc-relax", hpc=load_hpc_profile("scarf")
    )

    assert advice.advice.analysis.geometry.value.dimensionality == "2d"
    assert advice.relax.value.cell_dofree == "all"
    assert advice.relax.value.warnings == ()


def test_vc_relax_of_a_slab_outside_the_xy_frame_is_refused(
    store: AssetStore,
) -> None:
    """The Si(001) slab turned about x keeps its vacuum along c, but b and
    the slab leave the xy plane, so '2Dxy' would distort it (#132).
    SlabGenerator writes its (111) and (110) slabs in such a frame."""
    advice = advise_relax(
        rotated_about_x(silicon_001_slab(), 30.0),
        calculation="vc-relax",
        hpc=load_hpc_profile("scarf"),
        store=store,
    )

    report = check_relax(advice)

    assert len(report.blocking) == 1
    assert "cell_dofree" in report.blocking[0]
    assert "xy plane" in report.blocking[0]


def test_a_2d_cell_constraint_on_a_tilted_bulk_cell_is_refused(
    store: AssetStore,
) -> None:
    """'2Dxy' would hold the z components of a and b fixed while their x
    and y components relax, distorting the cell as it distorted
    rhombohedral Bi2Se3 on SCARF (#132)."""
    advice = advise_relax(
        primitive_diamond_silicon(),
        calculation="vc-relax",
        hpc=load_hpc_profile("scarf"),
        overrides=build_overrides({"cell_dofree": "2Dxy"}),
        store=store,
    )

    report = check_relax(advice)

    assert advice.relax.value.cell_dofree == "2Dxy"
    assert len(report.blocking) == 1
    assert "cell_dofree='2Dxy'" in report.blocking[0]
    assert "xy plane" in report.blocking[0]


def test_lda_graphite_gets_no_d3_dispersion_and_a_warning() -> None:
    """#141: QE's DFT-D3 has no D3(BJ) parameters for LDA, so graphite keeps
    its default D3 only under a functional that has them."""
    hpc = load_hpc_profile("scarf")
    default = advise(graphite(), hpc=hpc)
    lda = advise(graphite(), hpc=hpc, overrides=build_overrides({"functional": "LDA"}))

    assert default.system.vdw.value.use_vdw is True
    assert lda.analysis.geometry.value.dimensionality == "2d"
    assert lda.system.vdw.value.use_vdw is False
    codes = {warning["code"] for warning in lda.warnings()}
    assert "vdw.d3bj_unavailable_for_functional" in codes
    assert not any("use_vdw" in reason for reason in check(lda).blocking)


def test_d3_dispersion_with_lda_is_refused_before_generation() -> None:
    """#141: pw.x would stop at startup with 'functional name unknown'."""
    advice = advise(
        graphite(),
        hpc=load_hpc_profile("scarf"),
        overrides=build_overrides({"functional": "LDA", "use_vdw": True}),
    )

    report = check(advice)

    assert advice.system.vdw.source == "human"
    dispersion = [reason for reason in report.blocking if "use_vdw=false" in reason]
    assert len(dispersion) == 1
    assert "'LDA'" in dispersion[0]
