"""Dimensionality-driven decisions through the real service pipeline.

``advise()``/``check()``/``generate()`` and their relax variants on Si
cells, with the synthetic SSSP Si table from ``tests/support.py``, so these
run without installed pseudopotentials (#126, #132, #161). The dispersion
tests (#141), the graphite vc-relax test (#132), the isolation gate on
graphite and a MoS2 monolayer (#161), and the slab-only advice and
fix_bottom_layers gate (#160) read only the advice and the check, which
need no pseudopotentials either.
"""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import SlabGenerator
from support import (
    graphite,
    janus_molybdenum_sulfide_selenide,
    molybdenum_disulfide_monolayer,
    rotated_about_x,
    sssp_fixture_table_spec,
)

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
from goldilocks_core.set_overrides import InvalidSetting, build_overrides

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


def silicon_001_slab(min_vacuum_size: float = 12) -> Structure:
    """A Si(001) slab from pymatgen's SlabGenerator: 8 atoms 9.5 A thick,
    a and b in the xy plane, and vacuum along c, 17.65 A by default and
    23.08 A for ``min_vacuum_size=18``."""
    slab = SlabGenerator(
        diamond_silicon(),
        (0, 0, 1),
        min_slab_size=8,
        min_vacuum_size=min_vacuum_size,
    ).get_slab()
    return Structure.from_sites(slab)


def test_an_isolated_molecule_gets_martyna_tuckerman_and_no_stress(
    store: AssetStore,
) -> None:
    advice = advise(
        silicon_dimer_in_a_box(), hpc=load_hpc_profile("scarf"), store=store
    )
    report = check(advice)
    assert report.ok, report.blocking

    content = generate(advice, report)[0].files["scf.in"]

    assert advice.system.boundary.value.isolation == "isolated"
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
    assert "isolation='isolated'" in report.blocking[0]
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


def test_a_slab_override_writes_the_2d_coulomb_truncation(store: AssetStore) -> None:
    """#161: isolation='slab' reaches pw.x as assume_isolated='2D'. The
    Si(001) slab with 23.08 A of vacuum meets QE's preconditions: vacuum
    only along c, a and b in the xy plane, c above 21 bohr, and vacuum more
    than 20 bohr wider than the 9.5 A slab."""
    advice = advise(
        silicon_001_slab(min_vacuum_size=18),
        hpc=load_hpc_profile("scarf"),
        overrides=build_overrides({"isolation": "slab"}),
        store=store,
    )
    report = check(advice)
    assert report.ok, report.blocking

    content = generate(advice, report)[0].files["scf.in"]

    assert advice.system.boundary.source == "human"
    assert "assume_isolated  = '2D'" in content
    assert "tstress          = .true." in content


def test_vc_relax_of_a_slab_under_the_2d_truncation_holds_c_fixed(
    store: AssetStore,
) -> None:
    """Under '2D' pw.x computes only the in-plane stress (QE 7.3
    PW/Doc/INPUT_PW.def:1677-1678), which the slab's default '2Dxy' relaxes."""
    advice = advise_relax(
        silicon_001_slab(min_vacuum_size=18),
        calculation="vc-relax",
        hpc=load_hpc_profile("scarf"),
        overrides=build_overrides({"isolation": "slab"}),
        store=store,
    )
    report = check_relax(advice)
    assert report.ok, report.blocking

    content = generate_relax(advice, report)[0].files["vc-relax.in"]

    assert "assume_isolated  = '2D'" in content
    assert "tstress          = .true." in content
    assert "cell_dofree      = '2Dxy'" in content


def test_vc_relax_of_a_slab_under_the_2d_truncation_refuses_cell_dofree_all(
    store: AssetStore,
) -> None:
    advice = advise_relax(
        silicon_001_slab(min_vacuum_size=18),
        calculation="vc-relax",
        hpc=load_hpc_profile("scarf"),
        overrides=build_overrides({"isolation": "slab", "cell_dofree": "all"}),
        store=store,
    )

    report = check_relax(advice)

    assert len(report.blocking) == 1
    assert "cell_dofree='all'" in report.blocking[0]
    assert "cell_dofree='2Dxy'" in report.blocking[0]
    assert "isolation='periodic'" in report.blocking[0]


def test_a_slab_override_on_a_thick_slab_in_a_short_cell_is_refused(
    store: AssetStore,
) -> None:
    """QE wants the cell taller than twice the slab's thickness including its
    electrons, the atomic thickness plus 10 bohr (QE 7.3
    PW/Doc/INPUT_PW.def:1668-1675). The 9.5 A Si(001) slab in a 27.16 A
    cell would truncate part of its own interaction."""
    advice = advise(
        silicon_001_slab(),
        hpc=load_hpc_profile("scarf"),
        overrides=build_overrides({"isolation": "slab"}),
        store=store,
    )

    report = check(advice)

    assert len(report.blocking) == 1
    assert "17.65 Å of vacuum around a slab 9.51 Å thick" in report.blocking[0]
    assert "isolation='periodic'" in report.blocking[0]


def test_a_slab_override_on_a_monolayer_passes_the_isolation_gate() -> None:
    """All 15 MC2D monolayers of the 2026-10-19 batch meet QE's '2D'
    preconditions; the MoS2 sheet in a 20 A cell stands for them."""
    advice = advise(
        molybdenum_disulfide_monolayer(),
        hpc=load_hpc_profile("scarf"),
        overrides=build_overrides({"isolation": "slab"}),
    )

    report = check(advice)

    assert advice.system.magnetic.ok
    assert advice.system.boundary.source == "human"
    assert advice.system.boundary.value.isolation == "slab"
    assert not any("isolation" in reason for reason in report.blocking)


def test_a_slab_override_on_bulk_graphite_is_refused() -> None:
    """Graphite ranks '2d' but has no vacuum: truncating its Coulomb
    interaction along c would cut real interlayer electrostatics."""
    advice = advise(
        graphite(),
        hpc=load_hpc_profile("scarf"),
        overrides=build_overrides({"isolation": "slab"}),
    )

    report = check(advice)

    isolation = [reason for reason in report.blocking if "isolation=" in reason]
    assert len(isolation) == 1
    assert "isolation='slab'" in isolation[0]
    assert "the cell to be written has no vacuum" in isolation[0]
    assert "Set isolation='periodic'" in isolation[0]


def _dipole_notices(advice) -> list[str]:
    return [
        warning["message"]
        for warning in advice.warnings()
        if warning["code"] == "boundary.dipole_correction_suggested"
    ]


def test_bulk_graphite_gets_no_slab_advice() -> None:
    """#160: graphite ranks '2d', which used to bring the slab dipole notice
    and local-TF mixing, both justified by a vacuum it does not have."""
    advice = advise(graphite(), hpc=load_hpc_profile("scarf"))

    assert advice.analysis.geometry.value.dimensionality == "2d"
    assert _dipole_notices(advice) == []
    assert advice.step.kpoints.convergence.value.mixing_mode == "plain"


def test_a_symmetric_monolayer_gets_no_dipole_notice() -> None:
    advice = advise(molybdenum_disulfide_monolayer(), hpc=load_hpc_profile("scarf"))

    assert _dipole_notices(advice) == []
    assert advice.step.kpoints.convergence.value.mixing_mode == "local-TF"


def test_a_polar_sheet_gets_a_dipole_notice_along_its_vacuum_axis() -> None:
    """The Janus MoSSe sheet has no operation reversing c: the notice reads
    the point group of the input structure and its vacuum."""
    advice = advise(janus_molybdenum_sulfide_selenide(), hpc=load_hpc_profile("scarf"))

    (message,) = _dipole_notices(advice)
    assert "edir = 3" in message
    assert advice.analysis.symmetry.value.reversible_axes == ()


def test_fix_bottom_layers_on_bulk_graphite_is_refused() -> None:
    """#160, the issue's reproduction: with fix_bottom_layers=1 the relax of
    bulk graphite passed the check, and relax.in froze its z = 1/4 sheet."""
    advice = advise_relax(
        graphite(),
        calculation="relax",
        hpc=load_hpc_profile("scarf"),
        overrides=build_overrides({"fix_bottom_layers": 1}),
    )

    report = check_relax(advice)

    layers = [reason for reason in report.blocking if "fix_bottom_layers" in reason]
    assert len(layers) == 1
    assert "requires a slab" in layers[0]
    assert "the cell to be written has no vacuum" in layers[0]


def test_a_relax_of_a_slab_fixes_its_bottom_layers(store: AssetStore) -> None:
    """The Si(001) slab passes the gate, and the writer freezes its two
    lowest atoms, one per atomic layer."""
    slab = silicon_001_slab()
    advice = advise_relax(
        slab,
        calculation="relax",
        hpc=load_hpc_profile("scarf"),
        overrides=build_overrides({"fix_bottom_layers": 2}),
        store=store,
    )
    report = check_relax(advice)
    assert report.ok, report.blocking

    content = generate_relax(advice, report)[0].files["relax.in"]

    lines = content.splitlines()
    start = lines.index("ATOMIC_POSITIONS crystal") + 1
    rows = lines[start : start + len(slab)]
    fixed = {i for i, row in enumerate(rows) if row.split()[-3:] == ["0", "0", "0"]}
    z = slab.cart_coords[:, 2]
    assert fixed == {int(i) for i in z.argsort()[:2]}


def test_the_quantum_espresso_assume_isolated_key_is_not_a_setting() -> None:
    with pytest.raises(InvalidSetting, match="unknown setting 'assume_isolated'"):
        build_overrides({"assume_isolated": "none"})


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
