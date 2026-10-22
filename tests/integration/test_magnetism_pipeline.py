"""The magnetic decision is advised, checked and written as one input.

``magnetic_ordering=afm`` can replace the uploaded cell with a different
one: rock-salt NiO's 2-atom primitive cell and its 8-atom conventional cell
both become the 4-atom AFM-II cell. Every count the input carries -- the
electron count, ``nbnd``, ``conv_thr`` and the job's memory estimate --
must then describe that written cell (#128). They used to be counted on
the uploaded cell while the writer rendered the AFM one, and the writer
turned the electron difference into ``tot_charge = 24.0`` (2-atom input)
or ``-48.0`` (8-atom input).

The control is the written cell passed in directly, with its spins removed
(spin-decorated species never reach the analysis tier) and the AFM run's
own seeds as ``starting_magnetization``, so the magnetic step splits it
into the same species (#170). Runs on a synthetic SSSP-shaped Ni/O table
(z_valence 18 and 6) with the real enumlib.

Spin-orbit coupling makes a run noncollinear, where pw.x has no
``tot_magnetization`` and fixed occupations need none (#146). The CrI3
monolayer (MC2D mc2d-da9644c9) runs on a synthetic fully relativistic Cr/I
table (z_valence 14 and 7) and needs no enumlib.

Olivine LiFePO4 (MC3D mc3d-52501) has 28 sites but only 4 Fe for enumlib to
order, so ``magnetic_ordering=afm`` orders them (#171); it runs on a
synthetic Li/Fe/P/O table (z_valence 3, 16, 5 and 6) with the real enumlib.

A spin-polarized non-metal used to get fixed occupations without the
integer ``tot_magnetization`` pw.x needs for them on nspin=2, so the gate
refused the default for LiFePO4 and for NiO classified non-metal (#138).
Such a run now gets smearing, unless the applied ordering is compensated
by symmetry: then fixed occupations pin ``tot_magnetization = 0``.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import pytest
from pymatgen.core import Lattice, Structure
from support import sssp_fixture_table_spec

from goldilocks_core.advisors.magnetic_config import OrderingRecord
from goldilocks_core.advisors.magnetic_enumeration import enumlib_installed
from goldilocks_core.assets.pseudopotentials.importers import sssp_preparer
from goldilocks_core.assets.store import AssetStore
from goldilocks_core.inputs.hpc import Hardware, HpcProfile, Partition
from goldilocks_core.service import (
    Advice,
    AdviceIncomplete,
    advise,
    advise_dos,
    advise_relax,
    check,
    check_dos,
    check_relax,
    generate,
    generate_dos,
    generate_relax,
)
from goldilocks_core.service._magnetic_orderings import list_magnetic_orderings
from goldilocks_core.set_overrides import build_overrides

_needs_enumlib = pytest.mark.skipif(
    not enumlib_installed(),
    reason="needs the enumlib executables (enum.x, makeStr.py) on PATH",
)

_NEUTRAL_AFM_CELL_NELEC = 2 * 18.0 + 2 * 6.0


def _conventional_nio() -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(4.17), ["Ni", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def _primitive_nio() -> Structure:
    return _conventional_nio().get_primitive_structure()


def _hpc() -> HpcProfile:
    hardware = Hardware(
        cores_per_node=64, mem_per_node_gb=250, max_nodes=100, max_walltime_h=168
    )
    return HpcProfile(
        name="test-cluster",
        scheduler="slurm",
        launcher="srun",
        modules={"quantum_espresso": ("qe/7.2",)},
        partitions={"scarf": Partition(name="scarf", hardware=hardware, default=True)},
    )


@pytest.fixture
def store(tmp_path, monkeypatch) -> AssetStore:
    spec, registry_table = sssp_fixture_table_spec(
        tmp_path / "sources", {"Ni": 18.0, "O": 6.0}
    )
    monkeypatch.setattr(
        "goldilocks_core.service._pseudo.load_tables",
        lambda *a, **kw: {registry_table.id: registry_table},
    )
    installed = AssetStore(tmp_path / "store")
    installed.install(spec, sssp_preparer(registry_table))
    return installed


def _run(
    task: str, structure: Structure, settings: dict[str, object], store: AssetStore
) -> tuple[tuple[Advice, ...], tuple[str, ...]]:
    """Every pw.x step's ``Advice`` and every pw.x input for ``task``,
    after asserting the gate passes."""
    overrides = build_overrides(settings)
    if task == "scf":
        advice = advise(structure, hpc=_hpc(), overrides=overrides, store=store)
        report, steps = check(advice), (advice,)
        rendered = generate(advice, report) if report.ok else ()
    elif task == "dos":
        dos = advise_dos(structure, hpc=_hpc(), overrides=overrides, store=store)
        report, steps = check_dos(dos), (dos.scf, dos.nscf)
        rendered = generate_dos(dos, report) if report.ok else ()
    else:
        relax = advise_relax(
            structure, calculation=task, hpc=_hpc(), overrides=overrides, store=store
        )
        report, steps = check_relax(relax), (relax.advice,)
        rendered = generate_relax(relax, report) if report.ok else ()
    assert report.ok, report.blocking
    inputs = tuple(
        content
        for step in rendered
        if step.executable == "pw.x"
        for content in step.files.values()
    )
    return steps, inputs


def _keyword(content: str, name: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*(\S+)\s*$", content, re.M)
    return None if match is None else match.group(1)


def _counts(advice: Advice) -> dict[str, object]:
    kpoints = advice.step.kpoints
    job = advice.step.resources.job.value
    return {
        "nelec": advice.system.electron_count.value.nelec,
        "nbnd": kpoints.nbnd.value.nbnd,
        "conv_thr": kpoints.convergence.value.conv_thr,
        "mesh": kpoints.k_sampling.value.mesh,
        "n_irr_k": kpoints.n_irr_k.value,
        "npool": advice.step.resources.parallelisation.value.npool,
        "job": (
            job.nodes,
            job.ntasks,
            job.memory_per_process_mb,
            job.memory_total_mb,
        ),
    }


_INPUT_CELLS: dict[str, Callable[[], Structure]] = {
    "2-atom primitive": _primitive_nio,
    "8-atom conventional": _conventional_nio,
}


@_needs_enumlib
@pytest.mark.parametrize("task", ["scf", "relax", "dos"])
@pytest.mark.parametrize("input_cell", list(_INPUT_CELLS))
def test_every_count_describes_the_written_afm_cell(
    input_cell: str, task: str, store: AssetStore
) -> None:
    afm_steps, afm_inputs = _run(
        task, _INPUT_CELLS[input_cell](), {"magnetic_ordering": "afm"}, store
    )

    assert afm_inputs
    for content in afm_inputs:
        assert _keyword(content, "nat") == "4"
        assert _keyword(content, "tot_charge") is None
    electron_count = afm_steps[0].system.electron_count.value
    assert electron_count.nelec == _NEUTRAL_AFM_CELL_NELEC
    assert electron_count.tot_charge == 0.0

    magnetic = afm_steps[0].system.magnetic.value
    control = magnetic.cell.copy()
    control.remove_spin()
    control_steps, _ = _run(
        task,
        control,
        {
            "spin_polarized": True,
            "starting_magnetization": magnetic.starting_magnetization,
        },
        store,
    )
    assert [_counts(step) for step in afm_steps] == [
        _counts(step) for step in control_steps
    ]


@_needs_enumlib
def test_nelec_for_the_uploaded_cell_is_refused_once_the_ordering_resizes_it(
    store: AssetStore,
) -> None:
    """A ``nelec`` counts the electrons of the cell the user uploaded; an
    ordering that writes a cell of a different size makes it describe a
    cell the user did not get, so the electron count blocks instead of
    being rendered as a charge."""
    advice = advise(
        _primitive_nio(),
        hpc=_hpc(),
        overrides=build_overrides({"magnetic_ordering": "afm", "nelec": 24.0}),
        store=store,
    )

    report = check(advice)

    assert not report.ok
    assert not advice.system.electron_count.ok
    reason = advice.system.electron_count.root_cause()
    assert "nelec" in reason
    assert "2-atom" in reason
    assert "4-atom" in reason
    assert reason in report.blocking


def _starting_magnetization_lines(content: str) -> list[str]:
    return [
        line.strip()
        for line in content.splitlines()
        if line.strip().startswith("starting_magnetization(")
    ]


@_needs_enumlib
def test_a_listed_candidate_and_plain_afm_write_the_same_seeds_and_symmetry(
    store: AssetStore,
) -> None:
    """#170: a /magnetic-orderings candidate fed to a run seeded Ni at the
    flat 0.1 with O at +0.1, its symmetry counted on a spin-less CIF
    (Fm-3m), while magnetic_ordering=afm seeded the same AFM-II cell at
    5/18 (R-3m). Both are now one magnetic step: the candidate is its
    label, the deck seeds the Ni sublattice from z_valence and leaves O at
    0, and symmetry_eff is the split cell's."""
    nio = _primitive_nio()
    afm_steps, (afm_deck,) = _run("scf", nio, {"magnetic_ordering": "afm"}, store)
    applied = afm_steps[0].system.magnetic.value.ordering.applied
    [candidate] = [
        candidate
        for candidate in list_magnetic_orderings(nio).candidates
        if candidate.label == applied
    ]
    listed_steps, (listed_deck,) = _run("scf", nio, dict(candidate.overrides), store)

    assert applied == "afm-1"
    assert candidate.overrides == {"magnetic_ordering": "afm-1"}
    assert _starting_magnetization_lines(afm_deck) == _starting_magnetization_lines(
        listed_deck
    )
    for steps in (afm_steps, listed_steps):
        magnetic = steps[0].system.magnetic.value
        assert magnetic.starting_magnetization == pytest.approx(
            {"Ni1": 5.0 / 18.0, "Ni2": -5.0 / 18.0, "O": 0.0}
        )
        assert steps[0].system.symmetry_eff.value.space_group_symbol == "R-3m"
    species = afm_deck.split("ATOMIC_SPECIES\n")[1].split("\n\n")[0]
    assert [line.split()[0] for line in species.splitlines()] == ["Ni1", "Ni2", "O"]
    assert float(_keyword(afm_deck, "starting_magnetization(3)")) == 0.0


def _cri3_monolayer() -> Structure:
    """MC2D's CrI3 monolayer (mc2d-da9644c9), layer group p-31m: a Cr
    honeycomb between two iodine planes, 18.1 A between layers."""
    return Structure.from_spacegroup(
        "P-31m",
        Lattice.hexagonal(6.971, 18.10),
        ["Cr", "I"],
        [[1 / 3, 2 / 3, 0.5], [0.3706, 0.0, 0.5842]],
    )


@pytest.fixture
def spin_orbit_store(tmp_path, monkeypatch) -> AssetStore:
    """A fully relativistic Cr/I table, the one a spin-orbit run selects."""
    spec, registry_table = sssp_fixture_table_spec(
        tmp_path / "sources", {"Cr": 14.0, "I": 7.0}, relativistic="full"
    )
    monkeypatch.setattr(
        "goldilocks_core.service._pseudo.load_tables",
        lambda *a, **kw: {registry_table.id: registry_table},
    )
    installed = AssetStore(tmp_path / "store")
    installed.install(spec, sssp_preparer(registry_table))
    return installed


_SPIN_ORBIT_FIXED = {"spin_orbit_coupling": True, "occupations": "fixed"}


def test_spin_orbit_with_fixed_occupations_writes_a_noncollinear_deck(
    spin_orbit_store: AssetStore,
) -> None:
    """#146: the gate used to demand an integer tot_magnetization, which
    the writer then refused. A noncollinear run needs none: pw.x stops on
    fixed occupations without one only with nspin=2 (QE 7.3
    PW/src/input.f90:764-781), and QE 7.5 on SCARF ran CrI3 with
    noncolin, lspinorb, fixed occupations and starting moments alone."""
    steps, (content,) = _run(
        "scf", _cri3_monolayer(), _SPIN_ORBIT_FIXED, spin_orbit_store
    )

    assert steps[0].system.magnetic.value.spin_mode == "noncollinear"
    assert _keyword(content, "noncolin") == ".true."
    assert _keyword(content, "lspinorb") == ".true."
    assert _keyword(content, "occupations") == "'fixed'"
    # Cr's 5 Bohr magneton seed over its z_valence: a magnetic deck
    assert float(_keyword(content, "starting_magnetization(1)")) == pytest.approx(
        5.0 / 14.0
    )
    assert _keyword(content, "nspin") is None
    assert _keyword(content, "tot_magnetization") is None


def test_spin_orbit_refuses_tot_magnetization_before_generation(
    spin_orbit_store: AssetStore,
) -> None:
    """#146: pw.x accepts tot_magnetization only with nspin=2
    (input.f90:761-762). The gate refuses it -- so /explain shows it --
    and generate() never reaches a writer."""
    advice = advise(
        _cri3_monolayer(),
        hpc=_hpc(),
        overrides=build_overrides({**_SPIN_ORBIT_FIXED, "tot_magnetization": 6.0}),
        store=spin_orbit_store,
    )

    report = check(advice)

    assert any("tot_magnetization" in reason for reason in report.blocking), (
        report.blocking
    )
    with pytest.raises(AdviceIncomplete):
        generate(advice, report)


def test_an_ignored_tot_magnetization_never_reaches_a_spin_orbit_deck(
    spin_orbit_store: AssetStore,
) -> None:
    """#146: an explicit spin_polarized=false ignores tot_magnetization
    with a warning, and the deck then carries none."""
    steps, (content,) = _run(
        "scf",
        _cri3_monolayer(),
        {**_SPIN_ORBIT_FIXED, "spin_polarized": False, "tot_magnetization": 6.0},
        spin_orbit_store,
    )

    codes = {warning["code"] for warning in steps[0].warnings()}
    assert "magnetic.magnetization_ignored_spin_polarized_false" in codes
    assert _keyword(content, "noncolin") == ".true."
    assert "starting_magnetization" not in content
    assert _keyword(content, "tot_magnetization") is None


def _olivine_lifepo4() -> Structure:
    """Olivine LiFePO4 (Pnma, MC3D mc3d-52501): 28 sites, 4 of them Fe2+."""
    return Structure.from_spacegroup(
        "Pnma",
        Lattice.orthorhombic(10.242, 5.968, 4.687),
        ["Li", "Fe", "P", "O", "O", "O"],
        [
            [0.5, 0.5, 0.5],
            [0.7803, 0.75, 0.4789],
            [0.5948, 0.75, 0.9226],
            [0.5952, 0.75, 0.2504],
            [0.955, 0.75, 0.7068],
            [0.6672, 0.5435, 0.7876],
        ],
    )


@pytest.fixture
def olivine_store(tmp_path, monkeypatch) -> AssetStore:
    spec, registry_table = sssp_fixture_table_spec(
        tmp_path / "sources", {"Li": 3.0, "Fe": 16.0, "P": 5.0, "O": 6.0}
    )
    monkeypatch.setattr(
        "goldilocks_core.service._pseudo.load_tables",
        lambda *a, **kw: {registry_table.id: registry_table},
    )
    installed = AssetStore(tmp_path / "store")
    installed.install(spec, sssp_preparer(registry_table))
    return installed


@_needs_enumlib
def test_afm_orders_the_four_fe_of_olivine_lifepo4(olivine_store: AssetStore) -> None:
    """#171: a ceiling on all 28 sites kept LiFePO4 ferromagnetic and listed
    only 'fm', though pymatgen enumerates its 4 Fe in about a second."""
    olivine = _olivine_lifepo4()

    listing = list_magnetic_orderings(olivine)
    advice = advise(
        olivine,
        hpc=_hpc(),
        overrides=build_overrides({"magnetic_ordering": "afm"}),
        store=olivine_store,
    )

    labels = [candidate.label for candidate in listing.candidates]
    assert labels[0] == "fm"
    assert len(labels) > 1
    assert listing.warnings == ()
    magnetic = advice.system.magnetic.value
    assert magnetic.ordering == OrderingRecord(
        requested="afm", applied="afm-1", compensated=True
    )
    assert len(magnetic.cell) == 28
    assert {site.label for site in magnetic.cell} == {"Li", "Fe1", "Fe2", "P", "O"}
    assert magnetic.starting_magnetization["Fe1"] == pytest.approx(5.0 / 16.0)
    assert magnetic.starting_magnetization["Fe2"] == pytest.approx(-5.0 / 16.0)


def _bcc_chromium() -> Structure:
    """bcc Cr (MC3D mc3d-49144), its 1-atom primitive cell."""
    return Structure.from_spacegroup(
        "Im-3m", Lattice.cubic(2.8145), ["Cr"], [[0, 0, 0]]
    ).get_primitive_structure()


@pytest.fixture
def chromium_store(tmp_path, monkeypatch) -> AssetStore:
    spec, registry_table = sssp_fixture_table_spec(tmp_path / "sources", {"Cr": 14.0})
    monkeypatch.setattr(
        "goldilocks_core.service._pseudo.load_tables",
        lambda *a, **kw: {registry_table.id: registry_table},
    )
    installed = AssetStore(tmp_path / "store")
    installed.install(spec, sssp_preparer(registry_table))
    return installed


@_needs_enumlib
def test_an_afm_request_writes_an_afm_deck_for_a_non_magnetic_classification(
    chromium_store: AssetStore,
) -> None:
    """#145: with the ML tier classifying bcc Cr non-magnetic,
    magnetic_ordering=afm was dropped and the deck was byte-identical to
    the default one, with no nspin. The request now turns spin polarization
    on and writes the ordered cell, neutral (#128)."""
    steps, (content,) = _run(
        "scf",
        _bcc_chromium(),
        {"is_magnetic": False, "magnetic_ordering": "afm"},
        chromium_store,
    )

    assert steps[0].analysis.is_magnetic.value == "non_magnetic"
    magnetic = steps[0].system.magnetic.value
    assert magnetic.ordering == OrderingRecord(
        requested="afm", applied="afm-1", compensated=True
    )
    assert _keyword(content, "nspin") == "2"
    assert _keyword(content, "nat") == "2"
    assert _keyword(content, "ntyp") == "2"
    assert _keyword(content, "tot_charge") is None
    assert float(_keyword(content, "starting_magnetization(1)")) == pytest.approx(
        5.0 / 14.0
    )
    assert float(_keyword(content, "starting_magnetization(2)")) == pytest.approx(
        -5.0 / 14.0
    )


@pytest.mark.parametrize(
    ("structure", "store_fixture"),
    [(_olivine_lifepo4, "olivine_store"), (_primitive_nio, "store")],
    ids=["LiFePO4", "NiO"],
)
def test_a_magnetic_insulator_s_default_passes_the_gate_on_smearing(
    structure: Callable[[], Structure],
    store_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    """#138: LiFePO4 and NiO classified non-metal got occupations='fixed'
    on nspin=2 with no tot_magnetization, which pw.x stops on (QE 7.3
    PW/src/input.f90:764-778), so check refused the default (a 422 on
    /run). Nothing knows their moment, so they get smearing."""
    steps, (content,) = _run(
        "scf",
        structure(),
        {"is_metal": False},
        request.getfixturevalue(store_fixture),
    )

    advice = steps[0]
    assert advice.analysis.is_metal.value == "non_metal"
    assert advice.system.magnetic.value.spin_mode == "collinear"
    assert advice.step.kpoints.occupations.value.occupations == "smearing"
    assert _keyword(content, "occupations") == "'smearing'"
    assert _keyword(content, "nspin") == "2"
    assert _keyword(content, "tot_magnetization") is None


@_needs_enumlib
@pytest.mark.parametrize("task", ["scf", "relax"])
def test_a_compensated_afm_insulator_pins_a_zero_moment_on_fixed_occupations(
    task: str, store: AssetStore
) -> None:
    """#138: the engine-built AFM-II NiO cell has an operation reversing
    every spin, so its moment is 0 by symmetry: fixed occupations with
    tot_magnetization = 0, both spin channels holding 24 electrons, and a
    neutral cell (#128). The request itself carries no moment."""
    steps, (content,) = _run(
        task, _primitive_nio(), {"magnetic_ordering": "afm", "is_metal": False}, store
    )

    magnetic = steps[0].system.magnetic.value
    occupations = steps[0].step.kpoints.occupations.value
    assert magnetic.ordering == OrderingRecord(
        requested="afm", applied="afm-1", compensated=True
    )
    assert magnetic.tot_magnetization is None
    assert occupations.occupations == "fixed"
    assert occupations.tot_magnetization == 0.0
    assert _keyword(content, "occupations") == "'fixed'"
    assert float(_keyword(content, "tot_magnetization")) == 0.0
    assert _keyword(content, "nbnd") == "24"
    assert _keyword(content, "tot_charge") is None
