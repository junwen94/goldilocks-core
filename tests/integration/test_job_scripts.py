"""Job scripts on the real SCARF profile, end to end through
``advise()``/``check()``/``generate()`` with the offline SSSP fixture.

The fixture cell is two Si atoms in a 5.43 A cubic cell: nbnd 8 on the
isolated asset root (no ML tier), n_irr_k 1 with ``k_grid=[1,1,1]`` and
6 with ``k_grid=[3,3,3]``. Before #133 SCARF rendered those as
``-npool 1 -ndiag 64`` and ``-npool 4 -ndiag 16``: the H2O/N2-box and
C2-box layouts that stopped or hung pw.x on SCARF. Before #134 the dos
task launched dos.x on all 64 tasks, where it aborts on every rank that
owns no FFT z-plane. Before #129 every step's check read out/pwscf.xml,
which pw.x writes only after a finished run, and passed on JOB DONE.
Before #169 the dos task's scf and nscf each got max_seconds = 0.95 x
the whole walltime, although pw.x counts it from its own start. Before
#148 the job record carried no memory figure, and nodes were sized on
one pool's single copy of the wavefunctions and density.
"""

from __future__ import annotations

import json

import pytest
from pymatgen.core import Lattice, Structure
from support import sssp_fixture_table_spec

from goldilocks_core.advisors.size import Species, memory_estimate, resource_estimate
from goldilocks_core.assets.pseudopotentials.importers import sssp_preparer
from goldilocks_core.assets.store import AssetStore
from goldilocks_core.bundle import DirectoryOutput, publish
from goldilocks_core.inputs.hpc import load_hpc_profile
from goldilocks_core.service import (
    advise,
    advise_dos,
    check,
    check_dos,
    generate,
    generate_dos,
    render_submission,
    render_submission_dos,
    to_bundle_input,
)
from goldilocks_core.set_overrides import build_overrides
from goldilocks_core.steps import default_shared_context


def _installed_store(tmp_path, monkeypatch) -> AssetStore:
    spec, registry_table = sssp_fixture_table_spec(tmp_path / "sources")
    monkeypatch.setattr(
        "goldilocks_core.service._pseudo.load_tables",
        lambda *a, **kw: {registry_table.id: registry_table},
    )
    store = AssetStore(tmp_path / "store")
    store.install(spec, sssp_preparer(registry_table))
    return store


def _silicon() -> Structure:
    return Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])


def _diamond_silicon() -> Structure:
    """Diamond Si's two-atom primitive cell, as in the #169 reproduction."""
    a = 5.43
    lattice = Lattice([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    return Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])


@pytest.mark.parametrize("k_grid", [[1, 1, 1], [3, 3, 3]])
def test_default_scf_leaves_ndiag_to_pw_x(k_grid, tmp_path, monkeypatch) -> None:
    store = _installed_store(tmp_path, monkeypatch)
    overrides = build_overrides({"k_grid": k_grid}, task="scf_single_point")

    advice = advise(
        _silicon(), hpc=load_hpc_profile("scarf"), overrides=overrides, store=store
    )
    report = check(advice)

    assert report.ok, report.blocking
    assert advice.step.kpoints.nbnd.value.nbnd == 8
    assert advice.step.resources.parallelisation.value.ndiag is None
    (scf,) = generate(advice, report)
    assert "-ndiag" not in scf.args


def _gamma_only_advice(ndiag: int, tmp_path, monkeypatch):
    store = _installed_store(tmp_path, monkeypatch)
    overrides = build_overrides(
        {"k_grid": [1, 1, 1], "ndiag": ndiag}, task="scf_single_point"
    )
    return advise(
        _silicon(), hpc=load_hpc_profile("scarf"), overrides=overrides, store=store
    )


def test_user_ndiag_that_cannot_hold_nbnd_is_refused(tmp_path, monkeypatch) -> None:
    """Gamma-only on one 64-core node: ndiag=64 is an 8x8 grid for 8
    bands, which pw.x cannot run (the H2O/N2-box failure)."""
    advice = _gamma_only_advice(64, tmp_path, monkeypatch)

    report = check(advice)

    (reason,) = report.blocking
    assert "ndiag=64" in reason
    assert "8x8" in reason
    assert "nbnd=8" in reason


def test_user_ndiag_that_holds_nbnd_reaches_pw_x(tmp_path, monkeypatch) -> None:
    advice = _gamma_only_advice(4, tmp_path, monkeypatch)

    report = check(advice)

    assert report.ok, report.blocking
    (scf,) = generate(advice, report)
    assert scf.args[scf.args.index("-ndiag") + 1] == "4"


def test_dos_task_runs_dos_x_on_one_task(tmp_path, monkeypatch) -> None:
    store = _installed_store(tmp_path, monkeypatch)
    hpc = load_hpc_profile("scarf")
    ctx = default_shared_context()

    advice = advise_dos(_silicon(), hpc=hpc, store=store)
    report = check_dos(advice)
    assert report.ok, report.blocking
    steps = generate_dos(advice, report, ctx=ctx)
    script = render_submission_dos(advice, hpc, "quantum_espresso", steps)

    lines = script.splitlines()
    assert "#SBATCH --ntasks=64" in lines
    assert "srun --nodes=1 --ntasks=1 dos.x -in dos.in > dos.out" in lines
    pw_lines = [line for line in lines if " pw.x " in line]
    assert len(pw_lines) == 2
    assert all(line.startswith("srun pw.x -npool ") for line in pw_lines)


def test_scf_script_checks_the_status_pw_x_always_writes(tmp_path, monkeypatch) -> None:
    """#129: pw.x copies its status to out/pwscf.xml only after a finished
    run, so the check must read the .save schema and fingerprint it first."""
    store = _installed_store(tmp_path, monkeypatch)
    hpc = load_hpc_profile("scarf")
    advice = advise(_silicon(), hpc=hpc, store=store)
    report = check(advice)
    assert report.ok, report.blocking
    steps = generate(advice, report, ctx=default_shared_context())

    script = render_submission(advice, hpc, "quantum_espresso", steps)

    schema = "./out/pwscf.save/data-file-schema.xml"
    lines = script.splitlines()
    (launch,) = (i for i, line in enumerate(lines) if line.startswith("srun pw.x "))
    assert lines[launch].endswith(" -in scf.in > scf.out")
    assert (
        lines[launch - 1] == f'goldilocks_before=$(goldilocks_fingerprint "{schema}")'
    )
    assert lines[launch + 1] == (
        f'goldilocks_check_exit_status "scf" "scf.out" "{schema}" "$goldilocks_before"'
    )
    assert "pwscf.xml" not in script
    assert "JOB DONE" not in script


def test_dos_script_checks_each_step_by_its_own_completion(
    tmp_path, monkeypatch
) -> None:
    """The nscf is fingerprinted separately from the scf, so a max_seconds
    stop in the nscf is not passed on the scf's status before dos.x runs."""
    store = _installed_store(tmp_path, monkeypatch)
    hpc = load_hpc_profile("scarf")
    advice = advise_dos(_silicon(), hpc=hpc, store=store)
    report = check_dos(advice)
    assert report.ok, report.blocking
    steps = generate_dos(advice, report, ctx=default_shared_context())

    script = render_submission_dos(advice, hpc, "quantum_espresso", steps)

    schema = "./out/pwscf.save/data-file-schema.xml"
    lines = script.splitlines()
    assert lines.count(f'goldilocks_before=$(goldilocks_fingerprint "{schema}")') == 2
    assert (
        f'goldilocks_check_exit_status "nscf" "nscf.out" "{schema}" '
        '"$goldilocks_before"'
    ) in lines
    assert lines[-1] == 'goldilocks_check_stdout_marker "dos" "dos.out" "JOB DONE"'


def _max_seconds(content: str) -> int:
    (line,) = (line for line in content.splitlines() if "max_seconds" in line)
    return int(line.split("=")[1])


def test_dos_task_splits_one_walltime_across_its_pw_x_steps(
    tmp_path, monkeypatch
) -> None:
    """#169: pw.x counts max_seconds from its own start, so the scf and the
    nscf of one allocation share 95% of its walltime (13680 s of 4 h) by
    workload: k-point/spin pairs x nbnd x 3 SCF passes for the scf, x 1
    for the nscf (npw and npol are the same for both steps and cancel).
    On the fixture that is 104 x 8 x 3 against 286 x 8, so neither step
    is assumed to be the larger one."""
    store = _installed_store(tmp_path, monkeypatch)
    hpc = load_hpc_profile("scarf")
    overrides = build_overrides({"walltime_h": 4}, task="dos")
    advice = advise_dos(_diamond_silicon(), hpc=hpc, overrides=overrides, store=store)
    report = check_dos(advice)
    assert report.ok, report.blocking
    steps = generate_dos(advice, report, ctx=default_shared_context())
    scf, nscf, _dos = steps

    seconds = {
        "scf": _max_seconds(scf.files["scf.in"]),
        "nscf": _max_seconds(nscf.files["nscf.in"]),
    }
    assert sum(seconds.values()) <= 13680
    assert min(seconds.values()) > 0
    workload = {
        name: (
            step.resources.parallelisation.value.kpoint_spin_pairs
            * step.kpoints.nbnd.value.nbnd
            * passes
        )
        for name, step, passes in (
            ("scf", advice.scf.step, 3),
            ("nscf", advice.nscf.step, 1),
        )
    }
    for name, share in seconds.items():
        assert abs(share - 13680 * workload[name] / sum(workload.values())) < 1

    records = advice.records()
    assert "nscf_job" not in records
    job = records["job"].value
    assert job.max_seconds_by_step == seconds
    script = render_submission_dos(advice, hpc, "quantum_espresso", steps)
    lines = script.splitlines()
    assert f"#SBATCH --partition={job.partition}" in lines
    assert f"#SBATCH --nodes={job.nodes}" in lines
    assert f"#SBATCH --ntasks={job.ntasks}" in lines
    assert f"#SBATCH --ntasks-per-node={job.ntasks_per_node}" in lines
    assert job.walltime_h == 4
    assert "#SBATCH --time=4:00:00" in lines


def test_goldilocks_json_publishes_the_memory_the_job_was_sized_on(
    tmp_path, monkeypatch
) -> None:
    """#148: the job record carries pw.x's estimated memory per process
    and in total for what the bundle renders -- the cell, the fixture
    UPF's 4 projectors up to l=1, the cutoffs, bands, k-point/spin pairs
    and mixing mode of its records, and the layout that runs (-npool 16
    for 65 pairs on 64 tasks) -- the figures the node count was sized on."""
    store = _installed_store(tmp_path, monkeypatch)
    hpc = load_hpc_profile("scarf")
    ctx = default_shared_context()
    advice = advise(_silicon(), hpc=hpc, store=store)
    report = check(advice)
    assert report.ok, report.blocking
    steps = generate(advice, report, ctx=ctx)
    script = render_submission(advice, hpc, "quantum_espresso", steps)
    publish(
        to_bundle_input(advice, steps, script, ctx), DirectoryOutput(tmp_path / "out")
    )

    records = json.loads((tmp_path / "out" / "goldilocks.json").read_text())["records"]
    job = records["job"]["value"]
    parallel = records["parallelisation"]["value"]
    cutoffs = records["cutoffs"]["value"]
    assert records["magnetic"]["value"]["spin_polarized"] is False
    assert parallel["npool"] == 16
    assert parallel["ndiag"] is None  # 8 bands: pw.x keeps a 1x1 grid
    estimate = resource_estimate(
        _silicon().lattice,
        (Species(atoms=2, projectors=4, l_max=1, ultrasoft=False),),
        ecutwfc_ry=cutoffs["ecutwfc_ry"],
        ecutrho_ry=cutoffs["ecutrho_ry"],
        nbnd=records["nbnd"]["value"]["nbnd"],
        n_kpoint_spin=parallel["kpoint_spin_pairs"],
        nspin=1,
        npol=1,
        nspin_mag=1,
        scf_loop=True,
        variable_cell=False,
        local_tf_mixing=records["convergence"]["value"]["mixing_mode"] == "local-TF",
    )
    memory = memory_estimate(
        estimate, ntasks=job["ntasks"], npool=parallel["npool"], diag_side=1
    )
    assert job["memory_per_process_mb"] == round(memory.per_process_mb, 2)
    assert job["memory_total_mb"] == round(memory.total_mb, 2)
    assert job["nodes"] == 1


def test_the_memory_estimate_counts_the_species_the_writer_renders(
    tmp_path, monkeypatch
) -> None:
    """#148: pw.x's memory counts structure factors and qrad tables per
    species, so the estimate takes scf.in's species -- one per element and
    starting magnetization -- not one per element: Si0 and Si1 with
    opposite moments are written as Si1 and Si2."""
    store = _installed_store(tmp_path, monkeypatch)
    hpc = load_hpc_profile("scarf")
    structure = Structure(
        Lattice.cubic(5.43),
        ["Si", "Si"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
        labels=["Si0", "Si1"],
    )
    overrides = build_overrides(
        {
            "spin_polarized": True,
            "starting_magnetization": {"Si0": 0.5, "Si1": -0.5},
            "occupations": "smearing",
        },
        task="scf_single_point",
    )
    advice = advise(structure, hpc=hpc, overrides=overrides, store=store)
    report = check(advice)
    assert report.ok, report.blocking
    (scf,) = generate(advice, report)
    assert "ntyp             = 2" in scf.files["scf.in"]

    job = advice.step.resources.job.value
    parallel = advice.step.resources.parallelisation.value
    one_species = Species(atoms=1, projectors=4, l_max=1, ultrasoft=False)
    estimate = resource_estimate(
        structure.lattice,
        (one_species, one_species),
        ecutwfc_ry=advice.system.cutoffs.value.ecutwfc_ry,
        ecutrho_ry=advice.system.cutoffs.value.ecutrho_ry,
        nbnd=advice.step.kpoints.nbnd.value.nbnd,
        n_kpoint_spin=parallel.kpoint_spin_pairs,
        nspin=2,
        npol=1,
        nspin_mag=2,
        scf_loop=True,
        variable_cell=False,
        local_tf_mixing=advice.step.kpoints.convergence.value.mixing_mode == "local-TF",
    )
    memory = memory_estimate(
        estimate, ntasks=job.ntasks, npool=parallel.npool, diag_side=1
    )
    assert job.memory_per_process_mb == round(memory.per_process_mb, 2)
