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
"""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure
from support import sssp_fixture_table_spec

from goldilocks_core.assets.pseudopotentials.importers import sssp_preparer
from goldilocks_core.assets.store import AssetStore
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


@pytest.mark.parametrize("k_grid", [[1, 1, 1], [3, 3, 3]])
def test_default_scf_leaves_ndiag_to_pw_x(k_grid, tmp_path, monkeypatch) -> None:
    store = _installed_store(tmp_path, monkeypatch)
    overrides = build_overrides({"k_grid": k_grid})

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
    overrides = build_overrides({"k_grid": [1, 1, 1], "ndiag": ndiag})
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
