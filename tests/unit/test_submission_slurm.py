from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from goldilocks_core.advisors.job_resources import JobDecision
from goldilocks_core.inputs.hpc import Hardware, HpcProfile, Partition
from goldilocks_core.steps import ExitStatusCheck, StdoutMarkerCheck, Step
from goldilocks_core.submission.errors import SubmissionError
from goldilocks_core.submission.slurm import render_slurm_script

_HARDWARE = Hardware(
    cores_per_node=64, mem_per_node_gb=250.0, max_nodes=454, max_walltime_h=168.0
)
_HPC = HpcProfile(
    name="scarf",
    scheduler="slurm",
    launcher="srun",
    modules={"quantum_espresso": ("QuantumESPRESSO/7.5-foss-2025a",)},
    partitions={"scarf": Partition(name="scarf", hardware=_HARDWARE, default=True)},
)
_JOB = JobDecision(
    partition="scarf",
    nodes=1,
    ntasks=64,
    ntasks_per_node=64,
    walltime_h=168.0,
    max_seconds_by_step={"scf": 574560},
    account=None,
)
_SCHEMA = "./out/pwscf.save/data-file-schema.xml"
_SCF_STEP = Step(
    name="scf",
    executable="pw.x",
    args=("-npool", "4", "-in", "scf.in"),
    files={"scf.in": "&CONTROL\n/\n"},
    stdout="scf.out",
    completion=ExitStatusCheck(_SCHEMA),
)
_NSCF_STEP = Step(
    name="nscf",
    executable="pw.x",
    args=("-npool", "4", "-in", "nscf.in"),
    files={"nscf.in": "&CONTROL\n/\n"},
    stdout="nscf.out",
    completion=ExitStatusCheck(_SCHEMA),
)
_DOS_STEP = Step(
    name="dos",
    executable="dos.x",
    args=("-in", "dos.in"),
    files={"dos.in": "&DOS\n/\n"},
    stdout="dos.out",
    completion=StdoutMarkerCheck(),
    ntasks=1,
)


def test_renders_sbatch_directives_from_job_decision() -> None:
    script = render_slurm_script(_HPC, _JOB, "quantum_espresso", [_SCF_STEP])

    assert script.startswith("#!/bin/bash\n")
    assert "#SBATCH --partition=scarf" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks=64" in script
    assert "#SBATCH --ntasks-per-node=64" in script
    assert "#SBATCH --time=168:00:00" in script


def test_account_directive_only_emitted_when_set() -> None:
    without_account = render_slurm_script(_HPC, _JOB, "quantum_espresso", [_SCF_STEP])
    assert "--account" not in without_account

    with_account = render_slurm_script(
        _HPC, replace(_JOB, account="myproject"), "quantum_espresso", [_SCF_STEP]
    )
    assert "#SBATCH --account=myproject" in with_account


def test_loads_the_module_for_the_given_code_only() -> None:
    script = render_slurm_script(_HPC, _JOB, "quantum_espresso", [_SCF_STEP])

    assert "module load QuantumESPRESSO/7.5-foss-2025a" in script


def test_no_modules_declared_for_a_code_emits_no_module_load_line() -> None:
    script = render_slurm_script(_HPC, _JOB, "vasp", [_SCF_STEP])

    assert "module load" not in script


def test_step_launch_line_never_interprets_code_specific_args() -> None:
    script = render_slurm_script(_HPC, _JOB, "quantum_espresso", [_SCF_STEP])

    assert "srun pw.x -npool 4 -in scf.in > scf.out" in script


def test_a_step_that_declares_ntasks_runs_on_that_many_tasks() -> None:
    """#134: dos.x on all 64 tasks aborts wherever a rank owns no FFT
    plane; the pw.x step keeps the whole allocation."""
    script = render_slurm_script(_HPC, _JOB, "quantum_espresso", [_SCF_STEP, _DOS_STEP])

    lines = script.splitlines()
    assert "#SBATCH --ntasks=64" in lines
    assert "srun pw.x -npool 4 -in scf.in > scf.out" in lines
    assert "srun --nodes=1 --ntasks=1 dos.x -in dos.in > dos.out" in lines


@pytest.mark.parametrize("ntasks, nodes", [(1, 1), (64, 1), (65, 2)])
def test_a_reduced_step_spans_only_the_nodes_its_tasks_fill(
    ntasks: int, nodes: int
) -> None:
    job = replace(_JOB, nodes=2, ntasks=128)
    step = replace(_DOS_STEP, ntasks=ntasks)

    script = render_slurm_script(_HPC, job, "quantum_espresso", [step])

    expected = f"srun --nodes={nodes} --ntasks={ntasks} dos.x -in dos.in > dos.out"
    assert expected in script.splitlines()


def test_a_reduced_step_keeps_the_launchers_own_srun_options() -> None:
    hpc = replace(_HPC, launcher="srun --mpi=pmix")

    script = render_slurm_script(hpc, _JOB, "quantum_espresso", [_DOS_STEP])

    expected = "srun --mpi=pmix --nodes=1 --ntasks=1 dos.x -in dos.in > dos.out"
    assert expected in script.splitlines()


def test_a_reduced_step_on_a_non_srun_launcher_is_refused() -> None:
    """The per-step flags are srun's; other launchers still run
    whole-allocation steps."""
    hpc = replace(_HPC, name="other", launcher="mpirun")

    whole = render_slurm_script(hpc, _JOB, "quantum_espresso", [_SCF_STEP])
    assert "mpirun pw.x -npool 4 -in scf.in > scf.out" in whole.splitlines()

    with pytest.raises(SubmissionError, match=r"dos sets ntasks=1.*other.*mpirun"):
        render_slurm_script(hpc, _JOB, "quantum_espresso", [_SCF_STEP, _DOS_STEP])


def test_every_step_is_wrapped_in_the_check_its_writer_declared() -> None:
    """#129: a pw.x step fingerprints its status file before the launch
    and checks it after; dos.x is checked by its stdout marker."""
    script = render_slurm_script(
        _HPC, _JOB, "quantum_espresso", [_SCF_STEP, _NSCF_STEP, _DOS_STEP]
    )

    lines = script.splitlines()
    fingerprint = f'goldilocks_before=$(goldilocks_fingerprint "{_SCHEMA}")'
    nscf = lines.index("srun pw.x -npool 4 -in nscf.in > nscf.out")
    assert lines[nscf - 1] == fingerprint
    assert lines[nscf + 1] == (
        f'goldilocks_check_exit_status "nscf" "nscf.out" "{_SCHEMA}"'
        ' "$goldilocks_before"'
    )
    dos = lines.index("srun --nodes=1 --ntasks=1 dos.x -in dos.in > dos.out")
    assert lines[dos - 1] == "# dos"
    assert lines[dos + 1] == 'goldilocks_check_stdout_marker "dos" "dos.out" "JOB DONE"'
    assert lines.count(fingerprint) == 2
    assert script.count("goldilocks_check_exit_status() {") == 1
    assert script.count("goldilocks_check_stdout_marker() {") == 1
    assert "pwscf.xml" not in script


def test_only_the_check_functions_the_steps_use_are_emitted() -> None:
    pw_only = render_slurm_script(_HPC, _JOB, "quantum_espresso", [_SCF_STEP])
    dos_only = render_slurm_script(_HPC, _JOB, "quantum_espresso", [_DOS_STEP])

    assert "goldilocks_fingerprint() {" in pw_only
    assert "goldilocks_check_stdout_marker" not in pw_only
    assert "goldilocks_check_stdout_marker() {" in dos_only
    assert "goldilocks_fingerprint" not in dos_only
    assert "goldilocks_check_exit_status" not in dos_only


def test_workdir_emits_a_cd_before_the_status_fingerprint() -> None:
    step = replace(_SCF_STEP, args=("-in", "scf.in"), workdir="run/scf")

    script = render_slurm_script(_HPC, _JOB, "quantum_espresso", [step])

    lines = script.splitlines()
    cd_index = lines.index('cd "run/scf"')
    assert lines[cd_index + 1].startswith("goldilocks_before=$(")
    assert lines[cd_index + 2] == "srun pw.x -in scf.in > scf.out"


@pytest.mark.parametrize(
    "hours, expected", [(168.0, "168:00:00"), (1.5, "1:30:00"), (0.25, "0:15:00")]
)
def test_walltime_formatting(hours: float, expected: str) -> None:
    job = replace(_JOB, walltime_h=hours)
    script = render_slurm_script(_HPC, job, "quantum_espresso", [_SCF_STEP])

    assert f"#SBATCH --time={expected}" in script


def test_non_slurm_scheduler_is_rejected() -> None:
    hpc = HpcProfile(
        name="other",
        scheduler="pbs",
        launcher="mpirun",
        modules={},
        partitions={"main": Partition(name="main", hardware=_HARDWARE, default=True)},
    )

    with pytest.raises(SubmissionError, match="slurm"):
        render_slurm_script(hpc, _JOB, "quantum_espresso", [_SCF_STEP])


def test_no_steps_is_rejected() -> None:
    with pytest.raises(SubmissionError, match="no steps"):
        render_slurm_script(_HPC, _JOB, "quantum_espresso", [])


# The generated submit.sh, run in bash against stand-in programs.

_FAKE_SRUN = """\
#!/bin/bash
# srun stand-in: drop srun's own --options, run the program on this host.
while [ $# -gt 0 ]; do
  case "$1" in
    --*) shift ;;
    *) break ;;
  esac
done
exec "$@"
"""

_FAKE_PW_X = """\
#!/bin/bash
# pw.x stand-in that ends a run the way QE 7.3 does. <name>.status holds
# the exit_status for -in <name>.in (default 0); "none" writes no status
# file and "missing" writes one without <exit_status>. Every exit_status
# goes into out/pwscf.save/data-file-schema.xml (punch.f90:77); only
# punch('all'), after exit_status 0 or 3, copies it to out/pwscf.xml
# (punch.f90:110-120, run_pwscf.f90:197-212 and 340-354). pw.x prints
# JOB DONE and exits 0 whatever the status (stop_run.f90:48 and :65).
while [ $# -gt 0 ]; do
  if [ "$1" = "-in" ]; then name="${2%.in}"; fi
  shift
done
status=$(cat "$name.status" 2>/dev/null || echo 0)
runs=$(( $(cat fake_runs 2>/dev/null || echo 0) + 1 ))
echo "$runs" > fake_runs
schema=out/pwscf.save/data-file-schema.xml
mkdir -p out/pwscf.save
case "$status" in
  none) ;;
  missing) printf '<closed DATE="run %s"/>\\n' "$runs" > "$schema" ;;
  *) printf '<exit_status>%s</exit_status>\\n<closed DATE="run %s"/>\\n' \\
       "$status" "$runs" > "$schema" ;;
esac
if [ "$status" = "0" ] || [ "$status" = "3" ]; then
  cp "$schema" out/pwscf.xml
fi
if [ "$status" = "2" ]; then
  echo "     convergence NOT achieved after 80 iterations: stopping"
fi
echo "   JOB DONE."
"""

_FAKE_DOS_X = """\
#!/bin/bash
# dos.x stand-in: writes no XML; prints JOB DONE unless dos.status says
# "silent".
echo "     Program DOS"
if [ "$(cat dos.status 2>/dev/null)" != "silent" ]; then
  echo "   JOB DONE."
fi
"""

_BARE_HPC = replace(_HPC, name="bare", modules={})
"""srun with no module lines, so the script runs on this host."""


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """An empty run directory, with fake srun, pw.x and dos.x first on PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (
        ("srun", _FAKE_SRUN),
        ("pw.x", _FAKE_PW_X),
        ("dos.x", _FAKE_DOS_X),
    ):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    directory = tmp_path / "run"
    directory.mkdir()
    return directory


def _submit(run_dir: Path, steps: list[Step]) -> subprocess.CompletedProcess[str]:
    """Run the rendered submit.sh with bash in ``run_dir``."""
    script = render_slurm_script(_BARE_HPC, _JOB, "quantum_espresso", steps)
    (run_dir / "submit.sh").write_text(script)
    path = f"{run_dir.parent / 'bin'}{os.pathsep}{os.environ['PATH']}"
    return subprocess.run(
        ["bash", "submit.sh"],
        cwd=run_dir,
        env={**os.environ, "PATH": path},
        capture_output=True,
        text=True,
        check=False,
    )


def test_an_unconverged_scf_fails_although_pw_x_printed_job_done(
    run_dir: Path,
) -> None:
    """#129: after 'convergence NOT achieved' pw.x leaves no out/pwscf.xml,
    prints JOB DONE and exits 0; only the .save schema holds exit_status 2."""
    (run_dir / "scf.status").write_text("2")

    result = _submit(run_dir, [_SCF_STEP])

    assert "JOB DONE" in (run_dir / "scf.out").read_text()
    assert not (run_dir / "out" / "pwscf.xml").exists()
    assert result.returncode == 1
    assert result.stderr == (
        "goldilocks: scf: exit_status 2 (SCF not converged), see scf.out\n"
    )


@pytest.mark.parametrize(
    "status, reason",
    [
        ("1", "exit_status 1 (errore)"),
        ("3", "exit_status 3 (ionic relaxation not converged)"),
        ("131", "exit_status 131"),
    ],
)
def test_a_pw_x_step_that_records_a_failure_status_fails_the_job(
    run_dir: Path, status: str, reason: str
) -> None:
    (run_dir / "scf.status").write_text(status)

    result = _submit(run_dir, [_SCF_STEP])

    assert result.returncode == 1
    assert result.stderr == f"goldilocks: scf: {reason}, see scf.out\n"


def test_a_max_seconds_stop_fails_and_names_walltime_h(run_dir: Path) -> None:
    """goldilocks never renders restart_mode, so 255 names the fix rather
    than calling the step resumable."""
    (run_dir / "scf.status").write_text("255")

    result = _submit(run_dir, [_SCF_STEP])

    assert result.returncode == 1
    assert result.stderr == (
        "goldilocks: scf: exit_status 255 (stopped at max_seconds before "
        "finishing; raise walltime_h or restart from the saved data), "
        "see scf.out\n"
    )


@pytest.mark.parametrize("stale", [None, "0"])
def test_a_pw_x_step_that_writes_no_status_fails(
    run_dir: Path, stale: str | None
) -> None:
    """A status file this step did not write -- absent, or left at 0 by
    an earlier run -- is not this step's status, whatever its stdout."""
    if stale is not None:
        (run_dir / "out" / "pwscf.save").mkdir(parents=True)
        schema = f"<exit_status>{stale}</exit_status>\n"
        (run_dir / "out" / "pwscf.save" / "data-file-schema.xml").write_text(schema)
        (run_dir / "out" / "pwscf.xml").write_text(schema)
    (run_dir / "scf.status").write_text("none")

    result = _submit(run_dir, [_SCF_STEP])

    assert "JOB DONE" in (run_dir / "scf.out").read_text()
    assert result.returncode == 1
    assert result.stderr == (
        f"goldilocks: scf: pw.x wrote no status for this step ({_SCHEMA} "
        "missing or unchanged), see scf.out\n"
    )


def test_a_status_file_without_exit_status_fails(run_dir: Path) -> None:
    (run_dir / "scf.status").write_text("missing")

    result = _submit(run_dir, [_SCF_STEP])

    assert result.returncode == 1
    assert result.stderr == (
        f"goldilocks: scf: no <exit_status> in {_SCHEMA}, see scf.out\n"
    )


def test_the_second_pw_x_step_is_checked_against_its_own_status(
    run_dir: Path,
) -> None:
    """The scf's status file must not pass the nscf that follows it."""
    (run_dir / "nscf.status").write_text("none")

    result = _submit(run_dir, [_SCF_STEP, _NSCF_STEP, _DOS_STEP])

    assert result.returncode == 1
    assert result.stderr.startswith(
        "goldilocks: nscf: pw.x wrote no status for this step"
    )
    assert (run_dir / "fake_runs").read_text().strip() == "2"
    assert not (run_dir / "dos.out").exists()


def test_a_dos_x_step_without_job_done_fails(run_dir: Path) -> None:
    """dos.x writes no XML; the pw.x steps' status file must not pass it."""
    (run_dir / "dos.status").write_text("silent")

    result = _submit(run_dir, [_SCF_STEP, _NSCF_STEP, _DOS_STEP])

    assert result.returncode == 1
    assert result.stderr == "goldilocks: dos: no 'JOB DONE' in dos.out\n"


def test_a_dos_task_whose_steps_all_finish_passes(run_dir: Path) -> None:
    """Each pw.x step rewrites the status file with exit_status 0."""
    result = _submit(run_dir, [_SCF_STEP, _NSCF_STEP, _DOS_STEP])

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert (run_dir / "fake_runs").read_text().strip() == "2"
    assert "JOB DONE" in (run_dir / "dos.out").read_text()
