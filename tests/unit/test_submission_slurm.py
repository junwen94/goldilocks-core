from __future__ import annotations

from dataclasses import replace

import pytest

from goldilocks_core.advisors.job_resources import JobDecision
from goldilocks_core.inputs.hpc import Hardware, HpcProfile, Partition
from goldilocks_core.steps import SharedContext, Step
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
    has_scalapack={"quantum_espresso": True},
    partitions={"scarf": Partition(name="scarf", hardware=_HARDWARE, default=True)},
)
_JOB = JobDecision(
    partition="scarf",
    nodes=1,
    ntasks=64,
    ntasks_per_node=64,
    walltime_h=168.0,
    max_seconds=574560,
    account=None,
)
_CTX = SharedContext(prefix="pwscf", outdir="./out", pseudo_dir="./pseudo")
_SCF_STEP = Step(
    name="scf",
    executable="pw.x",
    args=("-npool", "4", "-in", "scf.in"),
    files={"scf.in": "&CONTROL\n/\n"},
    stdout="scf.out",
)


def test_renders_sbatch_directives_from_job_decision() -> None:
    script = render_slurm_script(_HPC, _JOB, "quantum_espresso", _CTX, [_SCF_STEP])

    assert script.startswith("#!/bin/bash\n")
    assert "#SBATCH --partition=scarf" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks=64" in script
    assert "#SBATCH --ntasks-per-node=64" in script
    assert "#SBATCH --time=168:00:00" in script


def test_account_directive_only_emitted_when_set() -> None:
    without_account = render_slurm_script(
        _HPC, _JOB, "quantum_espresso", _CTX, [_SCF_STEP]
    )
    assert "--account" not in without_account

    with_account = render_slurm_script(
        _HPC,
        replace(_JOB, account="myproject"),
        "quantum_espresso",
        _CTX,
        [_SCF_STEP],
    )
    assert "#SBATCH --account=myproject" in with_account


def test_loads_the_module_for_the_given_code_only() -> None:
    script = render_slurm_script(_HPC, _JOB, "quantum_espresso", _CTX, [_SCF_STEP])

    assert "module load QuantumESPRESSO/7.5-foss-2025a" in script


def test_no_modules_declared_for_a_code_emits_no_module_load_line() -> None:
    script = render_slurm_script(_HPC, _JOB, "vasp", _CTX, [_SCF_STEP])

    assert "module load" not in script


def test_step_launch_line_never_interprets_code_specific_args() -> None:
    script = render_slurm_script(_HPC, _JOB, "quantum_espresso", _CTX, [_SCF_STEP])

    assert "srun pw.x -npool 4 -in scf.in > scf.out" in script


def test_every_step_is_followed_by_an_exit_status_check() -> None:
    second_step = Step(
        name="nscf",
        executable="pw.x",
        args=("-npool", "4", "-in", "nscf.in"),
        files={"nscf.in": "&CONTROL\n/\n"},
        stdout="nscf.out",
    )
    script = render_slurm_script(
        _HPC, _JOB, "quantum_espresso", _CTX, [_SCF_STEP, second_step]
    )

    assert 'goldilocks_check_exit_status "scf" "scf.out" "./out/pwscf.xml"' in script
    assert 'goldilocks_check_exit_status "nscf" "nscf.out" "./out/pwscf.xml"' in script
    assert script.count("goldilocks_check_exit_status(") == 1  # defined once
    assert script.count('goldilocks_check_exit_status "') == 2  # called per step


def test_check_function_prefers_xml_exit_status_over_job_done() -> None:
    script = render_slurm_script(_HPC, _JOB, "quantum_espresso", _CTX, [_SCF_STEP])

    assert "<exit_status>" in script
    assert "JOB DONE" in script
    assert "255" in script  # resumable case called out distinctly


def test_workdir_emits_a_cd_before_the_launch_line() -> None:
    step = Step(
        name="scf",
        executable="pw.x",
        args=("-in", "scf.in"),
        files={"scf.in": ""},
        stdout="scf.out",
        workdir="run/scf",
    )
    script = render_slurm_script(_HPC, _JOB, "quantum_espresso", _CTX, [step])

    lines = script.splitlines()
    cd_index = lines.index('cd "run/scf"')
    assert "srun pw.x -in scf.in > scf.out" in lines[cd_index + 1]


@pytest.mark.parametrize(
    "hours, expected", [(168.0, "168:00:00"), (1.5, "1:30:00"), (0.25, "0:15:00")]
)
def test_walltime_formatting(hours: float, expected: str) -> None:
    job = replace(_JOB, walltime_h=hours)
    script = render_slurm_script(_HPC, job, "quantum_espresso", _CTX, [_SCF_STEP])

    assert f"#SBATCH --time={expected}" in script


def test_non_slurm_scheduler_is_rejected() -> None:
    hpc = HpcProfile(
        name="other",
        scheduler="pbs",
        launcher="mpirun",
        modules={},
        has_scalapack={},
        partitions={"main": Partition(name="main", hardware=_HARDWARE, default=True)},
    )

    with pytest.raises(SubmissionError, match="slurm"):
        render_slurm_script(hpc, _JOB, "quantum_espresso", _CTX, [_SCF_STEP])


def test_no_steps_is_rejected() -> None:
    with pytest.raises(SubmissionError, match="no steps"):
        render_slurm_script(_HPC, _JOB, "quantum_espresso", _CTX, [])
