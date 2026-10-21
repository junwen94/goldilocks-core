"""SLURM submission-script layout. This file makes no decisions.

New in v2 (v2 epic 7, #7); no v1 precedent (v1 never generated a
submission script at all). Where every number in the generated script
comes from -- goldilocks-core-design.md:1297-1310's own signpost,
translated:

    nodes / ntasks / walltime / partition  <- advisors/job_resources.py
                                               (the task's one allocation)
    npool / ndiag / nimage                 <- advisors/parallelisation.py
                                               (already baked into Step.args)
    which programs run, in what order,     <- generation/<code>/<task>.py's
    on how many tasks, and how each            produced Step sequence
    step's completion is checked               (Step.ntasks, Step.completion)
    module load / launcher / partition
    defaults                               <- inputs/profiles/<machine>.toml

This file only lays out SBATCH and launch-line syntax.

**The job sets the allocation; each step sets its own launch shape.**
A step with ``Step.ntasks=None`` runs on the whole allocation through
the bare profile launcher -- the task count ``pw.x``'s ``-npool``/
``-ndiag`` are computed for. A step that declares ``ntasks=N`` runs as
``srun --nodes=ceil(N / ntasks_per_node) --ntasks=N``, so no step
inherits an allocation its program cannot use (``dos.x`` runs on one
task; see ``generation/quantum_espresso/dos.py``). Those flags are
srun's: such a step on a profile whose launcher is not ``srun`` raises
``SubmissionError`` rather than guessing another launcher's syntax.

**Scheduler-only, not code-aware, except for one lookup and the
completion checks below.** goldilocks-core-design.md:1249-1259:
code-specific parallel flags (QE's ``-npool``/``-ndiag`` on the command
line; VASP's ``NPAR``/``KPAR``/``NCORE`` written into ``INCAR`` instead)
are already packed into ``Step.args``/``Step.files`` by the
``generation/<code>/`` writer that produced them -- this module never
inspects or interprets either, so the same ``render_slurm_script``
schedules QE and VASP steps identically. The one lookup is ``code``,
taken as an explicit parameter purely to select ``hpc.modules[code]`` --
which software module to load is a deployment fact about the HPC
profile, not a property of how a ``Step`` is invoked, so it does not
belong on ``Step`` itself.

**Every step is followed by the completion check its writer declared**
(``Step.completion``); a failed check stops the script with a
``goldilocks:`` line on stderr. ``set -e`` alone cannot see a QE program
fail: built without ``__RETURN_EXIT_STATUS``, ``do_stop`` is a bare
``STOP`` (QE 7.3 ``PW/src/stop_run.f90:65``), and ``stop_run`` prints
``JOB DONE.`` through ``environment_end`` whatever the outcome
(``stop_run.f90:48``, ``Modules/environment.f90:165``). ``pw.x``'s real
status is the ``<exit_status>`` it writes into
``<outdir>/<prefix>.save/data-file-schema.xml`` on every ``punch``
(``PW/src/punch.f90:77``): 0 finished, 1 ``errore``, 2 SCF not
converged, 3 ionic relaxation not converged, 255 stopped by
``max_seconds`` (``PW/src/run_pwscf.f90:17-28``). Only ``punch('all')``
copies it to ``<outdir>/<prefix>.xml`` (``punch.f90:110-120``), and a
non-converged or ``max_seconds`` stop calls ``punch('config')``
(``run_pwscf.f90:197-212``), so that copy is missing, or an earlier
step's, exactly when the step failed.

- ``ExitStatusCheck`` (``pw.x``): ``goldilocks_fingerprint`` takes the
  status file's ``cksum`` before the launch line, and
  ``goldilocks_check_exit_status`` passes the step only if the file
  exists, has changed, and records exit_status 0. An unchanged file is
  an earlier step's; deleting it first is not an option, because the
  nscf reads the scf's ``.save``. ``JOB DONE`` never passes a ``pw.x``
  step.
- ``StdoutMarkerCheck`` (``dos.x``, which writes no status file):
  ``goldilocks_check_stdout_marker`` passes the step only if its stdout
  contains the marker.

Only the functions the steps use are emitted. A QE build with
``__RETURN_EXIT_STATUS`` exits non-zero on 2, 3 and 255, so ``set -e``
stops the script before the check runs; the job still fails.

**One job for all steps; ``job_resources`` splits its walltime.**
``JobDecision`` is the task's one job, sized over every step's estimate,
and ``max_seconds_by_step`` gives each ``pw.x`` step a share of 95% of
the walltime in proportion to its estimated workload
(goldilocks-core-design.md:1226-1237's size-weighted split; see
``advisors/job_resources.py``). Each share is written into that step's
input by ``generation/quantum_espresso/scf.py``, not here: baking it in
at generation time keeps the published manifest's sha256 describing the
bytes actually run. ``dos.x`` has no ``max_seconds`` and runs in the
remaining 5% or more.
"""

from __future__ import annotations

import math

from goldilocks_core.advisors.job_resources import JobDecision
from goldilocks_core.inputs.hpc import HpcProfile
from goldilocks_core.steps import ExitStatusCheck, StdoutMarkerCheck, Step
from goldilocks_core.submission.errors import SubmissionError
from goldilocks_core.types import CodeName

_EXIT_STATUS_FUNCTIONS = """\
goldilocks_fingerprint() {
  if [ -f "$1" ]; then cksum < "$1"; fi
  return 0
}

goldilocks_check_exit_status() {
  step_name="$1"; stdout_file="$2"; status_file="$3"; before="$4"
  after=$(goldilocks_fingerprint "$status_file")
  if [ -z "$after" ] || [ "$after" = "$before" ]; then
    reason="pw.x wrote no status for this step (${status_file} missing or unchanged)"
  else
    status=$(grep -o '<exit_status>[0-9]*</exit_status>' "$status_file" \\
      | grep -o '[0-9][0-9]*' || true)
    case "$status" in
      0) return 0 ;;
      "") reason="no <exit_status> in ${status_file}" ;;
      1) reason="exit_status 1 (errore)" ;;
      2) reason="exit_status 2 (SCF not converged)" ;;
      3) reason="exit_status 3 (ionic relaxation not converged)" ;;
      255) reason="exit_status 255 (stopped at max_seconds before finishing;"
           reason="$reason raise walltime_h or restart from the saved data)" ;;
      *) reason="exit_status ${status}" ;;
    esac
  fi
  echo "goldilocks: ${step_name}: ${reason}, see ${stdout_file}" >&2
  exit 1
}
"""

_STDOUT_MARKER_FUNCTION = """\
goldilocks_check_stdout_marker() {
  step_name="$1"; stdout_file="$2"; marker="$3"
  if ! grep -qF -- "$marker" "$stdout_file"; then
    echo "goldilocks: ${step_name}: no '${marker}' in ${stdout_file}" >&2
    exit 1
  fi
}
"""

_CHECK_FUNCTIONS: dict[type, str] = {
    ExitStatusCheck: _EXIT_STATUS_FUNCTIONS,
    StdoutMarkerCheck: _STDOUT_MARKER_FUNCTION,
}
"""The shell functions each ``CompletionCheck`` kind calls, emitted once
per script for the kinds its steps use."""


def render_slurm_script(
    hpc: HpcProfile,
    job: JobDecision,
    code: CodeName,
    steps: list[Step],
) -> str:
    if hpc.scheduler != "slurm":
        raise SubmissionError(
            f"hpc profile {hpc.name!r} does not use slurm (scheduler={hpc.scheduler!r})"
        )
    if not steps:
        raise SubmissionError("no steps to submit")

    lines = ["#!/bin/bash", *_sbatch_directives(job)]
    lines += ["", "set -e", ""]
    lines += [f"module load {module}" for module in hpc.modules.get(code, ())]
    used = {type(step.completion) for step in steps}
    for kind, function in _CHECK_FUNCTIONS.items():
        if kind in used:
            lines += ["", function.rstrip("\n")]
    for step in steps:
        lines += ["", *_step_lines(hpc, job, step)]
    return "\n".join(lines) + "\n"


def _sbatch_directives(job: JobDecision) -> list[str]:
    directives = [
        f"#SBATCH --partition={job.partition}",
        f"#SBATCH --nodes={job.nodes}",
        f"#SBATCH --ntasks={job.ntasks}",
        f"#SBATCH --ntasks-per-node={job.ntasks_per_node}",
        f"#SBATCH --time={_format_walltime(job.walltime_h)}",
    ]
    if job.account is not None:
        directives.append(f"#SBATCH --account={job.account}")
    return directives


def _format_walltime(hours: float) -> str:
    total_seconds = round(hours * 3600)
    whole_hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{whole_hours}:{minutes:02d}:{seconds:02d}"


def _step_lines(hpc: HpcProfile, job: JobDecision, step: Step) -> list[str]:
    lines = [f"# {step.name}"]
    if step.workdir is not None:
        lines.append(f'cd "{step.workdir}"')
    before, after = _completion_lines(step)
    command = " ".join([*_launcher(hpc, job, step), step.executable, *step.args])
    return [*lines, *before, f"{command} > {step.stdout}", *after]


def _completion_lines(step: Step) -> tuple[list[str], list[str]]:
    """The lines before and after ``step``'s launch line that check it
    finished: an ``ExitStatusCheck`` fingerprints the status file before
    the launch and reads it after; a ``StdoutMarkerCheck`` greps the
    stdout after."""
    name, stdout, completion = step.name, step.stdout, step.completion
    if isinstance(completion, StdoutMarkerCheck):
        marker = completion.marker
        return [], [f'goldilocks_check_stdout_marker "{name}" "{stdout}" "{marker}"']
    status_file = completion.status_file
    return (
        [f'goldilocks_before=$(goldilocks_fingerprint "{status_file}")'],
        [
            f'goldilocks_check_exit_status "{name}" "{stdout}" "{status_file}"'
            ' "$goldilocks_before"'
        ],
    )


def _launcher(hpc: HpcProfile, job: JobDecision, step: Step) -> list[str]:
    """The launch words before ``step``'s executable: the profile launcher
    alone for a whole-allocation step, plus srun's ``--nodes``/``--ntasks``
    for a step that declares its own ``ntasks``. Raises ``SubmissionError``
    when such a step meets a launcher that is not srun."""
    if step.ntasks is None:
        return [hpc.launcher]
    if hpc.launcher.split()[:1] != ["srun"]:
        raise SubmissionError(
            f"step {step.name} sets ntasks={step.ntasks}, but hpc profile "
            f"{hpc.name} launches with {hpc.launcher}; a per-step task count "
            "is rendered only for srun"
        )
    nodes = math.ceil(step.ntasks / job.ntasks_per_node)
    return [hpc.launcher, f"--nodes={nodes}", f"--ntasks={step.ntasks}"]
