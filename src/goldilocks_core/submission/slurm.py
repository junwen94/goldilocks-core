"""SLURM submission-script layout. This file makes no decisions.

New in v2 (v2 epic 7, #7); no v1 precedent (v1 never generated a
submission script at all). Where every number in the generated script
comes from -- goldilocks-core-design.md:1297-1310's own signpost,
translated:

    nodes / ntasks / walltime / partition  <- advisors/job_resources.py
    npool / ndiag / nimage                 <- advisors/parallelisation.py
                                               (already baked into Step.args)
    which programs run, in what order      <- generation/<code>/<task>.py's
                                               produced Step sequence
    module load / launcher / partition
    defaults                               <- inputs/profiles/<machine>.toml

This file only lays out SBATCH syntax and formatting.

**Scheduler-only, not code-aware, except for one lookup.**
goldilocks-core-design.md:1249-1259: code-specific parallel flags
(QE's ``-npool``/``-ndiag`` on the command line; VASP's ``NPAR``/
``KPAR``/``NCORE`` written into ``INCAR`` instead) are already packed
into ``Step.args``/``Step.files`` by the ``generation/<code>/`` writer
that produced them -- this module never inspects or interprets either,
so the same ``render_slurm_script`` schedules QE and VASP steps
identically. The one exception is ``code``, taken as an explicit
parameter purely to select ``hpc.modules[code]`` -- which software
module to load is a deployment fact about the HPC profile, not a
property of how a ``Step`` is invoked, so it does not belong on
``Step`` itself.

**``set -e`` alone cannot catch a QE program reporting failure**
(goldilocks-core-design.md:1186-1224, verified against QE's own source,
2026-09-14): compiled without ``__RETURN_EXIT_STATUS``, ``PW/src/
stop_run.f90``'s ``do_stop`` calls a bare Fortran ``STOP`` -- exit code
0 to the shell -- regardless of whether SCF/ionic convergence actually
succeeded. QE instead writes its real ``exit_status`` (0 success, 1
``errore()``, 2 SCF not converged, 3 ionic relaxation not converged,
255 stopped by ``max_seconds``/signal -- resumable, unlike 2/3, which
mean "finished, but the result is not trustworthy") as an
``<exit_status>`` element via ``Modules/qexsd.f90``'s
``qexsd_closeschema``, copied by ``PW/src/punch.f90`` to
``<outdir>/<prefix>.xml`` for exactly this kind of check (both verified
directly against QE's ``develop`` branch source, 2026-09-14, not
inferred). ``goldilocks_check_exit_status`` (emitted once per script,
called after every step) reads that file; a step with no such file
(pure post-processing binaries like ``dos.x`` never write one) falls
back to grepping ``JOB DONE`` in its stdout, the universal completion
marker every QE program prints. A missing XML *and* no ``JOB DONE`` --
the SIGKILL-before-finishing case -- is treated as failure, same as an
explicit non-zero ``exit_status``.

**Walltime is allocated whole to however many steps there are.**
goldilocks-core-design.md:1226-1237 calls for a *proportional*, size
-weighted split across multiple steps -- there is only ever one step
in this epic's own generation output (``scf``), so there is nothing yet
to split proportionally between; ``JobDecision.max_seconds`` is written
into the QE input itself by ``generation/quantum_espresso/scf.py``, not
here (see that module's docstring for why: baking it in at generation
time, not injecting it into the input at submission time, is what keeps
the published manifest's sha256 describing the bytes actually run).
Splitting proportionally across several steps' own ``size`` estimates
is deferred to whichever future task-writer (``nscf``/``bands``/
``phonon``) actually produces more than one step.
"""

from __future__ import annotations

from goldilocks_core.advisors.job_resources import JobDecision
from goldilocks_core.inputs.hpc import HpcProfile
from goldilocks_core.steps import SharedContext, Step
from goldilocks_core.submission.errors import SubmissionError
from goldilocks_core.types import CodeName

_CHECK_FUNCTION = """\
goldilocks_check_exit_status() {
  step_name="$1"; stdout_file="$2"; xml_file="$3"
  if [ -f "$xml_file" ]; then
    status=$(grep -o '<exit_status>[0-9-]*</exit_status>' "$xml_file" \\
      | grep -o '[0-9-]*' || true)
    if [ -z "$status" ]; then
      echo "goldilocks: ${step_name}: no <exit_status> in ${xml_file}" >&2
      exit 1
    elif [ "$status" = "255" ]; then
      echo "goldilocks: ${step_name}: exit_status=255, resumable" >&2
      exit 1
    elif [ "$status" != "0" ]; then
      echo "goldilocks: ${step_name}: exit_status=${status}, not resumable" >&2
      exit 1
    fi
  elif ! grep -q "JOB DONE" "$stdout_file" 2>/dev/null; then
    echo "goldilocks: ${step_name}: no exit_status, no JOB DONE" >&2
    exit 1
  fi
}
"""


def render_slurm_script(
    hpc: HpcProfile,
    job: JobDecision,
    code: CodeName,
    ctx: SharedContext,
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
    lines += ["", _CHECK_FUNCTION.rstrip("\n")]
    for step in steps:
        lines += ["", *_step_lines(hpc, ctx, step)]
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


def _step_lines(hpc: HpcProfile, ctx: SharedContext, step: Step) -> list[str]:
    lines = [f"# {step.name}"]
    if step.workdir is not None:
        lines.append(f'cd "{step.workdir}"')
    command = " ".join([hpc.launcher, step.executable, *step.args])
    if step.stdout is not None:
        command += f" > {step.stdout}"
    lines.append(command)
    xml_file = f"{ctx.outdir}/{ctx.prefix}.xml"
    lines.append(
        f'goldilocks_check_exit_status "{step.name}" "{step.stdout}" "{xml_file}"'
    )
    return lines
