"""``advise()``/``check()``: the "diagnosis" half of the v2 orchestrator
(v2 epic 8, #8).

Two-phase split, per goldilocks-core-design.md's own promise: ``advise()``
diagnoses field by field and never refuses a request as a whole (a field
it cannot decide is ``unavailable``/``blocked``, not an exception);
``generate()`` (``service/_generate.py``) hard-fails via
``checks.check_all`` if anything needed is ``unavailable``/``blocked``
(delivery has preconditions). A fault is not an undecided field: it
propagates out of both. ``explain`` (CLI, HTTP, MCP) is ``advise()`` ->
``check()``, reporting ``CheckReport.blocking`` without rendering anything
(#165); ``run`` is ``advise()`` -> ``check()`` -> ``generate()`` ->
submission script -> bundle.

``advise_steps`` diagnoses every pw.x step of one task together (#169):
analysis and system advice run once, then each step's own advice, then
one job for the whole task (``_step.py``) -- the steps run one after
another in one allocation, so the job is sized over all of them and
splits the walltime between them. It returns one ``Advice`` per step,
all sharing the same analysis, system and job. ``advise()`` is its
one-scf-step case; ``service/_dos.py`` (scf + nscf) and
``service/_relax.py`` (one relax/vc-relax step) call it directly.
"""

from __future__ import annotations

from collections.abc import Mapping

from pymatgen.core import Structure

from goldilocks_core.advisors.relax import RelaxOptions, VcRelaxOptions
from goldilocks_core.assets.store import AssetStore
from goldilocks_core.checks import CheckReport, check_all
from goldilocks_core.inputs.hpc import HpcProfile
from goldilocks_core.resolution import FieldState
from goldilocks_core.service._advice import Advice, RunOverrides
from goldilocks_core.service._analysis import analyze
from goldilocks_core.service._step import StepOverrides, steps_advice
from goldilocks_core.service._system import system_advice


def advise(
    structure: Structure,
    *,
    code: str = "quantum_espresso",
    hpc: HpcProfile,
    overrides: RunOverrides | None = None,
    store: AssetStore | None = None,
    fetch_missing: bool = False,
) -> Advice:
    """Diagnosis: a field that cannot be decided is ``Unavailable`` or
    ``Blocked``, so a scientifically incomplete request still returns an
    ``Advice``. A fault raises instead: a model that is set up but fails
    to load or predict (see ``ml/predict.py``), an enumeration error that
    is not the enumerator's own answer about the structure (see
    ``advisors/magnetic_config.py``), or a worker process that dies
    (``worker_process.WorkerDied``).

    ``fetch_missing`` governs only the one real download this pipeline
    can trigger (installing the chosen pseudopotential table) -- P8's
    "no sneaky downloads" -- and still degrades to ``Unavailable``
    rather than raising when it is ``False`` and the table is missing.

    One scf step: ``advise_steps`` with ``{"scf": overrides.step}``.
    """
    overrides = overrides or RunOverrides()
    return advise_steps(
        structure,
        code=code,
        hpc=hpc,
        overrides=overrides,
        steps={"scf": overrides.step},
        store=store,
        fetch_missing=fetch_missing,
    )["scf"]


def advise_steps(
    structure: Structure,
    *,
    code: str,
    hpc: HpcProfile,
    overrides: RunOverrides,
    steps: Mapping[str, StepOverrides],
    store: AssetStore | None,
    fetch_missing: bool,
) -> dict[str, Advice]:
    """One ``Advice`` per entry of ``steps`` (a pw.x step's purpose -> its
    own step overrides, in run order), for one task. ``overrides`` gives
    the analysis and system overrides and the task-level job overrides
    (``overrides.step.resources.job``/``job_llm``); ``overrides.step``'s
    other fields reach a step only through ``steps``."""
    analysis = analyze(structure, overrides.analysis)
    system = system_advice(
        structure,
        analysis,
        overrides.system,
        store=store,
        fetch_missing=fetch_missing,
    )
    task_job = overrides.step.resources
    advice_by_step = steps_advice(
        structure, analysis, system, hpc, steps, task_job.job, task_job.job_llm
    )
    return {
        purpose: Advice(
            structure=structure, analysis=analysis, system=system, step=step
        )
        for purpose, step in advice_by_step.items()
    }


def check(
    advice: Advice,
    *,
    purpose: str = "scf",
    relax: FieldState[RelaxOptions | VcRelaxOptions] | None = None,
) -> CheckReport:
    """The ``advise()``/``generate()`` boundary -- see ``checks.py``.

    ``purpose`` (v2 epic 9, #9): the "``occupations='fixed'`` needs an
    integer ``tot_magnetization``" rule (and its non-spin-polarized
    sibling, "needs an even electron count", #70) applies to every
    purpose that runs its own scf loop (``plan.SCF_LIKE_PURPOSES``) --
    an nscf step reads a prior scf step's already-converged density/spin
    and does not re-derive this constraint. Every existing caller is an
    scf step and keeps the previous default unchanged;
    ``service/_dos.py``'s nscf pass is the first caller to pass
    ``purpose="nscf"``.

    ``relax`` (v2 epic 10, #10): only ever passed by ``service/_relax.py``,
    which does not have a plain ``Advice`` field to fold it into --
    every other caller leaves it ``None``.

    ``job``/``parallel``/``nbnd``: always read straight off ``advice``
    (every ``Advice`` has all three) -- the ``npool``-must-divide-
    ``ntasks`` rule and the ``ndiag`` grid rule (#133, ``checks.py``'s
    ``_npool_must_divide_ntasks``/``_ndiag_grid_must_distribute_bands``)
    apply regardless of ``purpose``, each step with its own ``nbnd``.
    ``geometry`` (#44): same "always read straight off ``advice``"
    treatment -- every ``Advice`` has an ``analysis.geometry``, needed
    by ``_fix_bottom_layers_requires_2d_geometry`` regardless of
    ``purpose`` (a non-relax caller never sets
    ``relax.fix_bottom_layers`` in the first place, so the rule is a
    no-op for them). ``boundary`` (#126): read off ``advice`` too, so a
    vc-relax under Martyna-Tuckerman is refused; ``check_all`` also folds
    a ``Blocked`` boundary in, which is why ``SystemAdvice.field_states()``
    leaves it out.
    """
    return check_all(
        *advice.field_states(),
        occupations=advice.step.kpoints.occupations,
        magnetic=advice.system.magnetic,
        electron_count=advice.system.electron_count,
        relax=relax,
        geometry=advice.analysis.geometry,
        boundary=advice.system.boundary,
        job=advice.step.resources.job,
        parallel=advice.step.resources.parallelisation,
        nbnd=advice.step.kpoints.nbnd,
        purpose=purpose,
    )
