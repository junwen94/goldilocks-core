"""``advise_dos()``/``check_dos()``/``generate_dos()``: the ``dos`` task's
multi-step path (v2 epic 9, #9) -- goldilocks-implementation-plan.md's
five-scenario table, scenario 5 ("DOS multi-step"), and the design doc's
own admission that step-sequencing has never once been exercised before
this epic.

**Reuses the single-step pipeline twice, rather than rebuilding
``Advice``/``StepAdvice`` as "one-or-many".** ``plan.py``'s own
``expand_task`` already says how many steps a task needs and what
program each runs; nothing about *how* a single ``pw.x`` step's own
settings get resolved changes between an scf task and dos's scf/nscf
steps -- the same tri-state advisor pipeline (``advise()``) answers both,
just with the nscf-specific overrides ``_nscf_overrides`` below forces.
Rebuilding the whole delivery-layer response shape (``Advice``,
``StepAdvice``, and every CLI/HTTP/MCP consumer of them) to be
one-or-many per task is real work with no second task yet to prove it
against beyond this one -- exactly the "do not build for a caller that
does not exist yet" reasoning this codebase applies everywhere else,
and well past this epic's own explicitly narrow scope ("prove the
type-level split and a real multi-step path, not full DOS science").

**nscf overrides, sourced from a real, citable place** (aiida-
quantumespresso's own ``workflows/protocols/pdos.yaml``, confirmed
against its source 2026-09-14, "balanced" protocol): a denser k-mesh
(``kpoints_distance: 0.10``, versus this codebase's own scf heuristic
defaults of 0.15/0.30 Å^-1) and ``tetrahedra_opt`` occupations, for a
DOS-quality uniform-enough integration grid. aiida's own protocol also
sets ``nosym: True`` for the nscf step; this module does not, because
nothing in the generated Quantum ESPRESSO input would currently honor
it -- ``generation/quantum_espresso/scf.py``'s own docstring already
names ``nosym``/``noinv`` as "a pre-existing, documented gap in epic 6's
output, not something this epic re-decides". Forcing an override with no
visible effect on the rendered file would be misleading, not merely
incomplete, so it is left out until that gap has a real fix.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from pymatgen.core import Structure

from goldilocks_core.advisors.dos import (
    DosDecision,
    DosHumanInput,
    DosLlmInput,
    dos_settings,
)
from goldilocks_core.advisors.k_sampling import KSamplingHumanInput
from goldilocks_core.advisors.occupations import OccupationsHumanInput
from goldilocks_core.assets.store import AssetStore
from goldilocks_core.bundle import BundleInput
from goldilocks_core.checks import CheckReport, collect_blocked
from goldilocks_core.generation.quantum_espresso.dos import write_qe_dos
from goldilocks_core.inputs.hpc import HpcProfile
from goldilocks_core.resolution import FieldState
from goldilocks_core.service._advice import Advice, RunOverrides, warnings_from_records
from goldilocks_core.service._bundle import assemble_bundle_input
from goldilocks_core.service._generate import AdviceIncomplete, generate
from goldilocks_core.service._pipeline import advise, check
from goldilocks_core.step_settings import DosSettings
from goldilocks_core.steps import SharedContext, Step, default_shared_context
from goldilocks_core.submission.slurm import render_slurm_script

_NSCF_K_DISTANCE = 0.10
"""Å^-1. aiida-quantumespresso's own ``pdos.yaml`` "balanced" protocol
nscf ``kpoints_distance`` -- see this module's own docstring."""


@dataclass(frozen=True, slots=True)
class DosAdvice:
    scf: Advice
    nscf: Advice
    dos: FieldState[DosDecision]

    def field_states(self) -> tuple[FieldState[object], ...]:
        return self.scf.field_states() + self.nscf.field_states() + (self.dos,)

    def records(self) -> dict[str, FieldState[object]]:
        """``scf``'s records unprefixed (system-level fields --
        ``functional``/``cutoffs``/``pseudopotentials``/``magnetic``/
        etc. -- are identical between ``scf`` and ``nscf``, since only
        per-step overrides differ between the two ``advise()`` calls
        ``advise_dos`` makes), ``nscf``'s own per-step fields under an
        ``nscf_`` prefix (v2 epic 9, #9, #28's delivery-layer routing --
        the first caller that ever needs to tell the two steps' k
        -sampling/occupations/nbnd/convergence/job apart), plus ``dos``
        itself."""
        merged: dict[str, FieldState[object]] = dict(self.scf.records())
        merged.update(
            {f"nscf_{name}": state for name, state in self.nscf.step.records().items()}
        )
        merged["dos"] = self.dos
        return merged

    def warnings(self) -> list[dict[str, object]]:
        return warnings_from_records(self.records())


def _nscf_overrides(overrides: RunOverrides) -> RunOverrides:
    """Applies the aiida nscf protocol defaults only where the caller
    didn't already give an explicit occupations/k-sampling override
    (#33, v2 epic 9, #9) -- this used to replace both fields
    unconditionally, so a flat ``--set k_grid=...``/``--set
    occupations=...`` (there is no way to scope one to just the nscf
    step yet, see ``set_overrides.py``'s own docstring) was silently
    discarded for the nscf step specifically, even though the exact
    same flat override correctly reached the scf step. A caller's own
    explicit choice now applies to both steps, same as every other
    flat per-step override in this codebase, rather than being
    overwritten for one of them."""
    kpoints = overrides.step.kpoints
    if kpoints.occupations is None:
        kpoints = dataclasses.replace(
            kpoints, occupations=OccupationsHumanInput(occupations="tetrahedra_opt")
        )
    if kpoints.k_sampling is None:
        kpoints = dataclasses.replace(
            kpoints, k_sampling=KSamplingHumanInput(k_distance=_NSCF_K_DISTANCE)
        )
    return dataclasses.replace(
        overrides, step=dataclasses.replace(overrides.step, kpoints=kpoints)
    )


def advise_dos(
    structure: Structure,
    *,
    code: str = "quantum_espresso",
    hpc: HpcProfile,
    overrides: RunOverrides | None = None,
    dos_human: DosHumanInput | None = None,
    dos_llm: DosLlmInput | None = None,
    store: AssetStore | None = None,
    fetch_missing: bool = False,
) -> DosAdvice:
    """Diagnosis for all three of dos's steps, always returning -- same
    "never aborts as a whole" promise ``advise()`` makes."""
    overrides = overrides or RunOverrides()
    scf = advise(
        structure,
        code=code,
        hpc=hpc,
        overrides=overrides,
        store=store,
        fetch_missing=fetch_missing,
    )
    nscf = advise(
        structure,
        code=code,
        hpc=hpc,
        overrides=_nscf_overrides(overrides),
        store=store,
        fetch_missing=fetch_missing,
    )
    dos = dos_settings(nscf.step.kpoints.occupations, dos_human, dos_llm)
    return DosAdvice(scf=scf, nscf=nscf, dos=dos)


def check_dos(advice: DosAdvice) -> CheckReport:
    """``advise_dos()``/``generate_dos()``'s boundary -- folds all three
    steps' own blocking reasons together. The nscf pass is checked with
    ``purpose="nscf"`` (``checks.py``'s scf-only named rule does not
    apply to it, per this module's own docstring)."""
    scf_report = check(advice.scf, purpose="scf")
    nscf_report = check(advice.nscf, purpose="nscf")
    dos_blocking = collect_blocked(advice.dos)
    blocking = scf_report.blocking + nscf_report.blocking + dos_blocking
    return CheckReport(blocking=blocking)


def generate_dos(
    advice: DosAdvice, report: CheckReport, *, ctx: SharedContext | None = None
) -> tuple[Step, ...]:
    """Delivery: three real ``Step``s (scf, nscf, dos), sharing one
    ``SharedContext`` -- goldilocks-core-design.md's own "phonon's 4
    steps / bands' 3 steps must share one prefix + outdir" requirement,
    exercised here for the first time. Raises ``AdviceIncomplete`` when
    ``report`` is not ``ok``, same contract as the single-step
    ``generate()`` -- checked once, directly, rather than re-derived per
    step: ``report.blocking`` is exactly the concatenation of each
    step's own blocking reasons (``check_dos``), so if the whole is
    empty every part is too, and each inner ``generate()`` call below is
    passed a trivially-``ok`` report rather than recomputing one."""
    if not report.ok:
        raise AdviceIncomplete(report)
    shared_ctx = ctx or default_shared_context()
    ok_report = CheckReport()
    scf_steps = generate(advice.scf, ok_report, ctx=shared_ctx)
    nscf_steps = generate(advice.nscf, ok_report, ctx=shared_ctx, purpose="nscf")
    dos_settings_value = DosSettings(
        disk_io=None,
        size=None,
        mem_per_proc=None,
        emin=advice.dos.value.emin,
        emax=advice.dos.value.emax,
        delta_e=advice.dos.value.delta_e,
        broadening=advice.dos.value.broadening,
        ngauss=advice.dos.value.ngauss,
    )
    dos_steps = write_qe_dos(dos_settings_value, shared_ctx)
    return scf_steps + nscf_steps + tuple(dos_steps)


def render_submission_dos(
    advice: DosAdvice,
    hpc: HpcProfile,
    code: str,
    ctx: SharedContext,
    steps: tuple[Step, ...],
) -> str:
    """One shared script for a ``dos`` task's three steps (v2 epic 9,
    #9, #28's delivery-layer routing). ``render_slurm_script`` takes
    exactly one ``JobDecision`` for its ``#SBATCH`` directives, but
    ``DosAdvice`` carries two independent ones (``scf``, ``nscf``) --
    the scf step's governs the shared allocation, a documented choice:
    it is normally the dominant cost, and ``dos.x`` itself is fast."""
    return render_slurm_script(
        hpc, advice.scf.step.resources.job.value, code, ctx, list(steps)
    )


def to_bundle_input_dos(
    advice: DosAdvice,
    steps: tuple[Step, ...],
    submission_script: str,
    ctx: SharedContext,
) -> BundleInput:
    """``service/_bundle.py``'s ``to_bundle_input``, for a ``dos``
    task. Pseudopotential file bytes come from ``advice.scf`` -- the
    same system-level decision as ``advice.nscf``'s, since only
    per-step settings differ between the two ``advise()`` calls."""
    return assemble_bundle_input(
        advice.scf.system.pseudo.metadata,
        steps,
        submission_script,
        ctx,
        advice.records(),
    )
