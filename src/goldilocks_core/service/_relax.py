"""``advise_relax()``/``check_relax()``/``generate_relax()``: the
``relax``/``vc-relax`` tasks' path (v2 epic 10, #10).

**Single-step, unlike ``dos``.** ``relax``/``vc-relax`` are one ``pw.x``
run each (``calculation='relax'``/``'vc-relax'`` does the electronic
scf loop *and* the ionic/cell optimization together) -- not a second
task needing ``service/_dos.py``'s "call the single-step pipeline
twice" shape. This module calls ``advise()``/``generate()`` exactly
once, the same as a plain ``scf_single_point`` task, and adds exactly
one more resolved decision (``relax``) on top -- mirroring how
``service/_dos.py``'s own ``DosAdvice`` adds ``dos`` on top of two
``Advice``s, just with one ``Advice`` instead of two.

**``RelaxOverrides`` is real request-validation plumbing, not a
task-specific side channel.** ``relax``/``relax_llm`` are read off
``overrides.step.relax`` (``capabilities.py``'s own ``relax`` branch,
v2 epic 10, #10) -- unlike ``DosHumanInput``/``DosLlmInput`` (v2 epic 9,
#9), which the epic 9 delivery-layer audit found are never reachable
from any transport at all. ``--set nstep=100`` (etc.) actually reaches
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pymatgen.core import Structure

from goldilocks_core.advisors.relax import (
    RelaxOptions,
    VcRelaxOptions,
    relax_settings,
)
from goldilocks_core.assets.store import AssetStore
from goldilocks_core.bundle import BundleInput
from goldilocks_core.checks import CheckReport
from goldilocks_core.inputs.hpc import HpcProfile
from goldilocks_core.resolution import FieldState
from goldilocks_core.service._advice import Advice, RunOverrides, warnings_from_records
from goldilocks_core.service._bundle import assemble_bundle_input, render_submission
from goldilocks_core.service._generate import AdviceIncomplete, generate
from goldilocks_core.service._pipeline import advise, check
from goldilocks_core.steps import SharedContext, Step, default_shared_context


@dataclass(frozen=True, slots=True)
class RelaxAdvice:
    calculation: Literal["relax", "vc-relax"]
    advice: Advice
    relax: FieldState[RelaxOptions | VcRelaxOptions]

    def field_states(self) -> tuple[FieldState[object], ...]:
        return (*self.advice.field_states(), self.relax)

    def records(self) -> dict[str, FieldState[object]]:
        merged = dict(self.advice.records())
        merged["relax"] = self.relax
        return merged

    def warnings(self) -> list[dict[str, object]]:
        return warnings_from_records(self.records())


def advise_relax(
    structure: Structure,
    *,
    calculation: Literal["relax", "vc-relax"],
    code: str = "quantum_espresso",
    hpc: HpcProfile,
    overrides: RunOverrides | None = None,
    store: AssetStore | None = None,
    fetch_missing: bool = False,
) -> RelaxAdvice:
    """Diagnosis, always returning -- same "never aborts as a whole"
    promise ``advise()`` makes. ``relax_settings`` reads
    ``advice.step.kpoints.convergence`` (for ``etot_conv_thr``) and
    ``advice.analysis.symmetry``/``.geometry`` (for vc-relax's
    ``cell_dofree`` hexagonal-2D default) off the very same ``Advice``
    this function just built -- no second structure-analysis pass."""
    overrides = overrides or RunOverrides()
    advice = advise(
        structure,
        code=code,
        hpc=hpc,
        overrides=overrides,
        store=store,
        fetch_missing=fetch_missing,
    )
    relax_overrides = overrides.step.relax
    relax = relax_settings(
        calculation,
        advice.step.kpoints.convergence,
        advice.analysis.symmetry,
        advice.analysis.geometry,
        relax_overrides.relax,
        relax_overrides.relax_llm,
    )
    return RelaxAdvice(calculation=calculation, advice=advice, relax=relax)


def check_relax(advice: RelaxAdvice) -> CheckReport:
    """``advise_relax()``/``generate_relax()``'s boundary. ``relax`` is
    passed straight through to ``check()``/``checks.check_all`` -- which
    folds its own ``Blocked``-ness into ``collect_blocked`` and runs the
    vc-relax ``ion_dynamics`` rule (``checks.py``'s own
    ``_vc_relax_requires_bfgs_ion_dynamics``) -- so there is no separate
    ``collect_blocked(advice.relax)`` call here to double-count it."""
    return check(advice.advice, purpose=advice.calculation, relax=advice.relax)


def generate_relax(
    advice: RelaxAdvice, report: CheckReport, *, ctx: SharedContext | None = None
) -> tuple[Step, ...]:
    """Delivery: one real ``Step``. Raises ``AdviceIncomplete`` when
    ``report`` is not ``ok``, same contract as the single-step
    ``generate()`` -- checked once, directly, rather than re-derived:
    the inner ``generate()`` call below is passed a trivially-``ok``
    report, matching ``service/_dos.py``'s own ``generate_dos``."""
    if not report.ok:
        raise AdviceIncomplete(report)
    return generate(
        advice.advice,
        CheckReport(),
        ctx=ctx or default_shared_context(),
        purpose=advice.calculation,
        relax=advice.relax.value,
    )


def render_submission_relax(
    advice: RelaxAdvice,
    hpc: HpcProfile,
    code: str,
    ctx: SharedContext,
    steps: tuple[Step, ...],
) -> str:
    return render_submission(advice.advice, hpc, code, ctx, steps)


def to_bundle_input_relax(
    advice: RelaxAdvice,
    steps: tuple[Step, ...],
    submission_script: str,
    ctx: SharedContext,
) -> BundleInput:
    return assemble_bundle_input(
        advice.advice.system.pseudo.metadata,
        steps,
        submission_script,
        ctx,
        advice.records(),
    )
