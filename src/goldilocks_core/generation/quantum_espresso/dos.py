"""write_qe_dos: render one ``dos.x`` ``Step`` from resolved settings.

New in v2 (v2 epic 9, #9) -- the second Quantum ESPRESSO writer this
codebase has, and the concrete proof that ``StepSettings`` really is a
tagged union with different producers per program, not just a type that
happens to type-check (``step_settings.py``'s own acceptance bar,
goldilocks-implementation-plan.md's five-scenario table, scenario 5).

**No ``SystemSettings`` parameter, unlike ``write_qe_scf``.** ``dos.x``
reads a previous ``pw.x`` step's saved wavefunctions off ``prefix``/
``outdir`` alone -- it has no chemistry, cutoffs, or pseudopotential
awareness of its own (``step_settings.py``'s own ``DosSettings``
docstring: "no k_sampling/n_irr_k/nbnd/convergence of its own"; the same
reasoning extends one level up to the system-level facts ``pw.x``
needs and ``dos.x`` does not).

**No parallelisation args**, unlike ``write_qe_scf``'s ``-npool``:
``DosSettings`` carries no ``ParallelisationDecision`` field at all
(``step_settings.py``), so there is nothing to translate into a flag
here -- not an oversight, a reflection of that type's own shape.
"""

from __future__ import annotations

from goldilocks_core.generation.errors import GenerationError
from goldilocks_core.generation.quantum_espresso.namelists import render_namelist
from goldilocks_core.step_settings import DosSettings
from goldilocks_core.steps import SharedContext, Step


def write_qe_dos(step: DosSettings, ctx: SharedContext) -> list[Step]:
    for name, value in (("delta_e", step.delta_e), ("ngauss", step.ngauss)):
        if value is None:
            raise GenerationError(f"DosSettings.{name} is required to generate dos.in")

    keywords: dict[str, object] = {
        "prefix": ctx.prefix,
        "outdir": ctx.outdir,
        "deltae": step.delta_e,
        "ngauss": step.ngauss,
    }
    if step.emin is not None:
        keywords["emin"] = step.emin
    if step.emax is not None:
        keywords["emax"] = step.emax
    if step.broadening is not None:
        keywords["degauss"] = step.broadening

    content = render_namelist(keywords, binary="dos")

    return [
        Step(
            name="dos",
            executable="dos.x",
            args=("-in", "dos.in"),
            files={"dos.in": content},
            stdout="dos.out",
        )
    ]
