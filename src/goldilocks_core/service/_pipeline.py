"""``advise()``/``check()``: the "diagnosis" half of the v2 orchestrator
(v2 epic 8, #8).

Two-phase split, per goldilocks-core-design.md's own promise: ``advise()``
always returns (diagnosis is always available, degrading field-by-field,
never aborting as a whole); ``generate()`` (``service/_generate.py``)
hard-fails via ``checks.check_all`` if anything needed is
``unavailable``/``blocked`` (delivery has preconditions). CLI's
``explain`` command is exactly ``advise()``; ``run`` is
``advise()`` -> ``generate()`` -> submission script -> bundle.

Only the single-step task (``scf_single_point`` on Quantum ESPRESSO) is
wired here -- ``_step.py`` calls the per-step advisors and this module
calls ``generate()``'s writer directly, once. The multi-step ``dos``
task (v2 epic 9, #9's ``plan.py``/``PlannedStep``) is deliberately not
folded in here: ``service/_dos.py``'s ``advise_dos()``/``generate_dos()``
call ``advise()``/``generate()`` (this module's own two functions) twice
instead, rather than rewriting this module to be "one step or many" for
a second task that needs the same single-step machinery, just called
more than once. See ``_dos.py``'s own docstring for why.
"""

from __future__ import annotations

from pymatgen.core import Structure

from goldilocks_core.advisors.relax import RelaxOptions, VcRelaxOptions
from goldilocks_core.assets.store import AssetStore
from goldilocks_core.checks import CheckReport, check_all
from goldilocks_core.inputs.hpc import HpcProfile
from goldilocks_core.resolution import FieldState
from goldilocks_core.service._advice import Advice, RunOverrides
from goldilocks_core.service._analysis import analyze
from goldilocks_core.service._step import step_advice
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
    """Diagnosis: always returns, per the "diagnosis is always available"
    promise. Never raises for a scientifically-incomplete request --
    only a genuinely malformed ``structure`` (already rejected earlier,
    by ``inputs.structure.normalize_structure``, before this is called)
    would do that.

    ``fetch_missing`` governs only the one real download this pipeline
    can trigger (installing the chosen pseudopotential table) -- P8's
    "no sneaky downloads" -- and still degrades to ``Unavailable``
    rather than raising when it is ``False`` and the table is missing.
    """
    overrides = overrides or RunOverrides()
    analysis = analyze(structure, overrides.analysis)
    system = system_advice(
        structure,
        analysis,
        overrides.system,
        store=store,
        fetch_missing=fetch_missing,
    )
    step = step_advice(structure, code, analysis, system, hpc, overrides.step)
    return Advice(structure=structure, analysis=analysis, system=system, step=step)


def check(
    advice: Advice,
    *,
    purpose: str = "scf",
    relax: FieldState[RelaxOptions | VcRelaxOptions] | None = None,
) -> CheckReport:
    """The ``advise()``/``generate()`` boundary -- see ``checks.py``.

    ``purpose`` (v2 epic 9, #9): the "``occupations='fixed'`` needs an
    integer ``tot_magnetization``" rule applies to every purpose that
    runs its own scf loop (``checks.py``'s own ``_SCF_LIKE_PURPOSES``) --
    an nscf step reads a prior scf step's already-converged
    density/spin and does not re-derive this constraint. Every existing
    caller is an scf step and keeps the previous default unchanged;
    ``service/_dos.py``'s nscf pass is the first caller to pass
    ``purpose="nscf"``.

    ``relax`` (v2 epic 10, #10): only ever passed by ``service/_relax.py``,
    which does not have a plain ``Advice`` field to fold it into --
    every other caller leaves it ``None``.

    ``job``/``parallel``: always read straight off ``advice`` (every
    ``Advice`` has both) -- the ``npool``-must-divide-``ntasks`` rule
    (``checks.py``'s own ``_npool_must_divide_ntasks``) applies
    regardless of ``purpose``. ``geometry`` (#44): same "always read
    straight off ``advice``" treatment -- every ``Advice`` has an
    ``analysis.geometry``, needed by ``_fix_bottom_layers_requires_2d_geometry``
    regardless of ``purpose`` (a non-relax caller never sets
    ``relax.fix_bottom_layers`` in the first place, so the rule is a
    no-op for them).
    """
    return check_all(
        *advice.field_states(),
        occupations=advice.step.kpoints.occupations,
        magnetic=advice.system.magnetic,
        relax=relax,
        geometry=advice.analysis.geometry,
        job=advice.step.resources.job,
        parallel=advice.step.resources.parallelisation,
        purpose=purpose,
    )
