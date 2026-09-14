"""``advise()``/``check()``: the "diagnosis" half of the v2 orchestrator
(v2 epic 8, #8).

Two-phase split, per goldilocks-core-design.md's own promise: ``advise()``
always returns (diagnosis is always available, degrading field-by-field,
never aborting as a whole); ``generate()`` (``service/_generate.py``)
hard-fails via ``checks.check_all`` if anything needed is
``unavailable``/``blocked`` (delivery has preconditions). CLI's
``explain`` command is exactly ``advise()``; ``run`` is
``advise()`` -> ``generate()`` -> submission script -> bundle.

Only the one task epics 1-7 actually built a generation writer for
(``scf_single_point`` on Quantum ESPRESSO) is wired here. Task
sequencing (``PlannedStep``/``plan.py``) does not exist yet either
(``steps.py``'s own docstring: "add once a second task actually needs
step sequencing") -- ``_step.py`` calls the per-step advisors and this
module calls ``generate()``'s writer directly, once, rather than through
a task-expansion layer that would have no second caller to prove it
against.
"""

from __future__ import annotations

from pymatgen.core import Structure

from goldilocks_core.assets.store import AssetStore
from goldilocks_core.checks import CheckReport, check_all
from goldilocks_core.inputs.hpc import HpcProfile
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


def check(advice: Advice) -> CheckReport:
    """The ``advise()``/``generate()`` boundary -- see ``checks.py``."""
    return check_all(
        *advice.field_states(),
        occupations=advice.step.kpoints.occupations,
        magnetic=advice.system.magnetic,
        purpose="scf",
    )
