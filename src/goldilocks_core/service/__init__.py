"""service: the v2 orchestration layer, wiring epics 1-7 into one pipeline.

New in v2 (v2 epic 8, #8); no v1 precedent -- this is not a port of
``runtime/service.py``'s ``Service``/``Runtime``/``Dispatcher`` machinery
(that pipeline ran on v1's ``legacy_analysis``/``request.py`` and was
left untouched only until the CLI/HTTP/MCP modules this epic replaces
stopped calling it; it became dead code that v2 epic 9 then deleted,
commit d64f46b).

Nothing before this module called ``analysis/`` -> ``advisors/`` ->
``checks.check_all`` -> ``generation/quantum_espresso`` ->
``submission/slurm`` -> ``bundle.py`` together. Both ``checks.py`` (v2
epic 6, #6) and ``advisors/pseudo_selection.py`` (v2 epic 5, #5) say so
in their own docstrings ("a real orchestrator assembling
system_settings/per_step... deliberately left as a gap for whichever
future epic adds it"; "not `PseudoResolution`'s full `materialize()`
orchestration... that belongs with `generation/`/`bundle.py`") -- this
package is that orchestrator.

**A package, not one module**: this project's own import-surface ceiling
(``scripts/check_complexity.py``, default 12 origins/24 symbols per
module) cannot fit an orchestrator that touches every analysis/advisor
module into one file. Split by pipeline phase instead, matching
``analysis:``/``advisors:``'s own package convention:

- ``_analysis.py`` -- structure-only facts (epic 4)
- ``_pseudo.py`` -- pseudopotential table selection + asset-store
  materialization (the one place this pipeline touches the filesystem)
- ``_magnetic.py`` -- the simulated cell (spin, spin-orbit coupling and the
  applied magnetic ordering), decided once, then table selection and the
  starting moments in the order they depend on each other
- ``_system.py`` -- system-level advisors (epic 5), mirrors
  ``system_settings.SystemSettings``'s own field list one layer up
- ``_step.py`` -- per-step advisors (epics 6-7), mirrors
  ``step_settings.PwSettings``'s own field list one layer up, and one
  step's request overrides (``StepOverrides``, routed per step, #154)
- ``_step_protocol.py`` -- ``StepProtocol``, a task's own defaults for one
  of its steps, a tier below the request (the dos nscf protocol, #174)
- ``_advice.py`` -- ``Advice``/``RunOverrides``, composing the three
  phases above; ``RunOverrides`` holds one task's overrides, per step
- ``_pipeline.py`` -- ``advise()``/``check()``, and ``advise_steps``,
  which advises every ``pw.x`` step of one task with one shared job
- ``_generate.py`` -- ``generate()``, the hard-fail boundary
- ``_bundle.py`` -- submission-script rendering + bundle assembly
- ``_dos.py`` -- ``dos`` task multi-step path (epic 9): ``advise_steps``
  for its scf and nscf steps, rather than a fourth phase of its own
- ``_relax.py`` -- ``relax``/``vc-relax`` task path (epic 10):
  ``advise_steps`` for its one step (single-step tasks, unlike ``dos``)
- ``_magnetic_orderings.py`` -- ``list_magnetic_orderings`` (#87),
  deliberately outside the ``advise()``/``generate()`` pipeline above: a
  listing capability a caller consults *before* deciding which magnetic
  ordering to generate input files for, not a phase of one calculation

This ``__init__.py`` only re-exports; per ``check_complexity.py``'s own
logic, a package's pure-export ``__init__.py`` is exempt from the
ceiling (each re-exported name is still resolved back to its real
origin at every consumer), so the public API stays flat
(``from goldilocks_core.service import advise, generate, ...``) without
forcing every caller to know the internal phase split above.
"""

from __future__ import annotations

from goldilocks_core.service._advice import Advice, RunOverrides
from goldilocks_core.service._analysis import AnalysisFacts, AnalysisOverrides
from goldilocks_core.service._bundle import render_submission, to_bundle_input
from goldilocks_core.service._dos import (
    DosAdvice,
    advise_dos,
    check_dos,
    generate_dos,
    render_submission_dos,
    to_bundle_input_dos,
)
from goldilocks_core.service._generate import AdviceIncomplete, generate
from goldilocks_core.service._magnetic_orderings import (
    MagneticOrderingCandidate,
    MagneticOrderingsReport,
    list_magnetic_orderings,
    report_to_json,
)
from goldilocks_core.service._pipeline import advise, check
from goldilocks_core.service._pseudo import PseudoAdvice
from goldilocks_core.service._relax import (
    RelaxAdvice,
    advise_relax,
    check_relax,
    generate_relax,
    render_submission_relax,
    to_bundle_input_relax,
)
from goldilocks_core.service._step import (
    OVERRIDES_WARNING_CATALOGUE,
    DosOverrides,
    RelaxOverrides,
    StepAdvice,
    StepOverrides,
    applies_to_multiple_steps,
)
from goldilocks_core.service._step_kpoints import ElectronicStepAdvice, KpointsOverrides
from goldilocks_core.service._step_resources import (
    JobOverrides,
    ResourceOverrides,
    ResourceStepAdvice,
)
from goldilocks_core.service._system import SystemAdvice, SystemOverrides

__all__ = [
    "OVERRIDES_WARNING_CATALOGUE",
    "Advice",
    "AdviceIncomplete",
    "AnalysisFacts",
    "AnalysisOverrides",
    "DosAdvice",
    "DosOverrides",
    "ElectronicStepAdvice",
    "JobOverrides",
    "KpointsOverrides",
    "MagneticOrderingCandidate",
    "MagneticOrderingsReport",
    "PseudoAdvice",
    "RelaxAdvice",
    "RelaxOverrides",
    "ResourceOverrides",
    "ResourceStepAdvice",
    "RunOverrides",
    "StepAdvice",
    "StepOverrides",
    "SystemAdvice",
    "SystemOverrides",
    "advise",
    "advise_dos",
    "advise_relax",
    "applies_to_multiple_steps",
    "check",
    "check_dos",
    "check_relax",
    "generate",
    "generate_dos",
    "generate_relax",
    "list_magnetic_orderings",
    "render_submission",
    "render_submission_dos",
    "render_submission_relax",
    "report_to_json",
    "to_bundle_input",
    "to_bundle_input_dos",
    "to_bundle_input_relax",
]
