"""service: the v2 orchestration layer, wiring epics 1-7 into one pipeline.

New in v2 (v2 epic 8, #8); no v1 precedent -- this is not a port of
``runtime/service.py``'s ``Service``/``Runtime``/``Dispatcher`` machinery
(that pipeline still runs on v1's ``legacy_analysis``/``request.py`` and
is left untouched for the CLI/HTTP/MCP modules this epic also replaces
to stop calling; it becomes dead code for v2 epic 9 to delete).

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
- ``_system.py`` -- system-level advisors (epic 5), mirrors
  ``system_settings.SystemSettings``'s own field list one layer up
- ``_step.py`` -- per-step advisors (epics 6-7), mirrors
  ``step_settings.PwSettings``'s own field list one layer up
- ``_advice.py`` -- ``Advice``/``RunOverrides``, composing the three
  phases above
- ``_pipeline.py`` -- ``advise()``/``check()``
- ``_generate.py`` -- ``generate()``, the hard-fail boundary
- ``_bundle.py`` -- submission-script rendering + bundle assembly
- ``_dos.py`` -- ``dos`` task multi-step path (epic 9), reusing the
  single-step phases above twice rather than a fourth phase of its own
- ``_relax.py`` -- ``relax``/``vc-relax`` task path (epic 10), reusing
  the single-step phases above once (single-step tasks, unlike ``dos``)
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
from goldilocks_core.service._step import RelaxOverrides, StepAdvice, StepOverrides
from goldilocks_core.service._step_kpoints import ElectronicStepAdvice, KpointsOverrides
from goldilocks_core.service._step_resources import (
    ResourceOverrides,
    ResourceStepAdvice,
)
from goldilocks_core.service._system import SystemAdvice, SystemOverrides

__all__ = [
    "Advice",
    "AdviceIncomplete",
    "AnalysisFacts",
    "AnalysisOverrides",
    "DosAdvice",
    "ElectronicStepAdvice",
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
