"""``Task``: which code and which calculation task to run.

Moved here from the top-level ``inputs.py`` module (v2 epic 2, #3) once epic 3
(#4) needed ``inputs/`` to be a package -- see ``inputs/overrides.py``'s
docstring for why. This is also where the design doc's target tree
(goldilocks-core-design.md S8) puts ``task.py``: "task name -> which codes
support it." That lookup has no consumer yet (only one code, quantum_espresso,
exists), so only the ``Task`` type itself lands here for now.
"""

from __future__ import annotations

from pydantic import BaseModel

from goldilocks_core.types import CalcTask, CodeName


class Task(BaseModel):
    """Replaces v1's ``CalculationIntent``, which named a class after
    "intent" while its only real field was ``task``
    (``calculation.py:28,30``) -- "intent" also collided with the
    unrelated everyday sense of user intent used elsewhere in the
    codebase. ``code`` and ``task`` are independent identifiers here,
    not bundled into one "intent" concept.
    """

    code: CodeName = "quantum_espresso"
    task: CalcTask = "scf_single_point"
