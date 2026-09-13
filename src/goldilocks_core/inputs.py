"""The two override bags a caller can populate, plus the task/code
identifiers that replace v1's ``CalculationIntent``.

Top-level, not under any one ``analysis/``/``advisors/`` module: every
advisor receives a ``HumanInput``/``LlmInput``, so neither belongs to
one domain any more than ``resolution.py``'s primitives do
(goldilocks-core-design.md S13's "named top-level files, not utils/"
reasoning applies the same way here).

Only what v2 epic 3 has a concrete use for lands here.
``StructureSource``/``HpcProfile`` are real, larger pieces of plumbing
(structure normalization, HPC profile loading) that v1 already does
well -- they're ported near-verbatim in v2 epic 3's sibling, "port the
four verbatim v1 edges" (issue #4), not redesigned here. Building a
premature pydantic version now, before any advisor exists to consume
it, risks getting the shape wrong with no way to notice (the same
lesson goldilocks-ml's serving.py design draws about interfaces with
no consumer).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from goldilocks_core.types import CalcTask, CodeName


class HumanInput(BaseModel):
    """Operator-provided overrides for one step. Empty here: each
    advisor declares the specific fields it accepts by subclassing
    (see ``tests/unit/test_resolution.py``'s worked example) -- there
    is no shared shape to declare until a second advisor exists to
    compare against the first.

    ``extra="forbid"`` deliberately: a typo'd override field should
    fail loudly at construction, not be silently ignored the way an
    unrecognised dict key would be.
    """

    model_config = ConfigDict(extra="forbid")


class LlmInput(HumanInput):
    """Agent-provided overrides. Identical shape to ``HumanInput`` by
    construction (it *is* ``HumanInput``, not a hand-kept copy) --
    which bag a caller populated is the source label, so there is no
    separate ``source`` field for the two to disagree about."""


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
