"""``Step``/``SharedContext``: the rendered-output layer of generation.

New in v2 (v2 epic 7, #7). ``step_settings.py`` (v2 epic 6, #6) built the
middle layer (``PlannedStep -> StepSettings -> Step``) but deliberately
stopped one layer short: its own module docstring names ``Step`` as
"this epic's other deliverable" -- i.e. epic 6's job was per-step
*settings*, not the rendered artifact those settings turn into. This
module is that missing third layer, per
goldilocks-core-design.md:1152-1171.

Lives top-level, not under a ``contracts/`` package: the same reasoning
as ``resolution.py``/``step_settings.py`` applies -- v1's centralized
``contracts`` hub caused import cycles and organizational drift and was
deleted (see ``resolution.py``'s docstring), and ``Step``/
``SharedContext`` are used across three different packages
(``generation/``, ``submission/``, ``bundle.py``), so they belong next
to ``step_settings.py`` rather than inside any one of those.

**``PlannedStep``/``plan.py`` moved out to its own module in v2 epic 9,
#9** once ``dos`` became a second task that actually needs step
sequencing (this module's own docstring predicted exactly this:
"add ``PlannedStep``/``plan.py`` once a second task actually needs step
sequencing"). Not merged into this file: ``Step``/``SharedContext`` are
the *rendered* layer (per-program executable/args/files), while
``PlannedStep`` is one layer up (which programs run, in what order,
before any settings exist yet) -- distinct enough concerns that
``plan.py``'s own docstring explains its scope separately.

``Step.args`` is a tuple, not the design doc's literal ``list[str]``,
matching every other rendered-decision dataclass in this codebase
(``ResourceEstimate``, ``JobDecision``, ...): frozen dataclasses here
use immutable field types throughout, not just an immutable container.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Step:
    """One executable invocation, fully rendered -- no decisions left.

    ``args`` carries code-specific flags (QE's ``-npool``/``-ndiag``/
    ``-in``; VASP would carry none, per goldilocks-core-design.md:1244-1247)
    so ``submission/slurm.py`` can stay ignorant of which code produced
    the step it is scheduling.
    """

    name: str
    executable: str
    args: tuple[str, ...] = ()
    files: dict[str, str] = field(default_factory=dict)
    stdout: str | None = None
    workdir: str | None = None


@dataclass(frozen=True, slots=True)
class SharedContext:
    """What a multi-step task's steps must agree on to find each other's
    output on disk.

    Per goldilocks-core-design.md:2850: "phonon's 4 steps / bands' 3
    steps must share one ``prefix`` + ``outdir``, or a later step can't
    find the previous step's wavefunctions/charge density" -- and
    "``Step`` currently has ``files``/``workdir`` but nothing guarantees
    those two values agree" is exactly the gap this type closes. Only
    one step exists in this epic (scf), so nothing yet exercises the
    "shared across steps" property; the type exists now so the next task
    that adds a second step is not the one that also has to invent this.

    ``prefix``/``outdir``/``pseudo_dir`` replace v1's unconditional
    literals (``generation/qe/scf.py:134-135``: hardcoded
    ``pseudo_dir = './pseudo'``, ``outdir = './out'``, no ``prefix`` at
    all) -- constructing a ``SharedContext`` is now the one place those
    values are chosen, instead of being buried inside the writer with no
    record of why.
    """

    prefix: str
    outdir: str
    pseudo_dir: str


DEFAULT_PREFIX = "pwscf"
"""QE's own ``&CONTROL`` default for ``prefix`` (``INPUT_PW.txt``) -- used
by ``default_shared_context`` below, not invented here."""

DEFAULT_OUTDIR = "./out"
DEFAULT_PSEUDO_DIR = "./pseudo"
"""Same values v1 hardcoded (``generation/qe/scf.py:134-135``), now reached
through an explicit, named default rather than a literal inside the writer."""


def default_shared_context() -> SharedContext:
    """The heuristic-tier ``SharedContext``: package defaults, no operator
    input yet. A full ``human > ml > llm > heuristic`` cascade for these
    three fields is not built here -- nothing in this codebase needs to
    override them yet, and adding override plumbing with no caller would
    be speculative. This function exists so that gap has one obvious
    place to grow into, instead of every call site re-typing the
    defaults inline.
    """

    return SharedContext(
        prefix=DEFAULT_PREFIX, outdir=DEFAULT_OUTDIR, pseudo_dir=DEFAULT_PSEUDO_DIR
    )
