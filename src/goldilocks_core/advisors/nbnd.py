"""nbnd: number of Kohn-Sham bands to compute, from nelec and occupations.

New in v2 (v2 epic 6, #6). Fourth of five per-step advisors
(``occupations -> k_sampling -> n_irr_k -> nbnd -> convergence``,
goldilocks-core-design.md:3239-3248). v1 never had this advisor at all
-- ``nbnd`` was only ever computed inside ``resource_estimate``, with
nothing upstream of it deciding what to request from QE itself
(goldilocks-core-design.md's own note: "nbnd only computed in
resource_estimate, no advisor produces it").

**nbnd is pw.x's own default** (#162): a literal port of QE 7.3
``PW/src/setup.f90:326-345``, which pw.x applies when ``nbnd`` is not
given. It starts from the bands every spin channel needs,
``MAX(NINT(nelec/2), NINT(nelup), NINT(neldw))``; under smearing or
tetrahedra it pads that to ``MAX(NINT(1.2*nelec/2), NINT(1.2*nelup),
NINT(1.2*neldw), base + 4)``, 20% more with at least 4 more; for a
noncollinear run it doubles the padded count. ``nelup``/``neldw`` come
from ``Modules/electrons_base.f90:376-428`` (``set_nelup_neldw``):
``INT(nelec+1)/2`` and the rest without ``tot_magnetization``,
``(INT(nelec) +/- tot_magnetization)/2`` with an integer one, so a
pinned moment sizes the majority channel. ``NINT`` rounds half away
from zero, unlike Python's ``round()``. INPUT_PW's ``nbnd`` text leaves
out the ``nelup``/``neldw`` terms, the rounding and the order of padding
and doubling; the code follows ``setup.f90``.

The spin inputs are the ones the writer renders: the spin mode
(``MagneticConfigFacts.spin_mode``, #146) and, for a collinear run only
(nspin=2, the one mode pw.x accepts it in), the occupations decision's
``tot_magnetization``, the one rendered moment (#138). In a collinear
spin-polarized run QE doubles the number of k-points it solves, not the
number of bands: ``nbnd`` is per spin channel. A noncollinear (spin-orbit)
band is a two-component spinor holding one electron, so ``setup.f90``
doubles the per-spin count after padding; an odd electron count still
gets an even band count.

**Uses ``occupations`` (not ``is_metal``) to decide the metal branch.**
``advisors/occupations.py`` has already turned ``is_metal`` (plus any
overrides) into an actual ``fixed``/``smearing``/``tetrahedra_opt``
decision; using that decision here -- ``smearing`` or ``tetrahedra_opt``
means "treated as needing empty states above a Fermi level" -- keeps
this advisor consistent with what will actually run, instead of
re-deriving metallicity independently and risking disagreement with a
sibling advisor's own overrides.

**``nelec`` is a required, plain input, not a ``FieldState``.** No
advisor in this codebase computes total valence electron count yet --
that requires knowing which pseudopotential (and therefore which
``z_valence`` per element) was chosen, which is
``advisors/pseudo_selection.py``'s job, and nothing wires that output
into this one yet (the same reason ``magnetic_config.py`` takes an
optional ``z_valences`` rather than computing it itself). There is no
sensible fallback value for a missing electron count, so this is not
made optional with a guessed default -- a caller that does not yet know
``nelec`` should not call this advisor yet.

**Per-purpose band boosting is not directly implemented, but the DOS
task already gets it as a side effect, worth knowing about.** A
``nscf``/``bands``/``dos`` step commonly wants materially more empty
bands than a plain ``scf`` step, to actually cover the energy window a
DOS or band structure plot needs -- this module has no explicit
multiplier or target-energy-window rule keyed on ``purpose`` (it is
still accepted in the signature, signature-stable for whenever such a
rule exists, but not yet consulted -- the same treatment
``advisors/k_sampling.py`` gives its own currently-unused
``occupations`` input). In practice, ``service/_dos.py``'s nscf step
already forces ``occupations="tetrahedra_opt"`` (aiida-quantumespresso's
own DOS protocol default, chosen for integration-grid quality, not for
band count) -- and ``tetrahedra_opt`` is one of this module's own
``_METAL_LIKE_OCCUPATIONS``, so ``nbnd()`` called for that nscf step
takes the metal-like padding (20% more per spin channel, at least 4
more) regardless of the material's real classification, incidentally
giving the nscf step real empty bands an insulator's own scf step gets
none of (the bundled ``Si.cif``, nelec 32: scf ``nbnd=16``, exactly
``nelec/2``; nscf ``nbnd=20``). This is a genuine, working mechanism,
just an accidental one -- nobody chose 20%/+4 *for* band-gap
visibility, it is only there because of how ``occupations`` classifies
``tetrahedra_opt``. It has not been checked whether that margin is
actually enough for every case (a narrow-gap material, or a DOS/band
plot that wants to show several eV of conduction states) -- a future
``purpose``-aware rule, if one is ever added, should treat this as the
thing it would be replacing, not something already solved on purpose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import Field

from goldilocks_core.advisors.magnetic_config import MagneticConfigFacts, SpinMode
from goldilocks_core.advisors.occupations import OccupationsDecision
from goldilocks_core.inputs.overrides import HumanInput, LlmInput
from goldilocks_core.resolution import (
    Blocked,
    FieldState,
    Provenance,
    Resolved,
    Unavailable,
    Warning,
    blocked_by,
)

_METAL_LIKE_OCCUPATIONS = frozenset({"smearing", "tetrahedra_opt"})

_INTEGER_TOLERANCE = 1e-8
"""QE's ``eps8``, the tolerance ``set_nelup_neldw`` treats a charge or a
moment as an integer within."""

SPIN_NOTE = Warning(
    code="nbnd.spin_channel_note",
    level="info",
    category="nbnd",
    message=(
        "nspin=2 doubles the number of k-points QE solves internally, "
        "not the number of bands -- nbnd here is per spin channel, with "
        "room for the majority channel's electrons (pw.x's own default, "
        "QE setup.f90)."
    ),
)

NONCOLLINEAR_NOTE = Warning(
    code="nbnd.noncollinear_bands_doubled",
    level="info",
    category="nbnd",
    message=(
        "noncollinear/spin-orbit bands are spinors holding one electron "
        "each, not two, so nbnd here is twice pw.x's default for the same "
        "electron count and occupations without spin (QE setup.f90: "
        "padded per spin channel, then doubled)."
    ),
)

WARNING_CATALOGUE = (SPIN_NOTE, NONCOLLINEAR_NOTE)
"""Every warning code this module can emit -- ``capabilities.py``'s
``warnings[]`` catalogue aggregates one of these tuples per advisor."""


@dataclass(frozen=True, slots=True)
class NbndDecision:
    nbnd: int
    warnings: tuple[Warning, ...] = ()


class NbndHumanInput(HumanInput):
    """``nbnd`` (#35, v2 epic 9, #9) must be a positive integer at
    construction -- before this, ``nbnd=0``/negative was accepted and
    written straight into the generated ``&SYSTEM`` card as a
    ``Resolved`` success. ``extra_bands`` stays unconstrained here
    (it is a signed adjustment to the heuristic base, legitimately
    negative) -- the combined total is checked in ``nbnd()`` below,
    where the heuristic base is actually known."""

    nbnd: int | None = Field(default=None, gt=0)
    extra_bands: int | None = None


class NbndLlmInput(LlmInput):
    extra_bands: int | None = None


def nbnd(
    nelec: float,
    occupations: FieldState[OccupationsDecision],
    purpose: str = "scf",
    magnetic: FieldState[MagneticConfigFacts] | None = None,
    human: NbndHumanInput | None = None,
    llm: NbndLlmInput | None = None,
) -> FieldState[NbndDecision]:
    human = human or NbndHumanInput()
    llm = llm or NbndLlmInput()

    if human.nbnd is not None:
        return Resolved(NbndDecision(nbnd=human.nbnd), Provenance(source="human"))

    if isinstance(occupations, Blocked):
        return Blocked(by=occupations)
    if not occupations.ok:
        return Blocked(by=blocked_by(occupations))

    spin_mode, tot_magnetization = _spin_inputs(occupations.value, magnetic)
    base = _qe_default_nbnd(
        nelec,
        occupations.value.occupations in _METAL_LIKE_OCCUPATIONS,
        spin_mode,
        tot_magnetization,
    )

    if human.extra_bands is not None:
        extra, source = human.extra_bands, "human"
    elif llm.extra_bands is not None:
        extra, source = llm.extra_bands, "llm"
    else:
        extra, source = 0, "heuristic"

    total = base + extra
    if total <= 0:
        # #35 (v2 epic 9, #9): extra_bands is a legitimately-signed
        # adjustment (Field(gt=0) on it alone would reject valid
        # negative reductions), but the combined total must still be a
        # physically meaningful band count -- before this, a
        # sufficiently negative extra_bands silently produced a
        # zero/negative &SYSTEM nbnd card, reported as a resolved
        # success.
        return Unavailable(
            reason=(
                f"heuristic nbnd={base} + extra_bands={extra} = {total}, which "
                "is not a valid number of bands (must be positive)"
            )
        )

    warnings = _spin_note(spin_mode)
    decision = NbndDecision(nbnd=total, warnings=warnings)
    return Resolved(decision, Provenance(source=source))


def _spin_inputs(
    occupations: OccupationsDecision,
    magnetic: FieldState[MagneticConfigFacts] | None,
) -> tuple[SpinMode, float | None]:
    """QE's spin inputs as the writer renders them: the spin mode of
    ``magnetic`` (``none`` when there is no resolved one) and the
    ``tot_magnetization`` of ``occupations``, written for a collinear run
    only (module docstring)."""
    spin_mode = (
        magnetic.value.spin_mode if magnetic is not None and magnetic.ok else "none"
    )
    tot_magnetization = (
        occupations.tot_magnetization if spin_mode == "collinear" else None
    )
    return spin_mode, tot_magnetization


def _qe_default_nbnd(
    nelec: float,
    metal_like: bool,
    spin_mode: SpinMode,
    tot_magnetization: float | None,
) -> int:
    """pw.x's default ``nbnd`` (QE 7.3 ``PW/src/setup.f90:326-345``):
    the bands each spin channel needs, padded under smearing or
    tetrahedra, doubled for a noncollinear run."""
    nelup, neldw = _spin_channels(nelec, tot_magnetization)
    bands = max(_nint(nelec / 2), _nint(nelup), _nint(neldw))
    if metal_like:
        bands = max(
            _nint(1.2 * nelec / 2), _nint(1.2 * nelup), _nint(1.2 * neldw), bands + 4
        )
    if spin_mode == "noncollinear":
        bands *= 2
    return bands


def _spin_channels(
    nelec: float, tot_magnetization: float | None
) -> tuple[float, float]:
    """``nelup``/``neldw`` as QE 7.3 ``set_nelup_neldw`` sets them
    (``Modules/electrons_base.f90:376-428``), ``INT`` truncation
    included."""
    integer_charge = abs(nelec - _nint(nelec)) < _INTEGER_TOLERANCE
    if tot_magnetization is None:
        if not integer_charge:
            return nelec / 2, nelec / 2
        nelup = int(nelec + 1) // 2
        return nelup, nelec - nelup
    integer_moment = (
        abs(tot_magnetization - _nint(tot_magnetization)) < _INTEGER_TOLERANCE
    )
    charge = int(nelec) if integer_charge and integer_moment else nelec
    return (charge + tot_magnetization) / 2, (charge - tot_magnetization) / 2


def _nint(value: float) -> int:
    """Fortran ``NINT``: the nearest integer, halves away from zero."""
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def _spin_note(spin_mode: SpinMode) -> tuple[Warning, ...]:
    if spin_mode == "noncollinear":
        return (NONCOLLINEAR_NOTE,)
    if spin_mode == "collinear":
        return (SPIN_NOTE,)
    return ()
