"""nbnd: number of Kohn-Sham bands to compute, from nelec and occupations.

New in v2 (v2 epic 6, #6). Fourth of five per-step advisors
(``occupations -> k_sampling -> n_irr_k -> nbnd -> convergence``,
goldilocks-core-design.md:3239-3248). v1 never had this advisor at all
-- ``nbnd`` was only ever computed inside ``resource_estimate``, with
nothing upstream of it deciding what to request from QE itself
(goldilocks-core-design.md's own note: "nbnd only computed in
resource_estimate, no advisor produces it").

**Formula is QE's own official default** (``INPUT_PW.txt``'s ``nbnd``
entry, operator-confirmed 2026-09-14 against the current PWscf docs,
not derived or guessed): for an insulator, ``nbnd = nelec / 2`` (the
number of valence bands); for a metal, 20% more, with a floor of at
least 4 more than the insulator count -- i.e.
``max(1.2 * nelec/2, nelec/2 + 4)``, rounded up. QE's own docs also note
explicitly that in a spin-polarized calculation it is the number of
*k-points* that doubles internally, not the number of bands -- ``nbnd``
itself is not doubled for ``nspin=2``, and this module does not double
it either.

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

**Per-purpose band boosting is a known, real gap, not addressed here.**
A ``nscf``/``bands``/``dos`` step commonly wants materially more empty
bands than a plain ``scf`` step, to actually cover the energy window a
DOS or band structure plot needs -- but no source in this codebase
specifies a multiplier or a target energy window, so this heuristic
applies the same QE-official base formula regardless of ``purpose``.
``purpose`` is still accepted in the signature (signature-stable for
whenever that rule exists), it is simply not yet consulted -- the same
treatment ``advisors/k_sampling.py`` gives its own currently-unused
``occupations`` input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from goldilocks_core.advisors.magnetic_config import MagneticConfigFacts
from goldilocks_core.advisors.occupations import OccupationsDecision
from goldilocks_core.inputs.overrides import HumanInput, LlmInput
from goldilocks_core.resolution import (
    Blocked,
    FieldState,
    Provenance,
    Resolved,
    blocked_by,
)

_METAL_LIKE_OCCUPATIONS = frozenset({"smearing", "tetrahedra_opt"})


@dataclass(frozen=True, slots=True)
class NbndDecision:
    nbnd: int
    warnings: tuple[str, ...] = ()


class NbndHumanInput(HumanInput):
    nbnd: int | None = None
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

    is_metal_like = occupations.value.occupations in _METAL_LIKE_OCCUPATIONS
    base = _formula(nelec, is_metal_like)

    if human.extra_bands is not None:
        extra, source = human.extra_bands, "human"
    elif llm.extra_bands is not None:
        extra, source = llm.extra_bands, "llm"
    else:
        extra, source = 0, "heuristic"

    warnings = _spin_note(magnetic)
    decision = NbndDecision(nbnd=base + extra, warnings=warnings)
    return Resolved(decision, Provenance(source=source))


def _formula(nelec: float, is_metal_like: bool) -> int:
    valence_bands = nelec / 2
    if not is_metal_like:
        return max(1, math.ceil(valence_bands))
    return max(1, math.ceil(max(valence_bands * 1.2, valence_bands + 4)))


def _spin_note(magnetic: FieldState[MagneticConfigFacts] | None) -> tuple[str, ...]:
    if magnetic is not None and magnetic.ok and magnetic.value.spin_polarized:
        return (
            "nspin=2 doubles the number of k-points QE solves internally, "
            "not the number of bands -- nbnd here is per spin channel, "
            "unchanged from the non-spin-polarized formula.",
        )
    return ()
