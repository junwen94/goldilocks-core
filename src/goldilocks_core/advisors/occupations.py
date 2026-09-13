"""occupations: fixed | smearing(type, degauss), from is_metal.

New in v2 (v2 epic 6, #6). The first of five per-step advisors
(`occupations -> k_sampling -> n_irr_k -> nbnd -> convergence`,
goldilocks-core-design.md:3239-3248) -- ``k_sampling`` takes this
module's ``degauss`` as an *explicit* input rather than assuming a fixed
sigma the way v1's dataset generation implicitly did.

Replaces v1's ``_advise_smearing`` (``advice/parameters.py:101-155``),
which branched on v1's more granular ``analysis["electronic_character"]``
(``"metal"``/``"likely_metal"``/``"insulator"``/unknown -- a
v1-only classification that no longer exists in v2;
``analysis/is_metal.py``'s ``FieldState[Metallicity]`` replaces it, more
conservatively: ``Unavailable`` rather than a guessed ``"likely_metal"``).

Metals get QE's cold smearing (``degauss=0.01`` Ry, matching v1's own
``METALLIC_SMEARING_WIDTH_RY``, ``advice/parameters.py:19``); a
*confirmed* non-metal gets ``fixed`` occupations. ``is_metal``
``Unavailable`` -- composition alone could not confirm metallic
character either way -- also gets smearing, not fixed. This is a
deliberate departure from v1's own mapping (v1's "unknown" bucket also
fell through to fixed): v1's "unknown" was genuinely no-signal-at-all,
but v2's ``is_metal`` is more conservative than v1's
``electronic_character`` (``analysis/is_metal.py``'s own docstring: it
only ever confidently asserts "metal", never a guessed "non_metal"), so
its ``Unavailable`` bucket is broader and now includes cases that look
metallic but are not fully confirmed. Smearing is the safe universal
default there: it tolerates a small or absent gap, whereas fixed
occupations assumes one and can fail to converge, or converge to the
wrong state, on an actual metal it misclassified. This also matches
common practice elsewhere (e.g. aiida-quantumespresso's own protocols
default to smearing rather than requiring a prior metal/insulator
classification).

``tetrahedra_opt`` (QE's third occupations option, appearing in
goldilocks-core-design.md's per-step pipeline sketch) is not decided by
this heuristic: no source in this codebase specifies when it should win
over smearing/fixed, so this module does not guess. It remains a valid
override value -- ``human``/``llm`` can request it directly -- just not
something the heuristic tier will ever choose on its own.

**Why `magnetic` is an input.** QE requires ``occupations='fixed'`` with
``nspin=2`` to also carry an *integer* ``tot_magnetization`` -- a
cross-parameter constraint between this advisor's answer and
``advisors/magnetic_config.py``'s. This module cannot itself validate
that number (it belongs to a sibling advisor, and checking it is a
numeric fact-check, not an occupations decision) -- it only attaches an
informational warning when the combination is heading that way, the
same not-my-call-to-block pattern ``analysis/needs_soc.py`` uses toward
``magnetic_config.py``. ``checks.py`` (this epic's other deliverable) is
where the actual integer check, and the hard fail if it does not hold,
happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from goldilocks_core.advisors.magnetic_config import MagneticConfigFacts
from goldilocks_core.analysis.is_metal import Metallicity
from goldilocks_core.inputs.overrides import HumanInput, LlmInput
from goldilocks_core.resolution import Blocked, FieldState, Provenance, Resolved

Occupations = Literal["fixed", "smearing", "tetrahedra_opt"]

_METALLIC_DEGAUSS = 0.01
"""Ry. Matches v1's METALLIC_SMEARING_WIDTH_RY (advice/parameters.py:19)."""

_FIXED_SPIN_POLARIZED_WARNING = (
    "occupations='fixed' with a spin-polarized system requires an integer "
    "tot_magnetization -- checks.py enforces this, not this advisor."
)


@dataclass(frozen=True, slots=True)
class OccupationsDecision:
    occupations: Occupations
    smearing_type: str | None
    degauss: float | None
    warnings: tuple[str, ...] = ()


class OccupationsHumanInput(HumanInput):
    occupations: Occupations | None = None
    smearing_type: str | None = None
    degauss: float | None = None


class OccupationsLlmInput(LlmInput):
    occupations: Occupations | None = None


def occupations(
    is_metal: FieldState[Metallicity],
    magnetic: FieldState[MagneticConfigFacts] | None = None,
    human: OccupationsHumanInput | None = None,
    llm: OccupationsLlmInput | None = None,
) -> FieldState[OccupationsDecision]:
    human = human or OccupationsHumanInput()
    llm = llm or OccupationsLlmInput()

    if human.occupations is not None:
        return Resolved(
            _decision_for(
                human.occupations, human.smearing_type, human.degauss, magnetic
            ),
            Provenance(source="human"),
        )
    ml_value: Occupations | None = None  # no ml model wired yet; stubbed until epic 11
    if ml_value is not None:
        return Resolved(
            _decision_for(ml_value, None, None, magnetic), Provenance(source="ml")
        )
    if llm.occupations is not None:
        return Resolved(
            _decision_for(llm.occupations, None, None, magnetic),
            Provenance(source="llm"),
        )

    if isinstance(is_metal, Blocked):
        return Blocked(by=is_metal)
    if is_metal.ok and is_metal.value == "non_metal":
        return Resolved(_fixed_decision(magnetic), Provenance(source="heuristic"))
    # Resolved("metal") or Unavailable ("heuristic tried, could not tell")
    # both land here: smearing is the safe universal default, since unlike
    # fixed it does not depend on the classification being right. Treating
    # "could not tell" as "assume non-metal" would risk exactly the
    # silent-wrong-answer failure mode fixed occupations produces on an
    # actual metal -- see this module's docstring.
    return Resolved(_smearing_decision(), Provenance(source="heuristic"))


def _decision_for(
    choice: Occupations,
    smearing_type: str | None,
    degauss: float | None,
    magnetic: FieldState[MagneticConfigFacts] | None,
) -> OccupationsDecision:
    if choice == "fixed":
        return _fixed_decision(magnetic)
    if choice == "smearing":
        return OccupationsDecision(
            occupations="smearing",
            smearing_type=smearing_type or "cold",
            degauss=degauss if degauss is not None else _METALLIC_DEGAUSS,
        )
    return OccupationsDecision(
        occupations="tetrahedra_opt", smearing_type=None, degauss=None
    )


def _fixed_decision(
    magnetic: FieldState[MagneticConfigFacts] | None,
) -> OccupationsDecision:
    warnings: tuple[str, ...] = ()
    if magnetic is not None and magnetic.ok and magnetic.value.spin_polarized:
        warnings = (_FIXED_SPIN_POLARIZED_WARNING,)
    return OccupationsDecision(
        occupations="fixed", smearing_type=None, degauss=None, warnings=warnings
    )


def _smearing_decision() -> OccupationsDecision:
    return OccupationsDecision(
        occupations="smearing", smearing_type="cold", degauss=_METALLIC_DEGAUSS
    )
