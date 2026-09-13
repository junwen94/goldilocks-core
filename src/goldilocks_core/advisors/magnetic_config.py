"""magnetic_config: spin-polarization, starting magnetic moments, and
spin-orbit coupling, combined -- not three independent decisions that can
silently override each other.

New in v2 (v2 epic 5, #5). Fixes two real bugs found in v1's pipeline
(`advice/parameters.py`'s `_advise_magnetism`/`_advise_spin_orbit`,
`generation/qe/scf.py`'s `_spin_lines`), demonstrated here on this fresh
standalone advisor rather than by touching v1's still-running pipeline --
same policy as every prior epic in this rewrite (v1 stays untouched until
cutover, v2 epic 9). The two `tests/physics/test_dft_decisions.py` xfails
this issue names ("A2"/"A2b") exercise `compute()`, v1's own pipeline
end-to-end; they stay `xfail` because v1's `advice/parameters.py` and
`generation/qe/scf.py` genuinely have not changed. What this file actually
proves is in `tests/unit/test_advisors_magnetic_config.py`, against the same
physical scenarios (bulk Fe, with and without SOC).

- **A2** (stfc/goldilocks-core#177, still open upstream): v1 emits
  `nspin = 2` with every `starting_magnetization` implicitly zero, so QE
  relaxes to the non-magnetic solution while the run finishes, converges, and
  reports nothing obviously wrong. `spin_polarized=True` here always comes
  with a non-empty `starting_magnetization`.
- **A2b** (found while hardening `physics/` for v2 epic 1, #2): v1's
  `_spin_lines` checks `spin_orbit.enabled` first and returns before ever
  looking at `magnetism.spin_polarized` -- so enabling SOC on a structure
  independently advised as magnetic silently produces a non-magnetic
  noncollinear run. Here, spin-orbit coupling and magnetism are two
  independent fields on one result; enabling SOC never clears
  `starting_magnetization`, and a magnetic, SOC-enabled result also carries
  `angle1`/`angle2` (the moment direction QE's noncollinear mode needs) --
  SOC without magnetism only happens when the structure is genuinely
  non-magnetic, not because one flag silently overrode the other.

Also corrects a stricter-than-QE constraint the design doc's own audit found
elsewhere in itself (goldilocks-qe-pw-parameter-audit.md P0 finding #2):
giving both `tot_magnetization` and `starting_magnetization` does not error
in QE (`INPUT_PW`'s "do not specify both" is a recommendation, not an
enforced constraint -- verified against QE 7.3's `PW/src/input.f90`). Both
are accepted here as independent, unrelated overrides; this advisor never
treats supplying both as a blocking error.

`relabeled_structure` ships with AFM-ready shape but FM-only behaviour, per
goldilocks-core-design.md:3116-3136's "the duality needs to exist even before
species-splitting itself does, or every downstream advisor needs a second
signature later": it always equals the input `structure` for now. Actual
AFM species-splitting (producing distinct Fe1/Fe2 sites) is a later
extension, not this epic's job.

SOC stays opt-in, matching v1's own policy: `needs_soc` only ever produces a
warning recommending it be considered, never enables it automatically --
enabling SOC changes cost and setup, which is a human's call.
"""

from __future__ import annotations

from dataclasses import dataclass

from pymatgen.core import Structure

from goldilocks_core.analysis.composition import composition
from goldilocks_core.analysis.is_magnetic import Magnetism
from goldilocks_core.inputs.overrides import HumanInput, LlmInput
from goldilocks_core.resolution import Blocked, FieldState, Provenance, Resolved

DEFAULT_STARTING_MAGNETIZATION_FRACTION = 0.5
"""A generic, deliberately-nonzero starting guess (QE's `starting_magnetization`
is a fraction of a species' valence electron count, not a moment in Bohr
magnetons) -- large enough to break symmetry and let SCF find the real
magnetic solution instead of relaxing to zero (the A2 failure mode this file
exists to prevent), not a literature-calibrated value for any one element."""


@dataclass(frozen=True, slots=True)
class MagneticConfigFacts:
    relabeled_structure: Structure
    spin_polarized: bool
    magnetic_elements: tuple[str, ...]
    starting_magnetization: dict[str, float] | None
    tot_magnetization: float | None
    spin_orbit_enabled: bool
    angle1: dict[str, float] | None
    angle2: dict[str, float] | None
    warnings: tuple[str, ...] = ()


class MagneticConfigHumanInput(HumanInput):
    spin_polarized: bool | None = None
    spin_orbit_coupling: bool | None = None
    tot_magnetization: float | None = None
    starting_magnetization: dict[str, float] | None = None


class MagneticConfigLlmInput(LlmInput):
    spin_polarized: bool | None = None
    spin_orbit_coupling: bool | None = None


def magnetic_config(
    structure: Structure,
    is_magnetic: FieldState[Magnetism],
    needs_soc: FieldState[bool] | None = None,
    human: MagneticConfigHumanInput | None = None,
    llm: MagneticConfigLlmInput | None = None,
) -> FieldState[MagneticConfigFacts]:
    human = human or MagneticConfigHumanInput()
    llm = llm or MagneticConfigLlmInput()

    if isinstance(is_magnetic, Blocked):
        # Blocked (something upstream of is_magnetic failed) is not the same
        # as Unavailable (the heuristic tried and could not tell) -- only
        # the former leaves nothing safe to default to.
        return Blocked(by=is_magnetic)

    spin_polarized, source, warnings = _resolve_spin_polarized(is_magnetic, human, llm)

    magnetic_elements = _magnetic_elements(structure) if spin_polarized else ()

    starting_magnetization = None
    if spin_polarized:
        starting_magnetization = (
            dict(human.starting_magnetization)
            if human.starting_magnetization is not None
            else dict.fromkeys(
                magnetic_elements, DEFAULT_STARTING_MAGNETIZATION_FRACTION
            )
        )

    spin_orbit_enabled = human.spin_orbit_coupling is True
    needs_soc_resolved_true = needs_soc is not None and needs_soc.ok and needs_soc.value
    if not spin_orbit_enabled and needs_soc_resolved_true:
        warnings = (
            *warnings,
            "heavy, non-s-block elements are present; consider enabling "
            "spin_orbit_coupling (not enabled automatically -- it changes "
            "cost and setup).",
        )

    angle1 = angle2 = None
    if spin_orbit_enabled and spin_polarized:
        angle1 = dict.fromkeys(magnetic_elements, 0.0)
        angle2 = dict.fromkeys(magnetic_elements, 0.0)

    facts = MagneticConfigFacts(
        relabeled_structure=structure,
        spin_polarized=spin_polarized,
        magnetic_elements=magnetic_elements,
        starting_magnetization=starting_magnetization,
        tot_magnetization=human.tot_magnetization,
        spin_orbit_enabled=spin_orbit_enabled,
        angle1=angle1,
        angle2=angle2,
        warnings=warnings,
    )
    return Resolved(facts, Provenance(source=source))


def _resolve_spin_polarized(
    is_magnetic: FieldState[Magnetism],
    human: MagneticConfigHumanInput,
    llm: MagneticConfigLlmInput,
) -> tuple[bool, str, tuple[str, ...]]:
    if human.spin_polarized is not None:
        return human.spin_polarized, "human", ()
    ml_value: bool | None = None  # no ml model wired yet; stubbed until epic 11
    if ml_value is not None:
        return ml_value, "ml", ()
    if llm.spin_polarized is not None:
        return llm.spin_polarized, "llm", ()
    if is_magnetic.ok:
        return is_magnetic.value == "magnetic", "heuristic", ()
    return (
        False,
        "heuristic",
        (
            f"magnetism could not be determined ({is_magnetic.reason}); "
            "defaulted to non-magnetic.",
        ),
    )


def _magnetic_elements(structure: Structure) -> tuple[str, ...]:
    facts = composition(structure).value
    candidates = {*facts.transition_metals, *facts.lanthanides, *facts.actinides}
    return tuple(sorted(candidates))
