from __future__ import annotations

import shutil

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.advisors import magnetic_config as magnetic_config_module
from goldilocks_core.advisors.magnetic_config import (
    MagneticConfigHumanInput,
    magnetic_config,
)
from goldilocks_core.analysis.is_magnetic import is_magnetic
from goldilocks_core.resolution import Blocked, Provenance, Resolved, Unavailable

_IRON = Structure(Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]])
_SILICON = Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
_ROCK_SALT_FEO = Structure(
    Lattice.cubic(4.3), ["Fe", "O"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
)

_ENUMLIB_MISSING = (
    shutil.which("enum.x") is None and shutil.which("multienum.x") is None
)


def test_magnetic_structure_gets_spin_polarized_and_a_starting_magnetization() -> None:
    """A2 (stfc/goldilocks-core#177): spin_polarized=True must always come
    with a non-empty, non-zero starting_magnetization -- never implicit
    zero moments that relax to the non-magnetic solution. Without a
    z_valences (no pseudopotential chosen yet), every element gets
    aiida-quantumespresso's own flat default (0.1), not a guessed
    calibrated value."""
    state = magnetic_config(_IRON, is_magnetic(_IRON))

    assert state.ok
    assert state.value.spin_polarized is True
    assert state.value.magnetic_elements == ("Fe",)
    assert state.value.starting_magnetization == {"Fe": 0.1}


def test_starting_magnetization_uses_moment_target_when_z_valence_known() -> None:
    """Once a pseudopotential's z_valence is known, the fraction is
    aiida-quantumespresso's own target_moment / z_valence -- Fe's target is
    5 Bohr magnetons (magnetization.yaml), so a Fe pseudopotential with
    z_valence=16 gives 5/16, not the flat default."""
    state = magnetic_config(_IRON, is_magnetic(_IRON), z_valences={"Fe": 16.0})

    assert state.value.starting_magnetization == {"Fe": 5.0 / 16.0}


def test_starting_magnetization_is_keyed_by_site_label_not_element_symbol() -> None:
    """Regression for #32 (v2 epic 9, #9): a real structure's per-site
    labels are not always identical to its element symbols -- pymatgen's
    own CIF writer/reader already assigns distinct labels like
    'Fe0'/'Fe1' by default, not just AFM-relabeled structures. Before the
    fix, this heuristic default was keyed by element symbol
    (composition(structure).value.elements), which no longer matched
    generation/quantum_espresso/scf.py's label-keyed species_index once
    #27 made that label-keyed -- a bare KeyError on every bundled example
    structure. Two same-element, same-spin sites with distinct labels
    must both appear, with the same (positive) fraction."""
    two_iron_sites = Structure(
        Lattice.cubic(2.87),
        ["Fe", "Fe"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        labels=["Fe0", "Fe1"],
    )

    state = magnetic_config(two_iron_sites, is_magnetic(two_iron_sites))

    assert state.ok
    assert state.value.starting_magnetization == {"Fe0": 0.1, "Fe1": 0.1}


def test_starting_magnetization_covers_every_element_not_just_magnetic_candidates() -> (
    None
):
    """Matches aiida-quantumespresso's own behaviour: every kind gets an
    entry once spin-polarized, including non-magnetic-candidate species like
    O, so no single species sits at exactly zero and re-locks the spin
    symmetry the rest of the structure is trying to break."""
    iron_oxide = Structure(
        Lattice.cubic(4.0),
        ["Fe", "Fe", "O", "O", "O"],
        [[i / 5, 0, 0] for i in range(5)],
    )

    state = magnetic_config(iron_oxide, is_magnetic(iron_oxide))

    assert state.value.starting_magnetization == {"Fe": 0.1, "O": 0.1}


def test_unlisted_element_falls_back_to_the_flat_default_even_with_z_valence() -> None:
    """Cu has a filled d-shell as a neutral element and has no target in
    aiida's table -- it must not be scaled, even if a z_valence is
    supplied for it."""
    copper = Structure(Lattice.cubic(3.6), ["Cu"], [[0.0, 0.0, 0.0]])

    state = magnetic_config(
        copper,
        Resolved("magnetic", Provenance(source="heuristic")),
        z_valences={"Cu": 11.0},
    )

    assert state.value.starting_magnetization == {"Cu": 0.1}


def test_non_magnetic_structure_has_no_starting_magnetization() -> None:
    state = magnetic_config(_SILICON, is_magnetic(_SILICON))

    assert state.value.spin_polarized is False
    assert state.value.starting_magnetization is None
    assert state.value.magnetic_elements == ()


def test_unavailable_magnetism_defaults_to_non_magnetic_not_blocked() -> None:
    """Unavailable (the heuristic tried and could not tell) is not the same
    as Blocked (nothing upstream could even attempt an answer) -- there is
    still a safe, documented default to fall back on here."""
    state = magnetic_config(_IRON, Unavailable(reason="ambiguous oxidation state"))

    assert state.ok
    assert state.value.spin_polarized is False
    assert any(
        "could not be determined" in warning.message for warning in state.value.warnings
    )


def test_blocked_magnetism_propagates_as_blocked() -> None:
    state = magnetic_config(_IRON, Blocked(by="synthetic failure"))

    assert not state.ok
    assert state.status == "blocked"
    assert state.root_cause() == "synthetic failure"


def test_human_can_force_spin_polarization_on() -> None:
    state = magnetic_config(
        _SILICON,
        is_magnetic(_SILICON),
        human=MagneticConfigHumanInput(spin_polarized=True),
    )

    assert state.value.spin_polarized is True
    assert state.source == "human"


def test_giving_both_tot_and_starting_magnetization_does_not_error() -> None:
    """goldilocks-qe-pw-parameter-audit.md P0 finding #2: QE's own source
    (input.f90) does not error on this combination -- the design doc had
    briefly asserted it was a hard mutual exclusion elsewhere in itself and
    self-corrected. Both pass through unchanged."""
    state = magnetic_config(
        _IRON,
        is_magnetic(_IRON),
        human=MagneticConfigHumanInput(
            tot_magnetization=2.0, starting_magnetization={"Fe": 0.7}
        ),
    )

    assert state.ok
    assert state.value.tot_magnetization == 2.0
    assert state.value.starting_magnetization == {"Fe": 0.7}


def test_soc_enabled_on_a_magnetic_structure_keeps_the_magnetism() -> None:
    """A2b: enabling SOC must not silently discard magnetism -- both
    starting_magnetization and the angle1/angle2 direction QE's
    noncollinear mode needs must be present together."""
    state = magnetic_config(
        _IRON,
        is_magnetic(_IRON),
        human=MagneticConfigHumanInput(spin_orbit_coupling=True),
    )

    assert state.value.spin_polarized is True
    assert state.value.spin_orbit_enabled is True
    assert state.value.starting_magnetization == {"Fe": 0.1}
    assert state.value.angle1 == {"Fe": 0.0}
    assert state.value.angle2 == {"Fe": 0.0}


def test_soc_enabled_on_a_non_magnetic_structure_has_no_angles() -> None:
    """SOC without magnetism is only correct when the structure genuinely
    is non-magnetic -- confirmed by there being no is_magnetic override
    forcing it on here."""
    state = magnetic_config(
        _SILICON,
        is_magnetic(_SILICON),
        human=MagneticConfigHumanInput(spin_orbit_coupling=True),
    )

    assert state.value.spin_polarized is False
    assert state.value.spin_orbit_enabled is True
    assert state.value.angle1 is None
    assert state.value.angle2 is None


def test_soc_is_never_auto_enabled_even_when_needs_soc_says_yes() -> None:
    always_true = Resolved(True, is_magnetic(_IRON).provenance)

    state = magnetic_config(_IRON, is_magnetic(_IRON), needs_soc=always_true)

    assert state.value.spin_orbit_enabled is False
    assert any(
        "consider enabling" in warning.message for warning in state.value.warnings
    )


def test_relabeled_structure_is_the_identity_by_default() -> None:
    """AFM species-splitting (v2 epic 9, #9) is opt-in via
    ``magnetic_ordering="afm"`` -- without it, ``relabeled_structure`` stays
    the FM identity pass-through, same as before that feature existed."""
    state = magnetic_config(_IRON, is_magnetic(_IRON))

    assert state.value.relabeled_structure == _IRON


@pytest.mark.skipif(
    _ENUMLIB_MISSING,
    reason="needs the enumlib executables (enum.x, makeStr.py) on PATH",
)
def test_afm_ordering_finds_a_real_compensated_split() -> None:
    """v2 epic 9 (#9)'s AFM acceptance scenario: rock-salt FeO has one Fe
    sublattice by symmetry, so a real, compensated antiferromagnetic
    ordering needs distinguishing the two Fe sites -- proving
    relabeled_structure genuinely has more species than structure, not
    just that the type accepts it."""
    state = magnetic_config(
        _ROCK_SALT_FEO,
        is_magnetic(_ROCK_SALT_FEO),
        human=MagneticConfigHumanInput(magnetic_ordering="afm"),
    )

    assert state.ok
    facts = state.value
    assert len(set(facts.relabeled_structure.species)) > len(
        set(_ROCK_SALT_FEO.species)
    )
    fe_moments = [
        value
        for label, value in facts.starting_magnetization.items()
        if label.startswith("Fe")
    ]
    assert len(fe_moments) == 2
    assert any(moment > 0 for moment in fe_moments)
    assert any(moment < 0 for moment in fe_moments)
    assert any(
        warning.code == "magnetic.afm_ordering_applied" for warning in facts.warnings
    )


def test_afm_ordering_degrades_when_enumlib_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    state = magnetic_config(
        _ROCK_SALT_FEO,
        is_magnetic(_ROCK_SALT_FEO),
        human=MagneticConfigHumanInput(magnetic_ordering="afm"),
    )

    assert state.ok
    assert state.value.relabeled_structure == _ROCK_SALT_FEO
    assert any(
        warning.code == "magnetic.afm_ordering_unavailable"
        and "enumlib" in warning.message
        for warning in state.value.warnings
    )


def test_afm_ordering_degrades_for_oversized_structures() -> None:
    oversized = _ROCK_SALT_FEO * (3, 3, 3)  # 54 sites, over the ceiling

    state = magnetic_config(
        oversized,
        is_magnetic(oversized),
        human=MagneticConfigHumanInput(magnetic_ordering="afm"),
    )

    assert state.ok
    assert state.value.relabeled_structure == oversized
    assert any(
        warning.code == "magnetic.afm_ordering_unavailable"
        and "ceiling" in warning.message
        for warning in state.value.warnings
    )


def test_afm_ordering_degrades_when_no_compensated_ordering_is_found(
    monkeypatch,
) -> None:
    class _NoAfmCandidates:
        def __init__(self, *_args, **_kwargs) -> None:
            self.ordered_structures = [_IRON]
            self.ordered_structure_origins = ["fm"]

    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/enum.x")
    monkeypatch.setattr(
        magnetic_config_module, "MagneticStructureEnumerator", _NoAfmCandidates
    )

    state = magnetic_config(
        _IRON,
        is_magnetic(_IRON),
        human=MagneticConfigHumanInput(magnetic_ordering="afm"),
    )

    assert state.ok
    assert state.value.relabeled_structure == _IRON
    assert any(
        warning.code == "magnetic.afm_ordering_unavailable"
        and "no compensated" in warning.message
        for warning in state.value.warnings
    )
