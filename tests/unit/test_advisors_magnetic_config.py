from __future__ import annotations

import shutil

import numpy as np
import pytest
from numpy.linalg import LinAlgError
from pymatgen.core import Lattice, Species, Structure
from support import magnetic_decision

from goldilocks_core.advisors import magnetic_config as magnetic_config_module
from goldilocks_core.advisors.magnetic_config import (
    MagneticConfigHumanInput,
    OrderingRecord,
    enumerate_magnetic_orderings,
    magnetic_cell,
    magnetic_config,
    magnetic_elements_in,
)
from goldilocks_core.advisors.magnetic_enumeration import enumlib_installed
from goldilocks_core.analysis.is_magnetic import is_magnetic
from goldilocks_core.resolution import Blocked, Provenance, Resolved, Unavailable

_IRON = Structure(Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]])
_SILICON = Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
_ROCK_SALT_FEO = Structure(
    Lattice.cubic(4.3), ["Fe", "O"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
)

_ENUMLIB_MISSING = not enumlib_installed()


def _on_path(*names: str):
    """A ``shutil.which`` that finds exactly ``names``."""
    return lambda name: f"/usr/bin/{name}" if name in names else None


def _enumerating(*orderings: tuple[Structure, str]):
    """An ``enumerate_orderings`` stub returning ``(structure, origin)``
    pairs -- the worker process is out of reach of a monkeypatch."""
    return lambda *_args, **_kwargs: orderings


def _failing_with(error: BaseException):
    def enumerate_orderings(*_args, **_kwargs):
        raise error

    return enumerate_orderings


def test_magnetic_structure_gets_spin_polarized_and_a_starting_magnetization() -> None:
    """A2 (stfc/goldilocks-core#177): spin_polarized=True must always come
    with a non-empty, non-zero starting_magnetization -- never implicit
    zero moments that relax to the non-magnetic solution. Without a
    z_valences (no pseudopotential chosen yet), every element gets
    aiida-quantumespresso's own flat default (0.1), not a guessed
    calibrated value."""
    state = magnetic_decision(_IRON, is_magnetic(_IRON))

    assert state.ok
    assert state.value.spin_polarized is True
    assert state.value.magnetic_elements == ("Fe",)
    assert state.value.starting_magnetization == {"Fe": 0.1}


def test_starting_magnetization_uses_moment_target_when_z_valence_known() -> None:
    """Once a pseudopotential's z_valence is known, the fraction is
    aiida-quantumespresso's own target_moment / z_valence -- Fe's target is
    5 Bohr magnetons (magnetization.yaml), so a Fe pseudopotential with
    z_valence=16 gives 5/16, not the flat default."""
    state = magnetic_decision(_IRON, is_magnetic(_IRON), z_valences={"Fe": 16.0})

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

    state = magnetic_decision(two_iron_sites, is_magnetic(two_iron_sites))

    assert state.ok
    assert state.value.starting_magnetization == {"Fe0": 0.1, "Fe1": 0.1}


def test_starting_magnetization_override_with_an_unknown_label_is_blocked() -> None:
    """Regression for #46: a human override keyed by bare element symbol
    ('Fe') on a structure whose real labels are 'Fe0'/'Fe1' (pymatgen's
    own CIF default) used to be accepted verbatim and only crash later,
    deep in generation, with a raw KeyError -- checks.check_all saw
    nothing wrong. Must now be Blocked here instead, with a message
    naming both the bad key and the real labels."""
    two_iron_sites = Structure(
        Lattice.cubic(2.87),
        ["Fe", "Fe"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        labels=["Fe0", "Fe1"],
    )

    state = magnetic_cell(
        two_iron_sites,
        is_magnetic(two_iron_sites),
        human=MagneticConfigHumanInput(starting_magnetization={"Fe": 0.5}),
    )

    assert isinstance(state, Blocked)
    assert "Fe" in state.root_cause()
    assert "Fe0" in state.root_cause()
    assert "Fe1" in state.root_cause()


def test_starting_magnetization_override_with_real_labels_is_accepted() -> None:
    two_iron_sites = Structure(
        Lattice.cubic(2.87),
        ["Fe", "Fe"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        labels=["Fe0", "Fe1"],
    )

    state = magnetic_decision(
        two_iron_sites,
        is_magnetic(two_iron_sites),
        human=MagneticConfigHumanInput(
            starting_magnetization={"Fe0": 0.5, "Fe1": -0.5}
        ),
    )

    assert state.ok
    assert state.value.starting_magnetization == {"Fe0": 0.5, "Fe1": -0.5}


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

    state = magnetic_decision(iron_oxide, is_magnetic(iron_oxide))

    assert state.value.starting_magnetization == {"Fe": 0.1, "O": 0.1}


def test_unlisted_element_falls_back_to_the_flat_default_even_with_z_valence() -> None:
    """Cu has a filled d-shell as a neutral element and has no target in
    aiida's table -- it must not be scaled, even if a z_valence is
    supplied for it."""
    copper = Structure(Lattice.cubic(3.6), ["Cu"], [[0.0, 0.0, 0.0]])

    state = magnetic_decision(
        copper,
        Resolved("magnetic", Provenance(source="heuristic")),
        z_valences={"Cu": 11.0},
    )

    assert state.value.starting_magnetization == {"Cu": 0.1}


def test_non_magnetic_structure_has_no_starting_magnetization() -> None:
    state = magnetic_decision(_SILICON, is_magnetic(_SILICON))

    assert state.value.spin_polarized is False
    assert state.value.starting_magnetization is None
    assert state.value.magnetic_elements == ()


def test_unavailable_magnetism_defaults_to_non_magnetic_not_blocked() -> None:
    """Unavailable (the heuristic tried and could not tell) is not the same
    as Blocked (nothing upstream could even attempt an answer) -- there is
    still a safe, documented default to fall back on here."""
    state = magnetic_cell(_IRON, Unavailable(reason="ambiguous oxidation state"))

    assert state.ok
    assert state.value.spin_polarized is False
    assert any(
        "could not be determined" in warning.message for warning in state.value.warnings
    )


def test_blocked_magnetism_propagates_as_blocked() -> None:
    state = magnetic_cell(_IRON, Blocked(by="synthetic failure"))

    assert not state.ok
    assert state.status == "blocked"
    assert state.root_cause() == "synthetic failure"


def test_human_can_force_spin_polarization_on() -> None:
    state = magnetic_cell(
        _SILICON,
        is_magnetic(_SILICON),
        human=MagneticConfigHumanInput(spin_polarized=True),
    )

    assert state.value.spin_polarized is True
    assert state.source == "human"


def test_tot_magnetization_alone_implies_spin_polarized() -> None:
    """Regression for #34 (v2 epic 9, #9): a human-supplied
    tot_magnetization (or starting_magnetization) with no spin_polarized
    override at all is an unambiguous request for a spin-polarized
    calculation -- before this, it was silently dropped whenever
    spin_polarized otherwise resolved False (e.g. a non-magnetic
    structure's heuristic default)."""
    state = magnetic_decision(
        _SILICON,
        is_magnetic(_SILICON),
        human=MagneticConfigHumanInput(tot_magnetization=2.0),
    )

    assert state.ok
    assert state.value.spin_polarized is True
    assert state.source == "human"
    assert state.value.tot_magnetization == 2.0


def test_explicit_spin_polarized_false_wins_over_magnetization_and_warns() -> None:
    """An explicit spin_polarized=False is a human override too and must
    win outright over a simultaneously-given tot_magnetization -- but
    silently dropping the latter with zero signal is still wrong, so
    this warns instead."""
    state = magnetic_cell(
        _SILICON,
        is_magnetic(_SILICON),
        human=MagneticConfigHumanInput(spin_polarized=False, tot_magnetization=2.0),
    )

    assert state.ok
    assert state.value.spin_polarized is False
    assert any(
        warning.code == "magnetic.magnetization_ignored_spin_polarized_false"
        for warning in state.value.warnings
    )


def test_giving_both_tot_and_starting_magnetization_does_not_error() -> None:
    """goldilocks-qe-pw-parameter-audit.md P0 finding #2: QE's own source
    (input.f90) does not error on this combination -- the design doc had
    briefly asserted it was a hard mutual exclusion elsewhere in itself and
    self-corrected. Both pass through unchanged."""
    state = magnetic_decision(
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
    state = magnetic_decision(
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
    state = magnetic_decision(
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

    state = magnetic_decision(_IRON, is_magnetic(_IRON), needs_soc=always_true)

    assert state.value.spin_orbit_enabled is False
    assert any(
        "consider enabling" in warning.message for warning in state.value.warnings
    )


def test_the_cell_is_the_input_structure_by_default() -> None:
    """AFM species-splitting (v2 epic 9, #9) is opt-in via
    ``magnetic_ordering="afm"`` -- without it, the cell stays the FM
    identity pass-through, same as before that feature existed."""
    state = magnetic_cell(_IRON, is_magnetic(_IRON))

    assert state.value.structure is _IRON
    assert state.value.ordering == OrderingRecord(requested="fm", applied="fm")
    assert np.allclose(state.value.cell_in_input_basis, np.eye(3))


def _afm_rock_salt(structure: Structure) -> Structure:
    """A spin-decorated 4-site cell of a 2-site rock-salt input, its two
    Ni up and down along the doubled c, as the enumerator returns one."""
    supercell = structure * (1, 1, 2)
    spins = iter((5.0, -5.0))
    return Structure(
        supercell.lattice,
        [
            Species("Ni", spin=next(spins))
            if site.specie.symbol == "Ni"
            else site.specie
            for site in supercell
        ],
        supercell.frac_coords,
    )


_ROCK_SALT_NIO = Structure(
    Lattice.from_parameters(2.95, 2.95, 2.95, 60, 60, 60),
    ["Ni", "O"],
    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)
_AFM = MagneticConfigHumanInput(magnetic_ordering="afm")


@pytest.fixture
def afm_candidate(monkeypatch) -> Structure:
    """The one antiferromagnetic candidate a stubbed enumeration returns
    for ``_ROCK_SALT_NIO`` (the real enumerator runs in a worker process,
    out of a monkeypatch's reach)."""
    candidate = _afm_rock_salt(_ROCK_SALT_NIO)
    monkeypatch.setattr(shutil, "which", _on_path("enum.x", "makeStr.py"))
    monkeypatch.setattr(
        magnetic_config_module, "enumerate_orderings", _enumerating((candidate, "afm"))
    )
    return candidate


def test_an_applied_ordering_is_the_cell_with_its_basis_in_the_input(
    afm_candidate,
) -> None:
    """#128: the applied candidate is the cell everything downstream reads,
    labelled by sublattice, with the ordering asked for and applied, and
    its lattice in the input's basis -- here a doubling along c."""
    state = magnetic_cell(_ROCK_SALT_NIO, is_magnetic(_ROCK_SALT_NIO), _AFM)

    cell = state.value
    assert len(cell.structure) == 4
    assert sorted({site.label for site in cell.structure}) == ["Ni1", "Ni2", "O"]
    assert cell.ordering == OrderingRecord(requested="afm", applied="afm")
    assert np.allclose(cell.cell_in_input_basis, np.diag([1.0, 1.0, 2.0]))
    assert np.allclose(
        cell.structure.lattice.matrix,
        np.array(cell.cell_in_input_basis) @ _ROCK_SALT_NIO.lattice.matrix,
    )


def test_seeds_are_set_on_the_applied_cell_once_z_valence_is_known(
    afm_candidate,
) -> None:
    """The second step scales the cell's sublattice seeds by the selected
    pseudopotential's z_valence: Ni's 5 Bohr magneton target over 18
    electrons, with opposite signs on the two Ni sublattices."""
    cell = magnetic_cell(_ROCK_SALT_NIO, is_magnetic(_ROCK_SALT_NIO), _AFM)

    state = magnetic_config(cell, z_valences={"Ni": 18.0, "O": 6.0}, human=_AFM)

    assert state.value.cell is cell.value.structure
    assert state.value.ordering == cell.value.ordering
    assert state.value.starting_magnetization == {
        "Ni1": 5.0 / 18.0,
        "Ni2": -5.0 / 18.0,
        "O": 0.1,
    }
    assert state.source == cell.source


def test_a_human_starting_magnetization_is_checked_against_the_applied_cell(
    afm_candidate,
) -> None:
    """The labels a starting_magnetization may use are the written cell's
    (Ni1/Ni2), not the uploaded structure's (Ni)."""
    split = MagneticConfigHumanInput(
        magnetic_ordering="afm", starting_magnetization={"Ni1": 0.5, "Ni2": -0.5}
    )
    unsplit = MagneticConfigHumanInput(
        magnetic_ordering="afm", starting_magnetization={"Ni": 0.5}
    )

    accepted = magnetic_cell(_ROCK_SALT_NIO, is_magnetic(_ROCK_SALT_NIO), split)
    refused = magnetic_cell(_ROCK_SALT_NIO, is_magnetic(_ROCK_SALT_NIO), unsplit)

    assert accepted.ok
    assert isinstance(refused, Blocked)
    assert "Ni1" in refused.root_cause()
    assert "Ni2" in refused.root_cause()


def test_a_blocked_cell_blocks_the_starting_moments() -> None:
    state = magnetic_config(Blocked(by="unknown site label"), z_valences={"Fe": 16.0})

    assert isinstance(state, Blocked)
    assert state.root_cause() == "unknown site label"


@pytest.mark.skipif(
    _ENUMLIB_MISSING,
    reason="needs the enumlib executables (enum.x, makeStr.py) on PATH",
)
def test_afm_ordering_finds_a_real_compensated_split() -> None:
    """v2 epic 9 (#9)'s AFM acceptance scenario: rock-salt FeO has one Fe
    sublattice by symmetry, so a real, compensated antiferromagnetic
    ordering needs distinguishing the two Fe sites -- proving
    the cell genuinely has more species than structure, not just that the
    type accepts it."""
    state = magnetic_decision(
        _ROCK_SALT_FEO,
        is_magnetic(_ROCK_SALT_FEO),
        human=MagneticConfigHumanInput(magnetic_ordering="afm"),
    )

    assert state.ok
    facts = state.value
    assert len(set(facts.cell.species)) > len(set(_ROCK_SALT_FEO.species))
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

    state = magnetic_cell(
        _ROCK_SALT_FEO,
        is_magnetic(_ROCK_SALT_FEO),
        human=MagneticConfigHumanInput(magnetic_ordering="afm"),
    )

    assert state.ok
    assert state.value.structure == _ROCK_SALT_FEO
    assert state.value.ordering == OrderingRecord(requested="afm", applied="fm")
    assert any(
        warning.code == "magnetic.afm_ordering_unavailable"
        and "enumlib" in warning.message
        for warning in state.value.warnings
    )


def test_afm_ordering_degrades_for_oversized_structures() -> None:
    oversized = _ROCK_SALT_FEO * (3, 3, 3)  # 54 sites, over the ceiling

    state = magnetic_cell(
        oversized,
        is_magnetic(oversized),
        human=MagneticConfigHumanInput(magnetic_ordering="afm"),
    )

    assert state.ok
    assert state.value.structure == oversized
    assert any(
        warning.code == "magnetic.afm_ordering_unavailable"
        and "ceiling" in warning.message
        for warning in state.value.warnings
    )


def test_afm_ordering_degrades_when_no_compensated_ordering_is_found(
    monkeypatch,
) -> None:
    monkeypatch.setattr(shutil, "which", _on_path("enum.x", "makeStr.py"))
    monkeypatch.setattr(
        magnetic_config_module, "enumerate_orderings", _enumerating((_IRON, "fm"))
    )

    state = magnetic_cell(
        _IRON,
        is_magnetic(_IRON),
        human=MagneticConfigHumanInput(magnetic_ordering="afm"),
    )

    assert state.ok
    assert state.value.structure == _IRON
    assert any(
        warning.code == "magnetic.afm_ordering_unavailable"
        and "no compensated" in warning.message
        for warning in state.value.warnings
    )


def test_enumerate_magnetic_orderings_is_just_fm_when_enumlib_is_missing(
    monkeypatch,
) -> None:
    """#87's listing capability degrades the same way ``_attempt_afm_relabeling``
    does: no enumlib means no antiferromagnetic candidates, not an error."""
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    candidates, reason = enumerate_magnetic_orderings(
        _ROCK_SALT_FEO, magnetic_elements_in(_ROCK_SALT_FEO)
    )

    assert candidates == (("fm", _ROCK_SALT_FEO),)
    assert reason is not None and "enumlib" in reason


def test_enumerate_magnetic_orderings_never_picks_a_winner(monkeypatch) -> None:
    """Unlike ``_attempt_afm_relabeling``, listing every candidate is the
    whole point -- #87's mMACE ranking (or a human) picks the winner, not
    this function."""
    monkeypatch.setattr(shutil, "which", _on_path("enum.x", "makeStr.py"))
    monkeypatch.setattr(
        magnetic_config_module,
        "enumerate_orderings",
        _enumerating((_ROCK_SALT_FEO.copy(), "afm"), (_ROCK_SALT_FEO.copy(), "afm")),
    )

    candidates, reason = enumerate_magnetic_orderings(
        _ROCK_SALT_FEO, magnetic_elements_in(_ROCK_SALT_FEO)
    )

    labels = [label for label, _structure in candidates]
    assert labels == ["fm", "afm-1", "afm-2"]
    assert reason is None


@pytest.mark.parametrize(
    "installed",
    [(), ("enum.x",), ("multienum.x",), ("makeStr.py",), ("makestr.x", "makeStr.x")],
)
def test_afm_enumeration_needs_both_enumlib_executables(
    monkeypatch, installed: tuple[str, ...]
) -> None:
    """pymatgen refuses to enumerate without enum.x/multienum.x *and* a
    structure maker (#136), so a missing one is a reason, not a call."""
    monkeypatch.setattr(shutil, "which", _on_path(*installed))
    monkeypatch.setattr(
        magnetic_config_module,
        "enumerate_orderings",
        _failing_with(AssertionError("enumerated without both executables")),
    )

    candidates, reason = enumerate_magnetic_orderings(
        _ROCK_SALT_FEO, magnetic_elements_in(_ROCK_SALT_FEO)
    )

    assert candidates == (("fm", _ROCK_SALT_FEO),)
    assert reason is not None
    assert "enumlib" in reason
    assert "makeStr.py" in reason


@pytest.mark.parametrize(
    "installed",
    [
        ("enum.x", "makeStr.py"),
        ("multienum.x", "makestr.x"),
        ("enum.x", "makeStr.x"),
    ],
)
def test_any_name_pymatgen_looks_up_counts_as_installed(
    monkeypatch, installed: tuple[str, ...]
) -> None:
    monkeypatch.setattr(shutil, "which", _on_path(*installed))
    monkeypatch.setattr(
        magnetic_config_module,
        "enumerate_orderings",
        _enumerating((_ROCK_SALT_FEO.copy(), "afm")),
    )

    candidates, reason = enumerate_magnetic_orderings(
        _ROCK_SALT_FEO, magnetic_elements_in(_ROCK_SALT_FEO)
    )

    assert [label for label, _structure in candidates] == ["fm", "afm-1"]
    assert reason is None


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError(2, "No such file or directory", "vasp.3"),
        LinAlgError("Singular matrix"),
        RuntimeError("This shouldn't happen, found neighbors=[]"),
    ],
)
def test_an_enumeration_error_that_is_not_an_enumlib_outcome_propagates(
    monkeypatch, error: Exception
) -> None:
    """#136: overlapping enumerations sharing one cwd failed with exactly
    these errors (LinAlgError subclasses ValueError), which came back as a
    plausible ferromagnetic-only listing quoting a host path."""
    monkeypatch.setattr(shutil, "which", _on_path("enum.x", "makeStr.py"))
    monkeypatch.setattr(
        magnetic_config_module, "enumerate_orderings", _failing_with(error)
    )

    with pytest.raises(type(error)):
        enumerate_magnetic_orderings(
            _ROCK_SALT_FEO, magnetic_elements_in(_ROCK_SALT_FEO)
        )


@pytest.mark.parametrize(
    "message",
    [
        "Unable to enumerate",
        "/home/operator/goldilocks/tmpq3x9: Too many magnetic sites",
    ],
)
def test_an_unenumerable_structure_gets_a_fixed_reason_without_the_error_text(
    monkeypatch, message: str
) -> None:
    monkeypatch.setattr(shutil, "which", _on_path("enum.x", "makeStr.py"))
    monkeypatch.setattr(
        magnetic_config_module,
        "enumerate_orderings",
        _failing_with(ValueError(message)),
    )

    candidates, reason = enumerate_magnetic_orderings(
        _ROCK_SALT_FEO, magnetic_elements_in(_ROCK_SALT_FEO)
    )

    assert candidates == (("fm", _ROCK_SALT_FEO),)
    assert reason == (
        "pymatgen could not enumerate magnetic orderings for this structure"
    )


def test_an_enumlib_timeout_gets_a_fixed_reason_without_the_error_text(
    monkeypatch,
) -> None:
    monkeypatch.setattr(shutil, "which", _on_path("enum.x", "makeStr.py"))
    monkeypatch.setattr(
        magnetic_config_module,
        "enumerate_orderings",
        _failing_with(TimeoutError("Enumeration took more than timeout 5 minutes")),
    )

    state = magnetic_cell(
        _ROCK_SALT_FEO,
        is_magnetic(_ROCK_SALT_FEO),
        human=MagneticConfigHumanInput(magnetic_ordering="afm"),
    )

    assert state.value.structure == _ROCK_SALT_FEO
    [warning] = [
        warning
        for warning in state.value.warnings
        if warning.code == "magnetic.afm_ordering_unavailable"
    ]
    assert warning.message == (
        "enumlib did not finish within its time limit; kept ferromagnetic order."
    )


@pytest.mark.skipif(
    _ENUMLIB_MISSING,
    reason="needs the enumlib executables (enum.x, makeStr.py) on PATH",
)
def test_enumerate_magnetic_orderings_finds_the_same_split_relabeling_does() -> None:
    candidates, reason = enumerate_magnetic_orderings(
        _ROCK_SALT_FEO, magnetic_elements_in(_ROCK_SALT_FEO)
    )

    assert reason is None
    assert candidates[0][0] == "fm"
    assert candidates[0][1] == _ROCK_SALT_FEO
    afm_candidates = candidates[1:]
    assert len(afm_candidates) > 0
    for label, candidate in afm_candidates:
        assert label.startswith("afm-")
        assert len(set(candidate.species)) > len(set(_ROCK_SALT_FEO.species))
