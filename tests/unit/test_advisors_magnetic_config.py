from __future__ import annotations

import shutil

import numpy as np
import pytest
from numpy.linalg import LinAlgError
from pydantic import ValidationError
from pymatgen.core import Lattice, Species, Structure
from pymatgen.electronic_structure.core import Magmom
from support import magnetic_decision

from goldilocks_core.advisors import magnetic_config as magnetic_config_module
from goldilocks_core.advisors.magnetic_config import (
    MagneticConfigHumanInput,
    OrderingRecord,
    enumerate_magnetic_orderings,
    magnetic_cell,
    magnetic_config,
)
from goldilocks_core.advisors.magnetic_enumeration import enumlib_installed
from goldilocks_core.analysis.is_magnetic import is_magnetic
from goldilocks_core.analysis.local_moments import local_moments
from goldilocks_core.analysis.symmetry import symmetry
from goldilocks_core.resolution import Blocked, Provenance, Resolved, Unavailable

_IRON = Structure(Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]])
_SILICON = Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
_ROCK_SALT_FEO = Structure(
    Lattice.cubic(4.3), ["Fe", "O"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
)
_BCC_IRON = Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
_DIAMOND_SILICON = Structure(
    Lattice.from_parameters(3.84, 3.84, 3.84, 60, 60, 60),
    ["Si", "Si"],
    [[0, 0, 0], [0.25, 0.25, 0.25]],
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


def test_starting_magnetization_is_keyed_by_the_cell_s_species_labels() -> None:
    """Regression for #32 (v2 epic 9, #9): a real structure's per-site
    labels are not always identical to its element symbols -- pymatgen's
    own CIF writer/reader already assigns distinct labels like
    'Fe0'/'Fe1' by default, not just AFM-relabeled structures. Before the
    fix, this heuristic default was keyed by element symbol
    (composition(structure).value.elements), which no longer matched
    generation/quantum_espresso/scf.py's label-keyed species_index once
    #27 made that label-keyed -- a bare KeyError on every bundled example
    structure. The seeds are keyed by the cell's species labels: two
    same-element, same-spin sites are one species, whatever labels the
    upload gave them (#170)."""
    two_iron_sites = Structure(
        Lattice.cubic(2.87),
        ["Fe", "Fe"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        labels=["Fe0", "Fe1"],
    )

    state = magnetic_decision(two_iron_sites, is_magnetic(two_iron_sites))

    assert state.ok
    assert [site.label for site in state.value.cell] == ["Fe", "Fe"]
    assert state.value.starting_magnetization == {"Fe": 0.1}


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
    assert state.value.starting_magnetization == {"Fe1": 0.5, "Fe2": -0.5}


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


def test_unavailable_magnetism_defaults_to_spin_polarized_with_a_warning() -> None:
    """Unavailable (the heuristic tried and could not tell) is not the same
    as Blocked (nothing upstream could even attempt an answer) -- there is
    still a safe default: spin-polarized, because the SCF can quench a
    seeded moment but nspin=1 can never find one (#142)."""
    state = magnetic_decision(
        _BCC_IRON, Unavailable(reason="no oxidation-state guess for Fe")
    )

    assert state.ok
    assert state.value.spin_polarized is True
    assert state.value.starting_magnetization == {"Fe": 0.1}
    assert state.source == "heuristic"
    (warning,) = state.value.warnings
    assert warning.code == "magnetic.magnetism_unknown_defaulted_spin_polarized"
    assert warning.level == "warning"
    assert "no oxidation-state guess for Fe" in warning.message
    assert "spin_polarized" in warning.message


_DISORDERED_FECO = Structure(
    Lattice.cubic(2.86),
    [{"Fe": 0.5, "Co": 0.5}, {"Fe": 0.5, "Co": 0.5}],
    [[0, 0, 0], [0.5, 0.5, 0.5]],
)


@pytest.mark.parametrize(
    "human",
    [MagneticConfigHumanInput(), MagneticConfigHumanInput(spin_polarized=True)],
    ids=["undecidable-default", "spin_polarized=true"],
)
def test_a_spin_polarized_disordered_structure_is_blocked(
    human: MagneticConfigHumanInput,
) -> None:
    """Seeds are set per site species, which a fractionally occupied site
    does not have (seeding one raised AttributeError on site.specie). The
    magnetism of a disordered structure cannot be decided, so its default
    is spin-polarized too (#142)."""
    state = magnetic_decision(
        _DISORDERED_FECO, is_magnetic(_DISORDERED_FECO), human=human
    )

    assert isinstance(state, Blocked)
    assert "ordered structure" in state.root_cause()
    assert "fractional site occupancies" in state.root_cause()


def test_an_unpolarized_disordered_structure_is_not_blocked_here() -> None:
    state = magnetic_decision(
        _DISORDERED_FECO,
        is_magnetic(_DISORDERED_FECO),
        human=MagneticConfigHumanInput(spin_polarized=False),
    )

    assert state.ok
    assert state.value.spin_polarized is False
    assert state.value.starting_magnetization is None


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


@pytest.mark.parametrize(
    ("spin_orbit_coupling", "spin_mode"), [(False, "none"), (True, "noncollinear")]
)
def test_explicit_spin_polarized_false_drops_a_tot_magnetization_and_warns(
    spin_orbit_coupling: bool, spin_mode: str
) -> None:
    """An explicit spin_polarized=False is a human override too and must
    win outright over a simultaneously-given tot_magnetization -- but
    silently dropping the latter with zero signal is still wrong, so
    this warns instead. The ignored value is dropped from the decision,
    with or without spin-orbit coupling, so it never reaches the writer
    (#146: pw.x refuses tot_magnetization without nspin=2)."""
    state = magnetic_decision(
        _DIAMOND_SILICON,
        is_magnetic(_DIAMOND_SILICON),
        human=MagneticConfigHumanInput(
            spin_polarized=False,
            tot_magnetization=2.0,
            spin_orbit_coupling=spin_orbit_coupling,
        ),
    )

    assert state.ok
    assert state.value.spin_polarized is False
    assert state.value.tot_magnetization is None
    assert state.value.spin_mode == spin_mode
    assert any(
        warning.code == "magnetic.magnetization_ignored_spin_polarized_false"
        for warning in state.value.warnings
    )


def test_spin_orbit_coupling_refuses_a_tot_magnetization() -> None:
    """#146: spin-orbit coupling makes the run noncollinear (pw.x sets
    nspin=4, QE 7.3 PW/src/input.f90:718), and pw.x accepts
    tot_magnetization only with nspin=2 (input.f90:761-762). The request
    is refused here, where /explain shows it, instead of passing the gate
    and failing in the writer."""
    state = magnetic_decision(
        _BCC_IRON,
        is_magnetic(_BCC_IRON),
        human=MagneticConfigHumanInput(spin_orbit_coupling=True, tot_magnetization=2.0),
    )

    assert isinstance(state, Blocked)
    assert "tot_magnetization" in state.root_cause()
    assert "spin_orbit_coupling" in state.root_cause()
    assert "starting_magnetization" in state.root_cause()


@pytest.mark.parametrize(
    ("structure", "spin_orbit_coupling", "spin_mode"),
    [
        (_DIAMOND_SILICON, False, "none"),
        (_BCC_IRON, False, "collinear"),
        (_BCC_IRON, True, "noncollinear"),
        (_DIAMOND_SILICON, True, "noncollinear"),
    ],
)
def test_spin_mode_is_noncollinear_with_spin_orbit_otherwise_follows_spin(
    structure: Structure, spin_orbit_coupling: bool, spin_mode: str
) -> None:
    """The one spin treatment the gate, the occupations advisor, the
    writer and the job's layout read (#146): spin-orbit coupling is
    noncollinear whether or not the run is magnetic."""
    state = magnetic_decision(
        structure,
        is_magnetic(structure),
        human=MagneticConfigHumanInput(spin_orbit_coupling=spin_orbit_coupling),
    )

    assert state.value.spin_mode == spin_mode


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


def _same_sites(cell: Structure, structure: Structure) -> bool:
    """Whether ``cell`` has ``structure``'s lattice, elements and
    positions, in order -- the FM identity pass-through, whatever species
    and labels the split gave its sites."""
    return (
        np.allclose(cell.lattice.matrix, structure.lattice.matrix)
        and np.allclose(cell.frac_coords, structure.frac_coords)
        and [site.specie.symbol for site in cell]
        == [site.specie.symbol for site in structure]
    )


def test_the_cell_is_the_input_structure_by_default() -> None:
    """AFM species-splitting (v2 epic 9, #9) is opt-in via
    ``magnetic_ordering="afm"`` -- without it, the cell keeps the input's
    sites (the FM identity pass-through), every one seeded positive."""
    state = magnetic_cell(_IRON, is_magnetic(_IRON))

    assert _same_sites(state.value.structure, _IRON)
    assert _spins(state.value.structure, "Fe") == {1.0}
    assert state.value.ordering == OrderingRecord(requested="fm", applied="fm")
    assert np.allclose(state.value.cell_in_input_basis, np.eye(3))


def _afm_rock_salt(structure: Structure, axis: int = 2) -> Structure:
    """A spin-decorated 4-site cell of a 2-site rock-salt input, its two
    Ni up and down along the doubled lattice vector ``axis`` (c by
    default), as the enumerator returns one."""
    scaling = [1, 1, 1]
    scaling[axis] = 2
    supercell = structure * scaling
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
    assert cell.ordering == OrderingRecord(requested="afm", applied="afm-1")
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
    electrons, with opposite signs on the two Ni sublattices. O, off the
    sublattice, starts at 0 as in QE's own AFM references (#170); aiida's
    +0.1 on every species is the ferromagnetic default only."""
    cell = magnetic_cell(_ROCK_SALT_NIO, is_magnetic(_ROCK_SALT_NIO), _AFM)

    state = magnetic_config(cell, z_valences={"Ni": 18.0, "O": 6.0}, human=_AFM)
    ferromagnetic = magnetic_decision(
        _ROCK_SALT_NIO, is_magnetic(_ROCK_SALT_NIO), z_valences={"Ni": 18.0}
    )

    assert state.value.cell is cell.value.structure
    assert state.value.ordering == cell.value.ordering
    assert state.value.starting_magnetization == {
        "Ni1": 5.0 / 18.0,
        "Ni2": -5.0 / 18.0,
        "O": 0.0,
    }
    assert ferromagnetic.value.starting_magnetization == {"Ni": 5.0 / 18.0, "O": 0.1}
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


def _afm_8_site_rock_salt(structure: Structure) -> Structure:
    """A spin-decorated 8-site candidate of a 2-site rock-salt input."""
    supercell = structure * (2, 2, 1)
    spins = iter((5.0, -5.0, -5.0, 5.0))
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


def test_afm_applies_the_first_listed_candidate_and_records_its_label(
    monkeypatch,
) -> None:
    """#170: plain afm and the label the listing gives the same candidate
    write the same cell, so a listed candidate fed back to a run cannot
    diverge from the afm path. afm-1 is the smallest candidate, wherever
    the enumerator put it."""
    small = _afm_rock_salt(_ROCK_SALT_NIO)
    large = _afm_8_site_rock_salt(_ROCK_SALT_NIO)
    monkeypatch.setattr(shutil, "which", _on_path("enum.x", "makeStr.py"))
    monkeypatch.setattr(
        magnetic_config_module,
        "enumerate_orderings",
        _enumerating((large, "afm"), (small, "afm")),
    )

    def cell(ordering: str):
        return magnetic_cell(
            _ROCK_SALT_NIO,
            is_magnetic(_ROCK_SALT_NIO),
            MagneticConfigHumanInput(magnetic_ordering=ordering),
        ).value

    afm, first, second = cell("afm"), cell("afm-1"), cell("afm-2")

    assert afm.ordering == OrderingRecord(requested="afm", applied="afm-1")
    assert first.ordering == OrderingRecord(requested="afm-1", applied="afm-1")
    assert second.ordering == OrderingRecord(requested="afm-2", applied="afm-2")
    assert afm.structure == first.structure
    assert [site.label for site in afm.structure] == [
        site.label for site in first.structure
    ]
    assert len(first.structure) == 4
    assert len(second.structure) == 8


def test_afm_labels_follow_a_canonical_order_not_the_enumerator_s(
    monkeypatch,
) -> None:
    """#170: pymatgen orders candidates of one size and space group by
    the order enumlib's output files are read in, which can differ between
    the host that lists and the one that runs. afm-<n> names the same cell
    whatever order the enumerator returns them in."""
    along_a = _afm_rock_salt(_ROCK_SALT_NIO, axis=0)
    along_c = _afm_rock_salt(_ROCK_SALT_NIO, axis=2)
    monkeypatch.setattr(shutil, "which", _on_path("enum.x", "makeStr.py"))

    listings = []
    for order in ((along_a, along_c), (along_c, along_a)):
        monkeypatch.setattr(
            magnetic_config_module,
            "enumerate_orderings",
            _enumerating(*((candidate, "afm") for candidate in order)),
        )
        candidates, _warning = enumerate_magnetic_orderings(
            _ROCK_SALT_NIO, local_moments(_ROCK_SALT_NIO)
        )
        listings.append(
            {label: cell.lattice.matrix.round(6).tolist() for label, cell in candidates}
        )

    assert list(listings[0]) == ["fm", "afm-1", "afm-2"]
    assert listings[0] == listings[1]


def test_an_unknown_afm_label_is_blocked_and_names_the_available_ones(
    afm_candidate,
) -> None:
    """#170: a label names one listed candidate; one this structure does
    not have is refused rather than written as another cell."""
    state = magnetic_cell(
        _ROCK_SALT_NIO,
        is_magnetic(_ROCK_SALT_NIO),
        MagneticConfigHumanInput(magnetic_ordering="afm-2"),
    )

    assert isinstance(state, Blocked)
    assert "'afm-2'" in state.root_cause()
    assert "fm, afm-1" in state.root_cause()


def test_a_listed_label_is_blocked_with_the_reason_none_can_be_listed(
    monkeypatch,
) -> None:
    """#170: plain afm degrades to ferromagnetic order with a warning, but
    a request for one listed candidate cannot be met by writing another."""
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    state = magnetic_cell(
        _ROCK_SALT_NIO,
        is_magnetic(_ROCK_SALT_NIO),
        MagneticConfigHumanInput(magnetic_ordering="afm-1"),
    )

    assert isinstance(state, Blocked)
    assert "'afm-1'" in state.root_cause()
    assert "enumlib" in state.root_cause()


@pytest.mark.parametrize("value", ["afm-0", "afm-01", "AFM", "afm1", "ferri"])
def test_magnetic_ordering_takes_fm_afm_or_a_listed_label(value: str) -> None:
    with pytest.raises(ValidationError):
        MagneticConfigHumanInput(magnetic_ordering=value)


def test_a_human_split_is_a_species_split_of_the_cell_symmetry_sees() -> None:
    """#170: two Fe given opposite starting magnetizations are two pw.x
    species, so the cell symmetry_eff and n_irr_k read must hold them as
    two types (simple cubic B2, Pm-3m), not bcc Im-3m with the split
    applied only by the writer."""
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

    cell = state.value.cell
    assert [site.label for site in cell] == ["Fe1", "Fe2"]
    assert state.value.starting_magnetization == {"Fe1": 0.5, "Fe2": -0.5}
    assert symmetry(cell).value.space_group_symbol == "Pm-3m"
    assert symmetry(two_iron_sites).value.space_group_symbol == "Im-3m"


def test_the_cell_has_one_species_per_element_whatever_the_upload_s_labels() -> None:
    """Many CIFs label every site (Si1..Si40) with no physical meaning: the
    cell carries one species per element, so neither symmetry nor the
    writer sees them as distinct (moved from the writer, #170)."""
    labelled = Structure(
        Lattice.cubic(5.43),
        ["Si", "Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.0]],
        labels=["Si1", "Si7", "Si23"],
    )

    state = magnetic_cell(
        labelled,
        is_magnetic(labelled),
        MagneticConfigHumanInput(spin_polarized=False),
    )

    assert [site.label for site in state.value.structure] == ["Si", "Si", "Si"]


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


_CLASSIFIED_NON_MAGNETIC = Resolved("non_magnetic", Provenance(source="ml"))
"""What the ML tier answered for bcc Cr and La2CuO4 (#145)."""

_BCC_CHROMIUM = Structure(
    Lattice.cubic(2.88), ["Cr", "Cr"], [[0, 0, 0], [0.5, 0.5, 0.5]]
)


def test_an_afm_request_turns_spin_polarization_on_whatever_the_classifier_says(
    afm_candidate,
) -> None:
    """#145: magnetic_ordering=afm is a spin request, like tot_magnetization
    and starting_magnetization (#34). On a structure classified
    non-magnetic it used to be accepted and dropped: no nspin, no ordering,
    no warning."""
    state = magnetic_cell(_ROCK_SALT_NIO, _CLASSIFIED_NON_MAGNETIC, _AFM)

    assert state.value.spin_polarized is True
    assert state.source == "human"
    assert state.value.magnetic_elements == ("Ni",)
    assert state.value.ordering == OrderingRecord(requested="afm", applied="afm-1")
    assert len(state.value.structure) == 4


def test_an_afm_request_that_cannot_be_ordered_still_leaves_a_record(
    monkeypatch,
) -> None:
    """#145: with enumlib missing, an afm request on a structure classified
    non-magnetic runs spin-polarized and says why it stayed
    ferromagnetic."""
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    state = magnetic_cell(_BCC_CHROMIUM, _CLASSIFIED_NON_MAGNETIC, _AFM)

    assert state.value.spin_polarized is True
    assert state.source == "human"
    assert state.value.ordering == OrderingRecord(requested="afm", applied="fm")
    (warning,) = state.value.warnings
    assert warning.code == "magnetic.afm_ordering_unavailable"
    assert "enumlib" in warning.message


@pytest.mark.parametrize(
    ("magnetization", "codes"),
    [
        ({}, ["magnetic.ordering_ignored_spin_polarized_false"]),
        (
            {"tot_magnetization": 2.0},
            [
                "magnetic.magnetization_ignored_spin_polarized_false",
                "magnetic.ordering_ignored_spin_polarized_false",
            ],
        ),
    ],
    ids=["ordering", "ordering+tot_magnetization"],
)
def test_explicit_spin_polarized_false_keeps_an_afm_request_off_and_warns(
    magnetization: dict[str, float], codes: list[str]
) -> None:
    """#145: an explicit spin_polarized=false wins over magnetic_ordering=afm,
    as it does over the magnetization keys, and says that the ordering was
    ignored -- next to the magnetization warning when those were given
    too."""
    human = MagneticConfigHumanInput(
        spin_polarized=False, magnetic_ordering="afm", **magnetization
    )

    state = magnetic_cell(_BCC_CHROMIUM, is_magnetic(_BCC_CHROMIUM), human)

    assert state.value.spin_polarized is False
    assert state.source == "human"
    assert state.value.structure == _BCC_CHROMIUM
    assert state.value.ordering == OrderingRecord(requested="afm", applied="fm")
    assert sorted(warning.code for warning in state.value.warnings) == codes
    (ignored,) = (
        warning
        for warning in state.value.warnings
        if warning.code == "magnetic.ordering_ignored_spin_polarized_false"
    )
    assert ignored.level == "warning"
    assert "magnetic_ordering" in ignored.message
    assert "spin_polarized" in ignored.message


def test_an_fm_ordering_is_a_spin_request_on_the_input_cell() -> None:
    """#170: ``"fm"`` is the listing's ferromagnetic candidate, generated
    from the listed structure by its label alone. Like an afm label it asks
    for a spin-polarized run, on the input cell; read as the unset default,
    it wrote an unpolarized deck for a structure classified non-magnetic."""
    human = MagneticConfigHumanInput(magnetic_ordering="fm")

    state = magnetic_cell(_BCC_CHROMIUM, _CLASSIFIED_NON_MAGNETIC, human)

    assert state.value.spin_polarized is True
    assert state.source == "human"
    assert _same_sites(state.value.structure, _BCC_CHROMIUM)
    assert (state.value.ordering.requested, state.value.ordering.applied) == (
        "fm",
        "fm",
    )
    assert state.value.warnings == ()


def test_explicit_spin_polarized_false_ignores_an_fm_ordering_and_warns() -> None:
    human = MagneticConfigHumanInput(spin_polarized=False, magnetic_ordering="fm")

    state = magnetic_cell(_BCC_CHROMIUM, _CLASSIFIED_NON_MAGNETIC, human)

    assert state.value.spin_polarized is False
    assert [warning.code for warning in state.value.warnings] == [
        "magnetic.ordering_ignored_spin_polarized_false"
    ]


def test_an_unset_ordering_leaves_spin_polarization_to_the_classifier() -> None:
    state = magnetic_cell(_BCC_CHROMIUM, _CLASSIFIED_NON_MAGNETIC)

    assert state.value.spin_polarized is False
    assert state.source == "heuristic"
    assert state.value.warnings == ()


@pytest.mark.skipif(
    _ENUMLIB_MISSING,
    reason="needs the enumlib executables (enum.x, makeStr.py) on PATH",
)
def test_afm_on_a_54_site_supercell_of_a_one_fe_cell_is_applied() -> None:
    """#171: the ceiling counted all 54 sites of this 3x3x3 supercell and
    kept it ferromagnetic, though pymatgen reduces it to its primitive cell
    first, whose one Fe enumlib orders in about a second."""
    supercell = _ROCK_SALT_FEO * (3, 3, 3)

    state = magnetic_cell(supercell, is_magnetic(supercell), _AFM)

    assert state.value.ordering == OrderingRecord(requested="afm", applied="afm-1")
    assert any(
        warning.code == "magnetic.afm_ordering_applied"
        for warning in state.value.warnings
    )


def _feo_supercell(scaling: tuple[int, int, int], *, displaced: bool) -> Structure:
    """``_ROCK_SALT_FEO`` repeated by ``scaling``; ``displaced`` moves one O
    off its site, so the supercell is its own primitive cell."""
    supercell = _ROCK_SALT_FEO * scaling
    if displaced:
        oxygen = next(
            index for index, site in enumerate(supercell) if site.specie.symbol == "O"
        )
        supercell.translate_sites([oxygen], [0.02, 0.0, 0.0])
    return supercell


@pytest.mark.parametrize(
    ("scaling", "displaced"),
    [((3, 3, 3), False), ((2, 2, 2), True)],
    ids=["27-Fe-supercell-of-a-1-Fe-cell", "8-Fe-primitive-cell"],
)
def test_afm_enumerates_up_to_eight_magnetic_sites_in_the_primitive_cell(
    monkeypatch, scaling: tuple[int, int, int], displaced: bool
) -> None:
    """#171: what enumlib orders is the sublattice sites of the primitive
    cell, not every atom of the uploaded one."""
    enumerated: list[Structure] = []

    def enumerate_orderings(structure: Structure, **_kwargs):
        enumerated.append(structure)
        return ()

    monkeypatch.setattr(shutil, "which", _on_path("enum.x", "makeStr.py"))
    monkeypatch.setattr(
        magnetic_config_module, "enumerate_orderings", enumerate_orderings
    )
    supercell = _feo_supercell(scaling, displaced=displaced)

    _candidates, warning = enumerate_magnetic_orderings(
        supercell, local_moments(supercell)
    )

    assert len(enumerated) == 1
    assert warning.message == (
        "no compensated antiferromagnetic ordering was found; kept ferromagnetic order."
    )


def test_afm_refuses_more_than_eight_magnetic_sites_in_the_primitive_cell(
    monkeypatch,
) -> None:
    """#171: nine Fe that no translation maps onto each other are past the
    ceiling, which is refused before enumlib runs."""
    monkeypatch.setattr(shutil, "which", _on_path("enum.x", "makeStr.py"))
    monkeypatch.setattr(
        magnetic_config_module,
        "enumerate_orderings",
        _failing_with(AssertionError("enumerated past the ceiling")),
    )
    supercell = _feo_supercell((3, 3, 1), displaced=True)

    state = magnetic_cell(supercell, is_magnetic(supercell), _AFM)

    assert _same_sites(state.value.structure, supercell)
    assert state.value.ordering == OrderingRecord(requested="afm", applied="fm")
    [warning] = _afm_warnings(state.value.warnings)
    assert warning.code == "magnetic.afm_ordering_unavailable"
    assert warning.message == (
        "the primitive cell has 9 magnetic sites (Fe), over the 8 this "
        "heuristic enumerates orderings for; kept ferromagnetic order."
    )


def test_the_enumerator_gets_an_interactive_budget(monkeypatch) -> None:
    """#171: pymatgen gives enumlib 5 minutes per call, starts a joblib
    process pool on every call for its bookkeeping, and adds ferrimagnetic
    strategies whose candidates are never antiferromagnetic."""
    received: dict[str, object] = {}

    def enumerate_orderings(_structure: Structure, **kwargs):
        received.update(kwargs)
        return ()

    monkeypatch.setattr(shutil, "which", _on_path("enum.x", "makeStr.py"))
    monkeypatch.setattr(
        magnetic_config_module, "enumerate_orderings", enumerate_orderings
    )

    enumerate_magnetic_orderings(_ROCK_SALT_FEO, local_moments(_ROCK_SALT_FEO))

    assert received["automatic"] is False
    assert received["transformation_kwargs"] == {"timeout": 0.5, "n_jobs": 1}


def _canted_nio() -> Structure:
    """Conventional rock-salt NiO with the non-collinear Ni moments a magCIF
    carries as a ``magmom`` site property."""
    nio = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(4.17), ["Ni", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    directions = iter(([1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]))
    nio.add_site_property(
        "magmom",
        [
            Magmom(next(directions) if site.specie.symbol == "Ni" else [0, 0, 0])
            for site in nio
        ],
    )
    return nio


@pytest.mark.skipif(
    _ENUMLIB_MISSING,
    reason="needs the enumlib executables (enum.x, makeStr.py) on PATH",
)
def test_afm_seeds_the_sublattice_whatever_moments_the_upload_carries() -> None:
    """pymatgen refuses an input whose own moments are non-collinear; the
    AFM search never reads them, it seeds the sublattice itself."""
    nio = _canted_nio()

    state = magnetic_cell(nio, is_magnetic(nio), _AFM)

    assert state.value.ordering == OrderingRecord(requested="afm", applied="afm-1")
    assert _spins(state.value.structure, "Ni") == {1.0, -1.0}


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


def _la2cuo4() -> Structure:
    """T-type La2CuO4 (I4/mmm, Materials Project mp-19735): Cu2+ d9 and
    La3+ 4f0."""
    return Structure.from_spacegroup(
        "I4/mmm",
        Lattice.tetragonal(3.78, 13.2),
        ["La", "Cu", "O", "O"],
        [[0, 0, 0.3606], [0, 0, 0], [0, 0.5, 0], [0, 0, 0.1828]],
    )


def _fept() -> Structure:
    """L1_0 FePt (P4/mmm, MC3D mc3d-49510): Pt's moment is induced by Fe."""
    return Structure.from_spacegroup(
        "P4/mmm",
        Lattice.tetragonal(2.72, 3.71),
        ["Fe", "Pt"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )


_SOLID_OXYGEN = Structure(
    Lattice.from_parameters(3.5119, 3.5119, 4.2801, 67.257, 112.743, 115.955),
    ["O", "O"],
    [[0.9105, 0.0895, 0.8544], [0.0895, 0.9105, 0.1456]],
)
"""alpha-O2 (MC3D mc3d-25060): its moment is the O2 molecule's triplet, no
site carries a local d or f moment."""

_URANIUM_SULFIDE = Structure.from_spacegroup(
    "Fm-3m", Lattice.cubic(5.489), ["U", "S"], [[0, 0, 0], [0.5, 0.5, 0.5]]
)
"""Rock-salt US: no oxidation-state guess, and U's one common state (+6) is
5f0, so its local moments are undecidable."""


def _spins(cell: Structure, element: str) -> set[float]:
    return {
        getattr(site.specie, "spin", None) or 0.0
        for site in cell
        if site.specie.symbol == element
    }


def _afm_warnings(warnings) -> list:
    return [warning for warning in warnings if warning.code.startswith("magnetic.afm_")]


@pytest.mark.skipif(
    _ENUMLIB_MISSING,
    reason="needs the enumlib executables (enum.x, makeStr.py) on PATH",
)
def test_afm_orders_the_open_shell_cu_of_la2cuo4_and_never_la() -> None:
    """#143: the sublattice is the sites with a local moment. Seeding 4f0
    La3+ too let La carry the compensating spin while both Cu stayed
    parallel (afm-7 converged to Cu-FM on pw.x)."""
    la2cuo4 = _la2cuo4()

    state = magnetic_cell(la2cuo4, is_magnetic(la2cuo4), _AFM)

    cell = state.value
    assert cell.magnetic_elements == ("Cu",)
    assert cell.ordering == OrderingRecord(requested="afm", applied="afm-1")
    assert _spins(cell.structure, "Cu") == {1.0, -1.0}
    assert _spins(cell.structure, "La") == {0.0}
    assert {site.label for site in cell.structure} == {"Cu1", "Cu2", "La", "O"}


@pytest.mark.skipif(
    _ENUMLIB_MISSING,
    reason="needs the enumlib executables (enum.x, makeStr.py) on PATH",
)
def test_afm_on_fept_splits_fe_in_a_doubled_cell_not_fe_against_pt() -> None:
    """#143: with Pt seeded too, the 2-site input cell itself (Fe up, Pt
    down) was applied as 'antiferromagnetic' and relaxed to the
    ferromagnetic state. Fe alone needs at least a doubled cell."""
    fept = _fept()

    state = magnetic_cell(fept, is_magnetic(fept), _AFM)

    cell = state.value
    assert cell.magnetic_elements == ("Fe",)
    assert cell.ordering.applied == "afm-1"
    assert len(cell.structure) == 4
    assert _spins(cell.structure, "Fe") == {1.0, -1.0}
    assert _spins(cell.structure, "Pt") == {0.0}
    assert {site.label for site in cell.structure} == {"Fe1", "Fe2", "Pt"}


@pytest.mark.parametrize(
    ("structure", "human"),
    [
        (_SOLID_OXYGEN, _AFM),
        (
            _DIAMOND_SILICON,
            MagneticConfigHumanInput(magnetic_ordering="afm", spin_polarized=True),
        ),
    ],
    ids=["solid-O2-radical", "closed-shell-Si"],
)
def test_afm_without_a_moment_bearing_site_is_refused_before_enumlib(
    monkeypatch, structure: Structure, human: MagneticConfigHumanInput
) -> None:
    """#143: an empty sublattice reached pymatgen, whose 'Not detected as
    magnetic' text came back as the reason. It is caught first, under its
    own code naming the supported action."""
    monkeypatch.setattr(shutil, "which", _on_path("enum.x", "makeStr.py"))
    monkeypatch.setattr(
        magnetic_config_module,
        "enumerate_orderings",
        _failing_with(AssertionError("enumerated an empty sublattice")),
    )

    state = magnetic_cell(structure, is_magnetic(structure), human)

    assert state.ok
    assert _same_sites(state.value.structure, structure)
    assert state.value.magnetic_elements == ()
    assert state.value.ordering == OrderingRecord(requested="afm", applied="fm")
    [warning] = _afm_warnings(state.value.warnings)
    assert warning.code == "magnetic.afm_no_open_shell_sites"
    assert warning.level == "warning"
    assert "local d or f moment" in warning.message
    assert "starting_magnetization per site label" in warning.message
    assert "pymatgen" not in warning.message


def test_afm_with_undecidable_local_moments_reports_why(monkeypatch) -> None:
    """#143: 'no open-shell sites' would be false when the model cannot
    tell; the warning carries local_moments' own reason instead."""
    monkeypatch.setattr(shutil, "which", _on_path("enum.x", "makeStr.py"))
    monkeypatch.setattr(
        magnetic_config_module,
        "enumerate_orderings",
        _failing_with(AssertionError("enumerated an undecided sublattice")),
    )

    state = magnetic_cell(_URANIUM_SULFIDE, is_magnetic(_URANIUM_SULFIDE), _AFM)

    assert state.ok
    assert state.value.spin_polarized is True
    assert state.value.magnetic_elements == ()
    assert state.value.ordering == OrderingRecord(requested="afm", applied="fm")
    [warning] = _afm_warnings(state.value.warnings)
    assert warning.code == "magnetic.afm_ordering_unavailable"
    assert "could not be determined" in warning.message
    assert "U (+6)" in warning.message
    assert "starting_magnetization per site label" in warning.message


def test_enumerate_magnetic_orderings_is_just_fm_when_enumlib_is_missing(
    monkeypatch,
) -> None:
    """#87's listing capability degrades the same way ``magnetic_cell``
    does: no enumlib means no antiferromagnetic candidates, not an error."""
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    candidates, warning = enumerate_magnetic_orderings(
        _ROCK_SALT_FEO, local_moments(_ROCK_SALT_FEO)
    )

    assert candidates == (("fm", _ROCK_SALT_FEO),)
    assert warning.code == "magnetic.afm_ordering_unavailable"
    assert "enumlib" in warning.message


def test_enumerate_magnetic_orderings_never_picks_a_winner(monkeypatch) -> None:
    """Unlike ``magnetic_cell``, listing every candidate is the
    whole point -- #87's mMACE ranking (or a human) picks the winner, not
    this function."""
    monkeypatch.setattr(shutil, "which", _on_path("enum.x", "makeStr.py"))
    monkeypatch.setattr(
        magnetic_config_module,
        "enumerate_orderings",
        _enumerating((_ROCK_SALT_FEO.copy(), "afm"), (_ROCK_SALT_FEO.copy(), "afm")),
    )

    candidates, warning = enumerate_magnetic_orderings(
        _ROCK_SALT_FEO, local_moments(_ROCK_SALT_FEO)
    )

    labels = [label for label, _structure in candidates]
    assert labels == ["fm", "afm-1", "afm-2"]
    assert warning is None


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

    candidates, warning = enumerate_magnetic_orderings(
        _ROCK_SALT_FEO, local_moments(_ROCK_SALT_FEO)
    )

    assert candidates == (("fm", _ROCK_SALT_FEO),)
    assert warning.code == "magnetic.afm_ordering_unavailable"
    assert "enumlib" in warning.message
    assert "makeStr.py" in warning.message


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

    candidates, warning = enumerate_magnetic_orderings(
        _ROCK_SALT_FEO, local_moments(_ROCK_SALT_FEO)
    )

    assert [label for label, _structure in candidates] == ["fm", "afm-1"]
    assert warning is None


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError(2, "No such file or directory", "vasp.3"),
        LinAlgError("Singular matrix"),
        RuntimeError("This shouldn't happen, found neighbors=[]"),
        ValueError("Too many magnetic sites to sensibly perform enumeration."),
        ValueError("Input structure (NiO) is non-collinear."),
    ],
)
def test_an_enumeration_error_that_is_not_an_enumlib_outcome_propagates(
    monkeypatch, error: Exception
) -> None:
    """#136: overlapping enumerations sharing one cwd failed with the first
    three errors (LinAlgError subclasses ValueError), which came back as a
    plausible ferromagnetic-only listing quoting a host path. #171: the
    magnetic-site ceiling and the moment-free input leave pymatgen no other
    ValueError to raise about a structure, so one is a fault too."""
    monkeypatch.setattr(shutil, "which", _on_path("enum.x", "makeStr.py"))
    monkeypatch.setattr(
        magnetic_config_module, "enumerate_orderings", _failing_with(error)
    )

    with pytest.raises(type(error)):
        enumerate_magnetic_orderings(_ROCK_SALT_FEO, local_moments(_ROCK_SALT_FEO))


def test_enumlib_finding_no_ordering_gets_a_fixed_reason(monkeypatch) -> None:
    """pymatgen turns enumlib's EnumError for every cell size it tries into
    ValueError('Unable to enumerate'), the one answer about the structure a
    ValueError still gives (#171)."""
    monkeypatch.setattr(shutil, "which", _on_path("enum.x", "makeStr.py"))
    monkeypatch.setattr(
        magnetic_config_module,
        "enumerate_orderings",
        _failing_with(ValueError("Unable to enumerate")),
    )

    candidates, warning = enumerate_magnetic_orderings(
        _ROCK_SALT_FEO, local_moments(_ROCK_SALT_FEO)
    )

    assert candidates == (("fm", _ROCK_SALT_FEO),)
    assert warning.code == "magnetic.afm_ordering_unavailable"
    assert warning.message == (
        "enumlib found no ordering of the magnetic sites for this structure; "
        "kept ferromagnetic order."
    )


def test_an_enumlib_timeout_gets_a_fixed_reason_without_the_error_text(
    monkeypatch,
) -> None:
    monkeypatch.setattr(shutil, "which", _on_path("enum.x", "makeStr.py"))
    monkeypatch.setattr(
        magnetic_config_module,
        "enumerate_orderings",
        _failing_with(TimeoutError("Enumeration took more than timeout 0.5 minutes")),
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
        "enumlib did not finish within its 30 s time limit; kept ferromagnetic order."
    )


@pytest.mark.skipif(
    _ENUMLIB_MISSING,
    reason="needs the enumlib executables (enum.x, makeStr.py) on PATH",
)
def test_enumerate_magnetic_orderings_finds_the_same_split_relabeling_does() -> None:
    candidates, warning = enumerate_magnetic_orderings(
        _ROCK_SALT_FEO, local_moments(_ROCK_SALT_FEO)
    )

    assert warning is None
    assert candidates[0][0] == "fm"
    assert candidates[0][1] == _ROCK_SALT_FEO
    afm_candidates = candidates[1:]
    assert len(afm_candidates) > 0
    for label, candidate in afm_candidates:
        assert label.startswith("afm-")
        assert len(set(candidate.species)) > len(set(_ROCK_SALT_FEO.species))
