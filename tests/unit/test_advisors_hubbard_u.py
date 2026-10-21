from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Element

from goldilocks_core.advisors.functional import HYBRID_FUNCTIONAL
from goldilocks_core.advisors.hubbard_u import (
    HubbardU,
    HubbardUHumanInput,
    HubbardULlmInput,
    hubbard_u,
)
from goldilocks_core.analysis.composition import composition
from goldilocks_core.analysis.needs_correlation import needs_correlation
from goldilocks_core.resolution import Blocked

_MP = ("Sci. Rep. 11, 15496", "10.1038/s41598-021-94550-5")
_MP_RELAX_SET = ("MPRelaxSet", "10.1103/PhysRevB.84.045115")
_JAIN = ("Jain et al., Phys. Rev. B 84, 045115 (2011), Table I", _MP_RELAX_SET[1])
_TOPSAKAL = ("Topsakal", "10.1016/j.commatsci.2014.07.030")
_OQMD = ("Kirklin et al. (OQMD)", "10.1038/npjcompumats.2015.10")
_SUZUKI = ("Suzuki", "10.1103/PhysRevB.88.195146")
_CITED_U_TABLE = {
    "V": ("3d", 3.25, *_MP),
    "Cr": ("3d", 3.7, *_MP),
    "Mn": ("3d", 3.9, *_MP),
    "Fe": ("3d", 5.3, *_MP),
    "Co": ("3d", 3.32, *_MP),
    "Ni": ("3d", 6.2, *_MP),
    "Cu": ("3d", 4.0, *_JAIN),
    "Mo": ("4d", 4.38, *_MP_RELAX_SET),
    "W": ("5d", 6.2, *_MP_RELAX_SET),
    "Ce": ("4f", 5.0, "Loschen", "10.1103/PhysRevB.75.035115"),
    "Pr": ("4f", 4.0, *_TOPSAKAL),
    "Nd": ("4f", 3.1, *_TOPSAKAL),
    "Pm": ("4f", 3.4, *_TOPSAKAL),
    "Sm": ("4f", 3.3, *_TOPSAKAL),
    "Eu": ("4f", 3.0, *_TOPSAKAL),
    "Gd": ("4f", 4.6, *_TOPSAKAL),
    "Tb": ("4f", 5.0, *_TOPSAKAL),
    "Dy": ("4f", 5.0, *_TOPSAKAL),
    "Ho": ("4f", 4.9, *_TOPSAKAL),
    "Er": ("4f", 4.2, *_TOPSAKAL),
    "Tm": ("4f", 4.8, *_TOPSAKAL),
    "Yb": ("4f", 3.0, *_TOPSAKAL),
    "Pa": ("5f", 4.0, "qmpy configuration", _OQMD[1]),
    "U": ("5f", 4.0, *_OQMD),
    "Np": ("5f", 4.0, *_OQMD),
    "Pu": ("5f", 4.0, *_OQMD),
    "Am": ("5f", 4.0, *_SUZUKI),
    "Cm": ("5f", 4.0, *_SUZUKI),
    "Bk": ("5f", 3.54, "Kaltsoyannis", "10.1021/acs.jpcc.5c01663"),
}
"""The maintainer-reviewed table (#135): element -> (manifold, Dudarev
U_eff in eV, a citation fragment naming its source, DOI)."""
_TABULATED = frozenset(_CITED_U_TABLE)
_NO_U = frozenset(
    {
        *("Sc", "Ti", "Zn", "Y", "Zr", "Nb", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"),
        *("Hf", "Ta", "Re", "Os", "Ir", "Pt", "Au", "Hg", "La", "Lu", "Th"),
    }
)
_MISSING = frozenset({"Ac", "Cf", "Es", "Fm", "Md", "No", "Lr"})
_FLAGGED_WITH_PSEUDOPOTENTIALS = tuple(
    element.symbol
    for element in (Element.from_Z(z) for z in range(1, 104))
    if element.is_transition_metal or element.is_lanthanoid or element.is_actinoid
)


def _rock_salt(cation: str, anion: str = "O") -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(4.4), [cation, anion], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def _perovskite(a_site: str, b_site: str) -> Structure:
    return Structure.from_spacegroup(
        "Pm-3m",
        Lattice.cubic(3.9),
        [a_site, b_site, "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]],
    )


def _hubbard(structure: Structure, functional: str = "PBEsol", **overrides):
    return hubbard_u(
        structure, needs_correlation(composition(structure)), functional, **overrides
    )


def _warning(state, code: str):
    (warning,) = (item for item in state.value.warnings if item.code == code)
    return warning


def _codes(state) -> tuple[str, ...]:
    return tuple(warning.code for warning in state.value.warnings)


def test_the_element_lists_cover_every_correlated_element_once() -> None:
    """Every element the correlation rule can flag and a pseudopotential
    table covers (Z <= 103) gets a cited U, no U by convention, or the
    missing-value warning -- exactly one of the three."""
    assert sorted(_TABULATED | _NO_U | _MISSING) == sorted(
        _FLAGGED_WITH_PSEUDOPOTENTIALS
    )
    assert len(_TABULATED) + len(_NO_U) + len(_MISSING) == len(
        _FLAGGED_WITH_PSEUDOPOTENTIALS
    )


@pytest.mark.parametrize("symbol", _FLAGGED_WITH_PSEUDOPOTENTIALS)
def test_each_correlated_oxide_gets_its_policy(symbol: str) -> None:
    state = _hubbard(_rock_salt(symbol))

    assert state.ok
    if symbol in _TABULATED:
        (entry,) = state.value.u_values
        assert (entry.element, entry.value_source) == (symbol, "table")
        assert entry.u_ev > 0
        assert entry.citation
        assert entry.doi
        assert "hubbard.u_value_missing" not in _codes(state)
    elif symbol in _NO_U:
        assert state.value.u_values == ()
        assert state.value.warnings == ()
    else:
        assert state.value.u_values == ()
        assert _codes(state) == ("hubbard.u_value_missing",)


@pytest.mark.parametrize(
    ("symbol", "manifold", "u_ev", "citation", "doi"),
    [(symbol, *row) for symbol, row in _CITED_U_TABLE.items()],
)
def test_tabulated_values_carry_their_citation(
    symbol: str, manifold: str, u_ev: float, citation: str, doi: str
) -> None:
    state = _hubbard(_rock_salt(symbol))

    (entry,) = state.value.u_values
    assert (entry.element, entry.manifold, entry.u_ev) == (symbol, manifold, u_ev)
    assert entry.value_source == "table"
    assert citation in entry.citation
    assert entry.doi == doi
    assert state.source == "heuristic"


def test_metallic_transition_metal_does_not_get_plus_u() -> None:
    """stfc/goldilocks-core#175's "metallic Fe gets U=5.3" bug: no anion,
    no correction."""
    iron = Structure.from_spacegroup("Im-3m", Lattice.cubic(2.87), ["Fe"], [[0, 0, 0]])

    state = _hubbard(iron)

    assert state.ok
    assert state.value.u_values == ()
    assert state.value.warnings == ()


def test_forced_needs_correlation_on_a_non_correlated_structure_gets_no_u() -> None:
    """Regression for #36: forcing needs_correlation on a structure with no
    transition metal, lanthanide or actinide (plain Si) must not produce
    a +U decision or Hubbard warnings."""
    silicon = Structure.from_spacegroup(
        "Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]]
    )

    state = _hubbard(silicon, human=HubbardUHumanInput(needs_correlation=True))

    assert state.ok
    assert state.value.u_values == ()
    assert state.value.warnings == ()


@pytest.mark.parametrize("cation", ["Fe", "Co", "Mn", "Ni"])
def test_canonical_mott_insulators_get_plus_u(cation: str) -> None:
    """stfc/goldilocks-core#175: FeO/CoO/MnO/NiO are the textbook O-bearing
    transition-metal Mott insulators GGA/LDA get wrong without +U."""
    state = _hubbard(_rock_salt(cation))

    assert [entry.element for entry in state.value.u_values] == [cation]


def test_no_u_element_does_not_take_the_tabulated_one_down_with_it() -> None:
    """#135: La2CuO4 used to stop at a La-4f calibration request although
    Cu has a value. La gets no U by convention, so Cu alone does."""
    state = _hubbard(_perovskite("La", "Cu"))

    (entry,) = state.value.u_values
    assert (entry.element, entry.manifold, entry.u_ev) == ("Cu", "3d", 4.0)
    assert "hubbard.u_value_missing" not in _codes(state)


def test_missing_value_warning_names_the_element_manifold_and_override() -> None:
    state = _hubbard(_rock_salt("Cf"))

    message = _warning(state, "hubbard.u_value_missing").message
    assert "Cf (5f)" in message
    assert "plain DFT" in message
    assert 'u_by_element={"Cf": <U in eV>}' in message
    assert _warning(state, "hubbard.u_value_missing").level == "warning"


def test_missing_value_warning_names_lawrencium_s_6d_manifold() -> None:
    state = _hubbard(_rock_salt("Lr"))

    assert "Lr (6d)" in _warning(state, "hubbard.u_value_missing").message


def test_actinium_warning_says_u_by_element_cannot_apply_a_u() -> None:
    """The only installed Ac pseudopotential has no 5f channel, so pw.x
    stops on any Ac-5f U: the warning must not suggest one."""
    state = _hubbard(_rock_salt("Ac"))

    message = _warning(state, "hubbard.u_value_missing").message
    assert "Ac (5f)" in message
    assert "plain DFT" in message
    assert "no installed Ac pseudopotential has a 5f channel" in message
    assert "u_by_element cannot apply one" in message
    assert "u_by_element={" not in message


def test_missing_value_suggestion_leaves_out_actinium() -> None:
    state = _hubbard(_perovskite("Ac", "Cf"))

    (californium,) = (
        warning.message
        for warning in state.value.warnings
        if warning.code == "hubbard.u_value_missing" and "Cf (5f)" in warning.message
    )
    assert 'u_by_element={"Cf": <U in eV>}' in californium


def test_missing_value_suggestion_keeps_the_elements_that_have_a_u() -> None:
    """u_by_element replaces the built-in table, so the suggested setting
    lists the tabulated U next to the missing one."""
    state = _hubbard(_perovskite("Cf", "Ni"))

    assert [entry.element for entry in state.value.u_values] == ["Ni"]
    message = _warning(state, "hubbard.u_value_missing").message
    assert 'u_by_element={"Ni": 6.2, "Cf": <U in eV>}' in message


def test_tabulated_u_carries_the_projector_and_starting_guess_warnings() -> None:
    state = _hubbard(_rock_salt("Ni"))

    assert _codes(state) == (
        "hubbard.projector_mismatch",
        "hubbard.starting_ns_eigenvalue_missing",
    )


def test_projector_mismatch_warning_names_the_vasp_derived_values() -> None:
    state = _hubbard(_rock_salt("Gd"))

    message = _warning(state, "hubbard.projector_mismatch").message
    assert "ortho-atomic" in message
    assert "VASP" in message
    assert "V, Cr, Mn, Fe, Co, Ni, Cu, Mo, W, Ce, Pa, U, Np, Pu" in message
    assert "Gd" not in message


def test_hybrid_functional_suppresses_plus_u_regardless_of_composition() -> None:
    """goldilocks-core-design.md:2626-2627's cross-parameter rule."""
    state = _hubbard(_rock_salt("Ni"), HYBRID_FUNCTIONAL)

    assert state.value.u_values == ()
    assert state.value.warnings == ()


def test_blocked_needs_correlation_propagates() -> None:
    nickel_oxide = _rock_salt("Ni")

    state = hubbard_u(nickel_oxide, Blocked(by="synthetic failure"), "PBEsol")

    assert not state.ok
    assert state.status == "blocked"
    assert state.root_cause() == "synthetic failure"


def test_human_can_force_off() -> None:
    state = _hubbard(
        _rock_salt("Ni"), human=HubbardUHumanInput(needs_correlation=False)
    )

    assert state.value.u_values == ()
    assert state.source == "human"


def test_llm_can_force_off() -> None:
    state = _hubbard(_rock_salt("Ni"), llm=HubbardULlmInput(needs_correlation=False))

    assert state.value.u_values == ()
    assert state.source == "llm"


def test_llm_can_force_the_table_on_a_sulfide() -> None:
    """The O/F screening rule leaves NiS alone; an LLM override applies the
    table anyway."""
    state = _hubbard(
        _rock_salt("Ni", "S"), llm=HubbardULlmInput(needs_correlation=True)
    )

    assert [(entry.element, entry.u_ev) for entry in state.value.u_values] == [
        ("Ni", 6.2)
    ]
    assert state.source == "llm"


def test_user_u_by_element_replaces_the_table() -> None:
    state = _hubbard(
        _rock_salt("Ru"), human=HubbardUHumanInput(u_by_element={"Ru": 4.0})
    )

    assert state.value.u_values == (HubbardU("Ru", "4d", 4.0, "human"),)
    assert state.source == "human"
    assert "hubbard.u_by_element_unused" not in _codes(state)


def test_user_u_for_an_absent_element_applies_no_u_and_says_so() -> None:
    """NiO with only an Fe value used to render an empty HUBBARD card."""
    state = _hubbard(
        _rock_salt("Ni"), human=HubbardUHumanInput(u_by_element={"Fe": 5.0})
    )

    assert state.value.u_values == ()
    message = _warning(state, "hubbard.u_by_element_unused").message
    assert "Fe" in message
    assert "No +U is applied" in message


def test_user_u_keeps_present_elements_and_reports_absent_ones() -> None:
    state = _hubbard(
        _rock_salt("Ni"),
        human=HubbardUHumanInput(u_by_element={"Ni": 6.2, "Fe": 5.0}),
    )

    assert state.value.u_values == (HubbardU("Ni", "3d", 6.2, "human"),)
    message = _warning(state, "hubbard.u_by_element_unused").message
    assert "Fe" in message
    assert "Ni" not in message
    assert "No +U is applied" not in message
