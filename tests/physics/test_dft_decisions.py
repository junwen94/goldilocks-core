"""Physics regression suite, v2 shape (v2 epic 9, #9).

These tests used to exercise v1's compute()/legacy_analysis/advice/
parameters.py pipeline -- the acceptance-gate baseline recorded that way
in v2 epic 1 (#2), before v2 had a pipeline of its own to point at.
Epic 8 (#8) reconnected the delivery layers onto service.advise(); this
epic (#9) is where these tests move onto it too, calling the same
analysis/advisors functions the CLI/HTTP/MCP surface now calls, not the
whole-pipeline compute() entry point (deleted alongside the rest of v1's
tree at the end of this epic). Each test's own physical assertion is
unchanged from what it always was -- only the entry point moved.

QE-input-text assertions v1's own versions of these tests made (e.g.
"noncolin = .true." appears in the rendered file) are not repeated here:
that coverage now lives in tests/unit/test_generation_quantum_espresso_
scf.py, which tests the render step directly and in more depth than a
physics-focused test needs to. What belongs here is the *decision*, not
the rendering of it.
"""

from __future__ import annotations

from pymatgen.core import Lattice, Structure

from goldilocks_core.advisors.magnetic_config import (
    MagneticConfigHumanInput,
    magnetic_config,
)
from goldilocks_core.advisors.occupations import occupations
from goldilocks_core.advisors.pseudo_selection import (
    PseudoRequirements,
    pseudo_requirements,
    select_pseudopotential_table,
)
from goldilocks_core.analysis.composition import composition
from goldilocks_core.analysis.is_magnetic import is_magnetic
from goldilocks_core.analysis.is_metal import is_metal
from goldilocks_core.analysis.needs_soc import needs_soc
from goldilocks_core.assets.pseudopotentials.registry import PseudoTable
from goldilocks_core.assets.records import AssetFile, AssetSpec


def _table(
    table_id: str,
    *,
    provider: str = "pseudodojo",
    functional: str = "PBEsol",
    elements: tuple[str, ...] = ("Si",),
) -> PseudoTable:
    return PseudoTable(
        id=table_id,
        provider=provider,
        upstream_table="fixture",
        version="1",
        functional=functional,
        relativistic="scalar",
        accuracy="efficiency",
        licence="fixture licence",
        citation="fixture citation",
        elements=elements,
        asset=AssetSpec(
            f"pseudopotentials/{table_id}",
            "1",
            (AssetFile("pseudopotentials", "source/upf.tgz", "file:///fixture.tgz"),),
        ),
        charge_density_dual=4.0,
    )


def test_elemental_metal_uses_modest_cold_smearing_in_qe_rydberg_units() -> None:
    aluminium = Structure(Lattice.cubic(4.05), ["Al"], [[0.0, 0.0, 0.0]])

    metallicity = is_metal(aluminium)
    decision = occupations(metallicity, magnetic=None)

    # source="heuristic" is v2's replacement for v1's prose "Metallicity
    # was inferred from structure-only heuristics" warning -- the
    # provenance already carries that fact structurally, so no separate
    # warning duplicates it.
    assert metallicity.value == "metal"
    assert metallicity.source == "heuristic"
    assert decision.value.occupations == "smearing"
    assert decision.value.smearing_type == "cold"
    assert decision.value.degauss == 0.01


def test_heavy_element_prompts_for_soc_without_silently_enabling_it() -> None:
    iodine = Structure(Lattice.cubic(7.0), ["I"], [[0.0, 0.0, 0.0]])

    soc_needed = needs_soc(composition(iodine))
    magnetism = magnetic_config(iodine, is_magnetic(iodine), soc_needed)
    requirements = pseudo_requirements(
        "PBEsol", spin_orbit_enabled=magnetism.value.spin_orbit_enabled
    )

    assert soc_needed.value is True
    assert magnetism.value.spin_orbit_enabled is False
    assert requirements.relativistic == "scalar"
    assert any(
        warning.code == "magnetic.soc_suggested" for warning in magnetism.value.warnings
    )


def test_explicit_soc_couples_fully_relativistic_pseudos_to_qe_noncollinear_flags() -> (
    None
):
    iodine = Structure(Lattice.cubic(7.0), ["I"], [[0.0, 0.0, 0.0]])

    magnetism = magnetic_config(
        iodine,
        is_magnetic(iodine),
        human=MagneticConfigHumanInput(spin_orbit_coupling=True),
    )
    requirements = pseudo_requirements(
        "PBEsol", spin_orbit_enabled=magnetism.value.spin_orbit_enabled
    )

    assert magnetism.value.spin_orbit_enabled is True
    assert requirements.relativistic == "full"


def test_pseudopotential_functional_must_match_calculation_functional() -> None:
    pbe = _table("pbe-table", functional="PBE")
    pbesol = _table("pbesol-table", functional="PBEsol")
    requirements = PseudoRequirements(
        functional="PBEsol", accuracy="efficiency", relativistic="scalar"
    )

    selected = select_pseudopotential_table(
        {pbe.id: pbe, pbesol.id: pbesol},
        table_id=None,
        elements={"Si"},
        requirements=requirements,
    )

    assert selected.ok
    assert selected.value.id == pbesol.id


def test_lanthanide_element_is_forced_onto_sssp_even_when_pseudodojo_also_matches() -> (
    None
):
    """A1: PseudoDojo's lanthanide table freezes the 4f shell in the core and
    assumes a trivalent ion (wrong for Eu/Yb/Ce) and has no actinide coverage at
    all, so lanthanides/actinides must always be served from SSSP even when a
    PseudoDojo entry would otherwise satisfy every other requirement."""
    pseudodojo = _table("pseudodojo-ce", provider="pseudodojo", elements=("Ce",))
    sssp = _table("sssp-ce", provider="sssp", elements=("Ce",))
    requirements = PseudoRequirements(
        functional="PBEsol", accuracy="efficiency", relativistic="scalar"
    )

    selected = select_pseudopotential_table(
        {pseudodojo.id: pseudodojo, sssp.id: sssp},
        table_id=None,
        elements={"Ce"},
        requirements=requirements,
    )

    assert selected.ok
    assert selected.value.id == sssp.id
    assert selected.value.provider == "sssp"


def test_spin_polarized_structure_gets_a_starting_magnetization() -> None:
    """A2 (stfc/goldilocks-core#177): a spin-polarized structure must not
    reach QE with nspin=2 and no starting_magnetization -- that combination
    relaxes to the non-magnetic solution while the run reports a normal,
    converged SCF. Fixed in v2 epic 5's advisors/magnetic_config.py; this
    test used to stay xfail against v1's own compute() pipeline (never
    fixed there), and now exercises magnetic_config() directly instead."""
    iron = Structure(Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]])

    magnetism = magnetic_config(iron, is_magnetic(iron))

    assert magnetism.value.spin_polarized is True
    assert magnetism.value.starting_magnetization
    assert all(
        fraction != 0.0 for fraction in magnetism.value.starting_magnetization.values()
    )


def test_soc_does_not_silently_discard_a_magnetic_structures_magnetism() -> None:
    """A2b (found while hardening physics/ for v2 epic 1): enabling SOC on a
    structure independently advised as magnetic must not silently drop the
    magnetism and emit a non-magnetic noncollinear run. Fixed alongside A2 in
    v2 epic 5's advisors/magnetic_config.py."""
    iron = Structure(Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]])

    magnetism = magnetic_config(
        iron,
        is_magnetic(iron),
        human=MagneticConfigHumanInput(spin_orbit_coupling=True),
    )

    assert magnetism.value.spin_polarized is True
    assert magnetism.value.spin_orbit_enabled is True
    assert magnetism.value.starting_magnetization
