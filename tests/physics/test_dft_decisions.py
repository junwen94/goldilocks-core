from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core import (
    CalculationDraft,
    CalculationHints,
    CalculationIntent,
    ComputeRequest,
    InMemoryStructureSource,
    PresetSelection,
    Runtime,
    compute,
)
from goldilocks_core.advice.parameters import ParameterAdvice
from goldilocks_core.assets.pseudopotentials.upf import PseudoMetadata
from goldilocks_core.assets.store import AssetStore
from goldilocks_core.generation.files import GeneratedFiles
from goldilocks_core.legacy_analysis import StructureAnalysisRecord
from goldilocks_core.selection import SelectionRecord


def test_elemental_metal_uses_modest_cold_smearing_in_qe_rydberg_units(
    pseudo_metadata_factory: Callable[..., PseudoMetadata],
    tmp_path: Path,
) -> None:
    aluminium = Structure(Lattice.cubic(4.05), ["Al"], [[0.0, 0.0, 0.0]])
    request = ComputeRequest(
        draft=CalculationDraft(
            structure=InMemoryStructureSource(aluminium),
            hints=CalculationHints(k_grid=(8, 8, 8)),
            pseudo_metadata=(
                pseudo_metadata_factory("Al", root=tmp_path, materialize=True),
            ),
        ),
        selection=PresetSelection("generate"),
    )
    with Runtime(asset_store=AssetStore(tmp_path / "assets")) as runtime:
        result = compute(request, runtime=runtime)
    analysis = result.records[StructureAnalysisRecord]
    advice = result.records[ParameterAdvice]

    assert analysis["electronic_character"] == "likely_metal"
    assert advice["smearing"]["smearing_type"] == "cold"
    assert advice["smearing"]["width_ry"] == 0.01
    qe_input = result.records[GeneratedFiles][0]["content"]
    assert "  occupations = 'smearing'" in qe_input
    assert "  smearing = 'cold'" in qe_input
    assert "  degauss = 0.01" in qe_input
    assert "Metallicity was inferred from structure-only heuristics." in result.warnings


def test_heavy_element_prompts_for_soc_without_silently_enabling_it() -> None:
    iodine = Structure(Lattice.cubic(7.0), ["I"], [[0.0, 0.0, 0.0]])
    result = compute(
        ComputeRequest(
            draft=CalculationDraft(
                structure=InMemoryStructureSource(iodine),
                hints=CalculationHints(k_grid=(2, 2, 2)),
                pseudo_metadata=(),
            ),
            selection=PresetSelection("recommend"),
        )
    )
    analysis = result.records[StructureAnalysisRecord]
    advice = result.records[ParameterAdvice]

    assert analysis["heavy_elements"] == [
        "I",
    ]
    assert advice["spin_orbit"]["consider"] is True
    assert advice["spin_orbit"]["enabled"] is False
    assert advice["pseudopotential_requirements"]["relativistic"] == "scalar"
    assert "SOC is not enabled automatically" in " ".join(result.warnings)


def test_explicit_soc_couples_fully_relativistic_pseudos_to_qe_noncollinear_flags(
    pseudo_metadata_factory: Callable[..., PseudoMetadata],
    tmp_path: Path,
) -> None:
    iodine = Structure(Lattice.cubic(7.0), ["I"], [[0.0, 0.0, 0.0]])
    result = compute(
        ComputeRequest(
            draft=CalculationDraft(
                structure=InMemoryStructureSource(iodine),
                hints=CalculationHints(k_grid=(2, 2, 2), spin_orbit_coupling=True),
                pseudo_metadata=(
                    pseudo_metadata_factory(
                        "I",
                        relativistic="full",
                        root=tmp_path,
                        materialize=True,
                    ),
                ),
            ),
            selection=PresetSelection("generate"),
        )
    )
    advice = result.records[ParameterAdvice]

    assert advice["spin_orbit"]["enabled"] is True
    assert advice["spin_orbit"]["consider"] is False
    assert advice["pseudopotential_requirements"]["relativistic"] == "full"
    assert result.records[SelectionRecord]["pseudopotentials"][0]["filename"] == "I.UPF"
    qe_input = result.records[GeneratedFiles][0]["content"]
    assert "  noncolin = .true." in qe_input
    assert "  lspinorb = .true." in qe_input
    assert "  nspin = 2" not in qe_input


def test_pseudopotential_functional_must_match_calculation_functional(
    silicon_structure: Structure,
    pseudo_metadata_factory: Callable[..., PseudoMetadata],
) -> None:
    pbe = pseudo_metadata_factory("Si", functional="PBE", root=Path("/pbe"))
    pbesol = pseudo_metadata_factory("Si", functional="PBEsol", root=Path("/pbesol"))
    result = compute(
        ComputeRequest(
            draft=CalculationDraft(
                structure=InMemoryStructureSource(silicon_structure),
                intent=CalculationIntent(functional="PBEsol"),
                hints=CalculationHints(k_grid=(4, 4, 4)),
                pseudo_metadata=(pbe, pbesol),
            ),
            selection=PresetSelection("recommend"),
        )
    )
    advice = result.records[ParameterAdvice]
    selection = result.records[SelectionRecord]

    assert advice["pseudopotential_requirements"]["functional"] == "PBEsol"
    assert selection["pseudopotentials"][0]["filepath"] == pbesol.filepath
    assert selection["pseudopotentials"][0]["filepath"] != pbe.filepath


def test_lanthanide_element_is_forced_onto_sssp_even_when_pseudodojo_also_matches(
    pseudo_metadata_factory: Callable[..., PseudoMetadata],
) -> None:
    """A1: PseudoDojo's lanthanide table freezes the 4f shell in the core and
    assumes a trivalent ion (wrong for Eu/Yb/Ce) and has no actinide coverage at
    all, so lanthanides/actinides must always be served from SSSP even when a
    PseudoDojo entry would otherwise satisfy every other requirement."""
    cerium = Structure(Lattice.cubic(5.16), ["Ce"], [[0.0, 0.0, 0.0]])
    pseudodojo = replace(
        pseudo_metadata_factory("Ce", functional="PBEsol", root=Path("/pseudodojo")),
        provider="pseudodojo",
    )
    sssp = replace(
        pseudo_metadata_factory("Ce", functional="PBEsol", root=Path("/sssp")),
        provider="sssp",
    )
    result = compute(
        ComputeRequest(
            draft=CalculationDraft(
                structure=InMemoryStructureSource(cerium),
                intent=CalculationIntent(functional="PBEsol"),
                hints=CalculationHints(k_grid=(4, 4, 4)),
                pseudo_metadata=(pseudodojo, sssp),
            ),
            selection=PresetSelection("recommend"),
        )
    )
    selection = result.records[SelectionRecord]

    assert selection["pseudopotentials"][0]["filepath"] == sssp.filepath
    assert selection["pseudopotentials"][0]["filepath"] != pseudodojo.filepath


@pytest.mark.xfail(
    strict=True,
    reason="A2 (stfc/goldilocks-core#177, still open): _spin_lines emits "
    "nspin=2 with no companion starting_magnetization, so QE relaxes to the "
    "non-magnetic solution while the run reports a normal, converged SCF. "
    "Fixed by v2 epic 5's magnetic_config advisor.",
)
def test_spin_polarized_structure_gets_a_starting_magnetization(
    pseudo_metadata_factory: Callable[..., PseudoMetadata],
    tmp_path: Path,
) -> None:
    iron = Structure(Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]])
    result = compute(
        ComputeRequest(
            draft=CalculationDraft(
                structure=InMemoryStructureSource(iron),
                hints=CalculationHints(k_grid=(8, 8, 8)),
                pseudo_metadata=(
                    pseudo_metadata_factory("Fe", root=tmp_path, materialize=True),
                ),
            ),
            selection=PresetSelection("generate"),
        )
    )
    advice = result.records[ParameterAdvice]
    qe_input = result.records[GeneratedFiles][0]["content"]

    assert advice["magnetism"]["spin_polarized"] is True
    assert "  nspin = 2" in qe_input
    assert "starting_magnetization" in qe_input


@pytest.mark.xfail(
    strict=True,
    reason="A2b (found while hardening physics/ for v2 epic 1, not in the "
    "original #177 report): _spin_lines lets spin_orbit.enabled short-circuit "
    "past magnetism.spin_polarized entirely, so enabling SOC on a structure "
    "independently advised as magnetic silently drops the magnetism and emits "
    "a non-magnetic noncollinear run. Fixed alongside A2 by v2 epic 5's "
    "magnetic_config advisor.",
)
def test_soc_does_not_silently_discard_a_magnetic_structures_magnetism(
    pseudo_metadata_factory: Callable[..., PseudoMetadata],
    tmp_path: Path,
) -> None:
    iron = Structure(Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]])
    result = compute(
        ComputeRequest(
            draft=CalculationDraft(
                structure=InMemoryStructureSource(iron),
                hints=CalculationHints(k_grid=(8, 8, 8), spin_orbit_coupling=True),
                pseudo_metadata=(
                    pseudo_metadata_factory(
                        "Fe",
                        relativistic="full",
                        root=tmp_path,
                        materialize=True,
                    ),
                ),
            ),
            selection=PresetSelection("generate"),
        )
    )
    advice = result.records[ParameterAdvice]
    qe_input = result.records[GeneratedFiles][0]["content"]

    assert advice["magnetism"]["spin_polarized"] is True
    assert "  noncolin = .true." in qe_input
    assert "  lspinorb = .true." in qe_input
    assert "starting_magnetization" in qe_input
