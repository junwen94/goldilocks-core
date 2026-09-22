from __future__ import annotations

import shutil

from pymatgen.core import Lattice, Species, Structure

from goldilocks_core.advisors import magnetic_config as magnetic_config_module
from goldilocks_core.advisors.magnetic_ordering_ml import (
    MagneticOrderingMlUnavailable,
    OrderingCandidate,
    OrderingMlResult,
)
from goldilocks_core.service import _magnetic_orderings as magnetic_orderings_module
from goldilocks_core.service._magnetic_orderings import list_magnetic_orderings

_ROCK_SALT_FEO = Structure(
    Lattice.cubic(4.3), ["Fe", "O"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
)

# Mimics real `MagneticStructureEnumerator` output: two Fe sites carrying
# opposite `Species.spin`, the sign information `_label_by_spin` splits into
# distinct QE labels (Fe1/Fe2) and `_bundle_inputs` must preserve through a
# CIF round-trip (see test_afm_overrides_preserve_sublattice_sign below).
_SPIN_SPLIT_AFM_FEO = Structure(
    Lattice.cubic(4.3),
    [Species("Fe", spin=5.0), Species("Fe", spin=-5.0), "O", "O"],
    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
)


def test_default_listing_is_unranked_and_never_touches_mmace(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("rank_orderings must not be called unless requested")

    monkeypatch.setattr(magnetic_orderings_module, "rank_orderings", fail)
    # Deterministic regardless of this machine's real PATH (#87): forces
    # AFM enumeration itself unavailable, so the resulting warning is
    # pinned rather than depending on whether enum.x happens to be
    # installed here.
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    report = list_magnetic_orderings(_ROCK_SALT_FEO)

    assert report.ranked is False
    assert [candidate.label for candidate in report.candidates] == ["fm"]
    assert report.candidates[0].energy_per_atom_ev is None
    assert report.candidates[0].is_recommended is False
    [warning] = report.warnings
    assert warning.code == "magnetic.afm_ordering_unavailable"
    assert "enumlib" in warning.message


def test_ranked_listing_marks_the_lowest_energy_candidate_as_recommended(
    monkeypatch,
) -> None:
    class _OneAfmCandidate:
        def __init__(self, *_args, **_kwargs) -> None:
            self.ordered_structures = [_ROCK_SALT_FEO.copy()]
            self.ordered_structure_origins = ["afm"]

    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/enum.x")
    monkeypatch.setattr(
        magnetic_config_module, "MagneticStructureEnumerator", _OneAfmCandidate
    )

    def fake_rank_orderings(candidates, *, device="cpu"):
        del device
        results = tuple(
            OrderingCandidate(
                label=label,
                structure=structure,
                energy_per_atom_ev=-1.0 if label == "afm-1" else 0.0,
                status="ok",
            )
            for label, structure in candidates
        )
        winner = min(results, key=lambda result: result.energy_per_atom_ev)
        return OrderingMlResult(
            winner=winner, candidates=results, model_id="stub.model"
        )

    monkeypatch.setattr(
        magnetic_orderings_module, "rank_orderings", fake_rank_orderings
    )

    report = list_magnetic_orderings(_ROCK_SALT_FEO, rank_with_mmace=True)

    assert report.ranked is True
    assert report.warnings == ()
    by_label = {candidate.label: candidate for candidate in report.candidates}
    assert by_label["fm"].energy_per_atom_ev == 0.0
    assert by_label["fm"].is_recommended is False
    assert by_label["afm-1"].energy_per_atom_ev == -1.0
    assert by_label["afm-1"].is_recommended is True


def test_afm_overrides_preserve_sublattice_sign(monkeypatch) -> None:
    """Regression for the CIF-round-trip trap `_bundle_inputs` documents:
    naively handing a candidate's structure_content back to `/run` with no
    overrides would silently resolve a ferromagnetic starting_magnetization
    (CIF drops `Species.spin`) despite the species labels still looking
    AFM-split. The override must carry both signs, keyed by whatever labels
    the structure_content will actually reload as."""

    class _SpinSplitAfmCandidate:
        def __init__(self, *_args, **_kwargs) -> None:
            self.ordered_structures = [_SPIN_SPLIT_AFM_FEO.copy()]
            self.ordered_structure_origins = ["afm"]

    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/enum.x")
    monkeypatch.setattr(
        magnetic_config_module, "MagneticStructureEnumerator", _SpinSplitAfmCandidate
    )

    report = list_magnetic_orderings(_ROCK_SALT_FEO)

    fm = next(candidate for candidate in report.candidates if candidate.label == "fm")
    assert fm.overrides == {}

    afm = next(
        candidate for candidate in report.candidates if candidate.label == "afm-1"
    )
    assert afm.overrides["spin_polarized"] is True
    fractions = afm.overrides["starting_magnetization"]
    reloaded = Structure.from_str(afm.structure_content, fmt="cif")
    assert set(fractions) == {site.label for site in reloaded}
    assert {value > 0 for value in fractions.values()} == {True, False}


def test_ranking_unavailable_falls_back_to_unranked_with_a_warning(monkeypatch) -> None:
    def fake_rank_orderings(candidates, *, device="cpu"):
        del candidates, device
        raise MagneticOrderingMlUnavailable("set GOLDILOCKS_MACE_BACKBONE ...")

    monkeypatch.setattr(
        magnetic_orderings_module, "rank_orderings", fake_rank_orderings
    )
    # Deterministic regardless of this machine's real PATH (#87) -- see
    # test_default_listing_is_unranked_and_never_touches_mmace above.
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    report = list_magnetic_orderings(_ROCK_SALT_FEO, rank_with_mmace=True)

    assert report.ranked is False
    assert [candidate.label for candidate in report.candidates] == ["fm"]
    codes = {warning.code for warning in report.warnings}
    assert codes == {
        "magnetic.afm_ordering_unavailable",
        "magnetic.ordering_ranking_unavailable",
    }
    ranking_warning = next(
        warning
        for warning in report.warnings
        if warning.code == "magnetic.ordering_ranking_unavailable"
    )
    assert "GOLDILOCKS_MACE_BACKBONE" in ranking_warning.message
