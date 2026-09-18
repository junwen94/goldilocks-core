from __future__ import annotations

import shutil

from pymatgen.core import Lattice, Structure

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


def test_default_listing_is_unranked_and_never_touches_mmace(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("rank_orderings must not be called unless requested")

    monkeypatch.setattr(magnetic_orderings_module, "rank_orderings", fail)

    report = list_magnetic_orderings(_ROCK_SALT_FEO)

    assert report.ranked is False
    assert report.warnings == ()
    assert [candidate.label for candidate in report.candidates] == ["fm"]
    assert report.candidates[0].energy_per_atom_ev is None
    assert report.candidates[0].is_recommended is False


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


def test_ranking_unavailable_falls_back_to_unranked_with_a_warning(monkeypatch) -> None:
    def fake_rank_orderings(candidates, *, device="cpu"):
        del candidates, device
        raise MagneticOrderingMlUnavailable("set GOLDILOCKS_MACE_BACKBONE ...")

    monkeypatch.setattr(
        magnetic_orderings_module, "rank_orderings", fake_rank_orderings
    )

    report = list_magnetic_orderings(_ROCK_SALT_FEO, rank_with_mmace=True)

    assert report.ranked is False
    assert [candidate.label for candidate in report.candidates] == ["fm"]
    assert len(report.warnings) == 1
    assert report.warnings[0].code == "magnetic.ordering_ranking_unavailable"
    assert "GOLDILOCKS_MACE_BACKBONE" in report.warnings[0].message
