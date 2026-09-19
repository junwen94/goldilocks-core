"""Tests for goldilocks_core.server._handlers (v2 epic 8, #8) --
transport-agnostic, so these run against real assets without needing
FastAPI or the mcp package installed at all.
"""

from __future__ import annotations

import pytest

from goldilocks_core.server import _handlers
from goldilocks_core.server.documents import (
    ComputeRequestDocument,
    InlineStructureDocument,
    MagneticOrderingsRequestDocument,
)
from goldilocks_core.service import AdviceIncomplete


class TestInspect:
    def test_returns_the_real_structure_inspection(
        self, real_assets, silicon_cif
    ) -> None:
        document = InlineStructureDocument(structure_content=silicon_cif)

        inspection = _handlers.inspect(document)

        assert inspection["structure"]["reduced_formula"] == "Si"


class TestMagneticOrderings:
    def test_lists_the_fm_candidate_unranked_by_default(self, silicon_cif) -> None:
        document = MagneticOrderingsRequestDocument(structure_content=silicon_cif)

        result = _handlers.magnetic_orderings(document)

        assert result["ranked"] is False
        assert result["candidates"] == [
            {
                "label": "fm",
                "formula": "Si",
                "natoms": 8,
                "energy_per_atom_ev": None,
                "status": None,
                "is_recommended": False,
            }
        ]
        assert result["warnings"] == []

    def test_rank_with_mmace_degrades_with_a_warning_when_unconfigured(
        self, silicon_cif, monkeypatch
    ) -> None:
        monkeypatch.delenv("GOLDILOCKS_MACE_BACKBONE", raising=False)
        document = MagneticOrderingsRequestDocument(
            structure_content=silicon_cif, rank_with_mmace=True
        )

        result = _handlers.magnetic_orderings(document)

        assert result["ranked"] is False
        assert len(result["warnings"]) == 1
        assert result["warnings"][0]["code"] == "magnetic.ordering_ranking_unavailable"


class TestExplain:
    def test_returns_records_and_warnings(self, real_assets, silicon_cif) -> None:
        document = ComputeRequestDocument(structure_content=silicon_cif, hpc="scarf")

        result = _handlers.explain(document)

        assert result["records"]["functional"]["value"] == "PBEsol"
        assert result["records"]["functional"]["status"] == "resolved"
        assert isinstance(result["warnings"], list)
        assert all(
            warning.keys() == {"code", "level", "category", "message"}
            for warning in result["warnings"]
        )

    def test_never_raises_for_a_scientifically_incomplete_request(
        self, real_assets, silicon_cif
    ) -> None:
        """The "diagnosis is always available" promise -- an unknown
        pseudo_table_id degrades every dependent field to Blocked, but
        explain() itself must still succeed."""
        document = ComputeRequestDocument(
            structure_content=silicon_cif,
            hpc="scarf",
            overrides={"pseudo_table_id": "does-not-exist"},
        )

        result = _handlers.explain(document)

        assert result["records"]["pseudo_table"]["status"] == "unavailable"
        assert result["records"]["cutoffs"]["status"] == "blocked"


class TestRun:
    def test_produces_a_complete_bundle_summary(self, real_assets, silicon_cif) -> None:
        document = ComputeRequestDocument(structure_content=silicon_cif, hpc="scarf")

        summary, bundle_input = _handlers.run(document)

        assert "scf.in" in summary["files"]
        assert "submit.sh" in summary["files"]
        assert any(path.endswith(".upf") for path in summary["files"])
        assert summary["records"]["functional"]["value"] == "PBEsol"
        assert any(bundle_input.records)

    def test_raises_advice_incomplete_when_blocked(
        self, real_assets, silicon_cif
    ) -> None:
        document = ComputeRequestDocument(
            structure_content=silicon_cif,
            hpc="scarf",
            overrides={"pseudo_table_id": "does-not-exist"},
        )

        with pytest.raises(AdviceIncomplete) as excinfo:
            _handlers.run(document)

        # AdviceIncomplete's own dedup (service/_generate.py) -- must not
        # repeat the same root cause once per field it blocks.
        message = str(excinfo.value)
        assert message.count("unknown pseudopotential table") == 1

    def test_overrides_flow_through_to_the_generated_input(
        self, real_assets, silicon_cif
    ) -> None:
        document = ComputeRequestDocument(
            structure_content=silicon_cif,
            hpc="scarf",
            overrides={"functional": "PBE"},
        )

        summary, _bundle_input = _handlers.run(document)

        assert summary["records"]["functional"]["value"] == "PBE"
        assert summary["records"]["functional"]["source"] == "human"


class TestBuildReadiness:
    def test_returns_a_working_readiness_checker(self, real_assets) -> None:
        readiness = _handlers.build_readiness()

        report = readiness.check()

        assert report.ready
