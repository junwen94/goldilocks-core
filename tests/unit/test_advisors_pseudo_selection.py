from __future__ import annotations

from goldilocks_core.advisors.pseudo_selection import (
    PseudoRequirements,
    is_table_eligible,
    match_selected_metadata,
    pseudo_requirements,
    requires_sssp,
    select_metadata_for_elements,
    select_pseudopotential_table,
)
from goldilocks_core.assets.pseudopotentials.registry import PseudoTable
from goldilocks_core.assets.pseudopotentials.upf import PseudoMetadata
from goldilocks_core.assets.records import AssetFile, AssetSpec
from goldilocks_core.provenance import Provenance as LegacyProvenance


def _table(
    table_id: str,
    *,
    provider: str = "pseudodojo",
    functional: str = "PBEsol",
    accuracy: str = "efficiency",
    relativistic: str = "scalar",
    elements: tuple[str, ...] = ("Si",),
) -> PseudoTable:
    return PseudoTable(
        id=table_id,
        provider=provider,
        upstream_table="fixture",
        version="1",
        functional=functional,
        relativistic=relativistic,
        accuracy=accuracy,
        licence="fixture licence",
        citation="fixture citation",
        elements=elements,
        asset=AssetSpec(
            f"pseudopotentials/{table_id}",
            "1",
            (AssetFile("pseudopotentials", "source/upf.tgz", "file:///fixture.tgz"),),
        ),
        charge_density_dual=4.0 if provider == "pseudodojo" else None,
    )


def _requirements(**overrides: object) -> PseudoRequirements:
    defaults = {
        "functional": "PBEsol",
        "accuracy": "efficiency",
        "relativistic": "scalar",
    }
    defaults.update(overrides)
    return PseudoRequirements(**defaults)


def test_requires_sssp_is_true_for_lanthanides_and_actinides() -> None:
    assert requires_sssp({"La"}) is True
    assert requires_sssp({"U"}) is True
    assert requires_sssp({"Si", "O"}) is False


def test_pseudo_requirements_relativistic_follows_spin_orbit_enabled() -> None:
    on = pseudo_requirements("PBEsol", spin_orbit_enabled=True)
    off = pseudo_requirements("PBEsol", spin_orbit_enabled=False)

    assert on.relativistic == "full"
    assert off.relativistic == "scalar"


def test_explicit_table_id_resolves_when_compatible() -> None:
    tables = {"fixture": _table("fixture")}

    state = select_pseudopotential_table(
        tables, table_id="fixture", elements={"Si"}, requirements=_requirements()
    )

    assert state.ok
    assert state.value.id == "fixture"


def test_explicit_unknown_table_id_is_unavailable_not_a_raise() -> None:
    """v2 epic 5 (#5) bug fix: v1 raised PseudoTableMismatch here."""
    state = select_pseudopotential_table(
        {"fixture": _table("fixture")},
        table_id="not-a-table",
        elements={"Si"},
        requirements=_requirements(),
    )

    assert not state.ok
    assert state.status == "unavailable"
    assert "unknown pseudopotential table" in state.reason


def test_explicit_table_id_mismatched_functional_is_unavailable_with_alternatives() -> (
    None
):
    tables = {
        "wrong": _table("wrong", functional="LDA"),
        "right": _table("right", functional="PBEsol"),
    }

    state = select_pseudopotential_table(
        tables, table_id="wrong", elements={"Si"}, requirements=_requirements()
    )

    assert not state.ok
    assert "functional is LDA, requested PBEsol" in state.reason
    assert "right" in state.reason


def test_automatic_selection_picks_the_only_compatible_table() -> None:
    tables = {"fixture": _table("fixture")}

    state = select_pseudopotential_table(
        tables, table_id=None, elements={"Si"}, requirements=_requirements()
    )

    assert state.ok
    assert state.value.id == "fixture"


def test_automatic_selection_prefers_sssp_for_lanthanides() -> None:
    tables = {
        "dojo": _table("dojo", provider="pseudodojo", elements=("La",)),
        "sssp": _table("sssp", provider="sssp", elements=("La",)),
    }

    state = select_pseudopotential_table(
        tables, table_id=None, elements={"La"}, requirements=_requirements()
    )

    assert state.value.provider == "sssp"


def test_automatic_selection_excludes_dojo_tables_missing_lanthanide_coverage() -> None:
    """is_table_eligible's rule, exercised through select_pseudopotential_table:
    a PseudoDojo table is never eligible for lanthanide/actinide elements
    even if it happens to list them, because requires_sssp forces sssp."""
    sssp_table = _table("sssp", provider="sssp", elements=("La",))

    assert is_table_eligible(_table("dojo", elements=("La",)), {"La"}) is False
    assert is_table_eligible(sssp_table, {"La"}) is True


def test_automatic_selection_with_no_match_is_unavailable_not_a_raise() -> None:
    tables = {"fixture": _table("fixture", elements=("Si",))}

    state = select_pseudopotential_table(
        tables, table_id=None, elements={"Fe"}, requirements=_requirements()
    )

    assert not state.ok
    assert "no pseudopotential table satisfies" in state.reason


class TestMatchSelectedMetadata:
    def test_matches_a_selection_to_its_metadata(self) -> None:
        metadata = (
            PseudoMetadata(
                filepath="/store/Si.upf",
                filename="Si.upf",
                header_format="attr",
                element="Si",
                table_id="pseudopotentials/fixture",
            ),
        )
        selection = {
            "pseudopotentials": [
                {
                    "element": "Si",
                    "filename": "Si.upf",
                    "filepath": "/store/Si.upf",
                    "provenance": LegacyProvenance(
                        source="analysis",
                        reason="fixture",
                        data_source="pseudopotentials/fixture",
                    ),
                }
            ],
            "warnings": [],
        }

        state = match_selected_metadata(selection, metadata)

        assert state.ok
        assert state.value[0].element == "Si"

    def test_unmatched_selection_is_unavailable_not_a_stop_iteration(self) -> None:
        """v2 epic 5 (#5) bug fix: v1's bare next(...) here raised an opaque
        StopIteration on a miss (pseudo/source.py:114-129)."""
        selection = {
            "pseudopotentials": [
                {
                    "element": "Si",
                    "filename": "Si.upf",
                    "filepath": "/store/Si.upf",
                    "provenance": LegacyProvenance(
                        source="analysis",
                        reason="fixture",
                        data_source="pseudopotentials/fixture",
                    ),
                }
            ],
            "warnings": [],
        }

        state = match_selected_metadata(selection, ())

        assert not state.ok
        assert state.status == "unavailable"
        assert "Si" in state.reason


class TestSelectMetadataForElements:
    def test_selects_exactly_the_requested_elements(self) -> None:
        metadata = (
            PseudoMetadata(
                filepath="/store/Si.upf",
                filename="Si.upf",
                header_format="attr",
                element="Si",
            ),
            PseudoMetadata(
                filepath="/store/O.upf",
                filename="O.upf",
                header_format="attr",
                element="O",
            ),
            PseudoMetadata(
                filepath="/store/Fe.upf",
                filename="Fe.upf",
                header_format="attr",
                element="Fe",
            ),
        )

        state = select_metadata_for_elements(metadata, {"Si", "O"})

        assert state.ok
        assert {item.element for item in state.value} == {"Si", "O"}

    def test_missing_element_is_unavailable_not_a_raise(self) -> None:
        metadata = (
            PseudoMetadata(
                filepath="/store/Si.upf",
                filename="Si.upf",
                header_format="attr",
                element="Si",
            ),
        )

        state = select_metadata_for_elements(metadata, {"Si", "O"})

        assert not state.ok
        assert state.status == "unavailable"
        assert "O" in state.reason

    def test_duplicate_element_entries_are_unavailable_not_silently_first_picked(
        self,
    ) -> None:
        metadata = (
            PseudoMetadata(
                filepath="/store/Si-a.upf",
                filename="Si-a.upf",
                header_format="attr",
                element="Si",
            ),
            PseudoMetadata(
                filepath="/store/Si-b.upf",
                filename="Si-b.upf",
                header_format="attr",
                element="Si",
            ),
        )

        state = select_metadata_for_elements(metadata, {"Si"})

        assert not state.ok
        assert state.status == "unavailable"
        assert "Si" in state.reason

    def test_result_is_deterministically_ordered_by_element(self) -> None:
        metadata = (
            PseudoMetadata(
                filepath="/store/O.upf",
                filename="O.upf",
                header_format="attr",
                element="O",
            ),
            PseudoMetadata(
                filepath="/store/Fe.upf",
                filename="Fe.upf",
                header_format="attr",
                element="Fe",
            ),
        )

        state = select_metadata_for_elements(metadata, {"O", "Fe"})

        assert state.ok
        assert [item.element for item in state.value] == ["Fe", "O"]
