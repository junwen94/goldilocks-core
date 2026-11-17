"""Tests for goldilocks_core.server.documents: the HTTP and MCP request
contract."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError
from support import INTERNAL_TEXT

from goldilocks_core.server.documents import (
    ComputeRequestDocument,
    InlineStructureDocument,
    MagneticOrderingsRequestDocument,
    MagneticOrderingsToolDocument,
    RunRequestDocument,
)

_DOCUMENTS: tuple[type[BaseModel], ...] = (
    InlineStructureDocument,
    ComputeRequestDocument,
    RunRequestDocument,
    MagneticOrderingsRequestDocument,
    MagneticOrderingsToolDocument,
)


class TestInlineStructureDocument:
    def test_accepts_real_multiline_cif_text(self) -> None:
        document = InlineStructureDocument(
            structure_content="data_si\n_cell_length_a 5.43\n..."
        )

        assert document.structure_content.startswith("data_si")

    def test_rejects_a_bare_path_looking_value(self) -> None:
        with pytest.raises(ValidationError, match="not a path"):
            InlineStructureDocument(structure_content="/etc/passwd")

    def test_rejects_empty_content(self) -> None:
        with pytest.raises(ValidationError):
            InlineStructureDocument(structure_content="   ")

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            InlineStructureDocument(
                structure_content="line one\nline two", structure_path="/tmp/x.cif"
            )

    @pytest.mark.parametrize("name", ["a/b.cif", "..", "  ", "a\\b"])
    def test_rejects_a_structure_name_that_is_not_one_file_name(
        self, name: str
    ) -> None:
        with pytest.raises(ValidationError, match="one file name"):
            InlineStructureDocument(
                structure_content="line one\nline two", structure_name=name
            )

    def test_defaults_name_and_format(self) -> None:
        document = InlineStructureDocument(structure_content="line one\nline two")

        assert document.structure_name == "structure"
        assert document.structure_format is None

    def test_rejects_an_invalid_format(self) -> None:
        with pytest.raises(ValidationError):
            InlineStructureDocument(
                structure_content="line one\nline two", structure_format="xyz"
            )


class TestComputeRequestDocument:
    def test_defaults_match_the_one_real_code_and_task(self) -> None:
        document = ComputeRequestDocument(structure_content="line one\nline two")

        assert document.code == "quantum_espresso"
        assert document.task == "scf_single_point"
        assert document.hpc is None
        assert document.overrides == {}

    def test_accepts_arbitrary_json_typed_overrides(self) -> None:
        document = ComputeRequestDocument(
            structure_content="line one\nline two",
            overrides={"ecutwfc_ry": 30.0, "k_grid": [4, 4, 4]},
        )

        assert document.overrides["ecutwfc_ry"] == 30.0
        assert document.overrides["k_grid"] == [4, 4, 4]

    @pytest.mark.parametrize(
        "field", [{"pseudo_root": "/tmp"}, {"fetch_missing": True}]
    )
    def test_rejects_unknown_top_level_fields(self, field: dict[str, Any]) -> None:
        """``fetch_missing`` included: only the operator installs assets on a
        server (#299)."""
        with pytest.raises(ValidationError):
            ComputeRequestDocument(structure_content="line one\nline two", **field)

    def test_rejects_a_code_core_does_not_write(self) -> None:
        with pytest.raises(ValidationError):
            ComputeRequestDocument(structure_content="line one\nline two", code="vasp")


class TestRunRequestDocument:
    def test_defaults_to_json_response(self) -> None:
        document = RunRequestDocument(structure_content="line one\nline two")

        assert document.respond_with == "json"

    def test_rejects_an_unknown_respond_with_value(self) -> None:
        with pytest.raises(ValidationError):
            RunRequestDocument(
                structure_content="line one\nline two", respond_with="xml"
            )


def _descriptions(schema: Any) -> list[str]:
    if isinstance(schema, dict):
        own = [schema["description"]] if "description" in schema else []
        return own + [
            text for value in schema.values() for text in _descriptions(value)
        ]
    if isinstance(schema, list):
        return [text for value in schema for text in _descriptions(value)]
    return []


@pytest.mark.parametrize("document", _DOCUMENTS, ids=lambda d: d.__name__)
def test_every_field_tells_the_caller_what_it_is(document: type[BaseModel]) -> None:
    """The schema is what an MCP client's model and an OpenAPI reader see."""
    properties = document.model_json_schema()["properties"]

    assert [
        name for name, field in properties.items() if not field.get("description")
    ] == []


@pytest.mark.parametrize("document", _DOCUMENTS, ids=lambda d: d.__name__)
def test_schema_descriptions_read_as_user_text(document: type[BaseModel]) -> None:
    """Class docstrings become schema descriptions too."""
    descriptions = _descriptions(document.model_json_schema())

    assert [text for text in descriptions if INTERNAL_TEXT.search(text)] == []
