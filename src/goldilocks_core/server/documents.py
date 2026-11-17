"""Request documents shared by the HTTP and MCP transports.

- A structure arrives as file text (``structure_content``), never a path:
  remote transports do not read the server's filesystem
  (goldilocks-agent-design.md S9.1). The CLI, a local process, reads paths.
- ``overrides`` is a flat ``{key: value}`` object validated at request time
  by ``set_overrides.build_overrides`` against ``capabilities.bindings()``,
  so the settable keys have one source instead of a second schema here.
- Every document forbids unknown fields. Field descriptions and class
  docstrings are published in the MCP tool schemas and in OpenAPI, so they
  are written for callers; design notes stay in this module docstring.
- The response documents are TypedDicts owned by their domains; the names
  below give both transports one set of response types.
- A failure is ``{"error": {kind, message, details}}`` on both transports:
  ``ExpectedFailure.public_error()`` for a refused request, ``invalid_request``
  for one that does not match these documents.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

from goldilocks_core.capabilities import Capabilities
from goldilocks_core.checks import FindingDocument
from goldilocks_core.inputs.structure import StructureInspection, require_filename
from goldilocks_core.resolution import RecordWarning, ResolvedField
from goldilocks_core.server.ranking_jobs import RankingStatus
from goldilocks_core.service import MagneticOrderingsDocument, RecordLayout
from goldilocks_core.types import CalcTask, CodeName

_STRICT = ConfigDict(extra="forbid")


class InlineStructureDocument(BaseModel):
    """A crystal structure sent as the text of a CIF or POSCAR file."""

    model_config = _STRICT

    structure_content: str = Field(
        description=(
            "The full text of a CIF or POSCAR file. Never a file path: the "
            "server does not read paths."
        )
    )
    structure_name: str = Field(
        "structure",
        description=(
            "A file name for the structure, with no directories. A .cif or "
            ".poscar suffix sets the format when structure_format is omitted."
        ),
    )
    structure_format: Literal["cif", "poscar"] | None = Field(
        None,
        description=(
            "The format of structure_content. When omitted, it is taken from "
            "structure_name's suffix, else from a leading 'data_' line (CIF), "
            "else POSCAR."
        ),
    )

    @field_validator("structure_content")
    @classmethod
    def _reject_path_shaped_content(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("structure_content must not be empty")
        if "\n" not in stripped and len(stripped) < 256:
            # A bare one-line, no-newline value is far more likely to be
            # a filesystem path a caller passed by mistake than real CIF/
            # POSCAR text, which always spans multiple lines.
            raise ValueError(
                "structure_content must be file text (e.g. a CIF), not a "
                "path -- remote transports do not accept local paths"
            )
        return value

    @field_validator("structure_name")
    @classmethod
    def _one_file_name(cls, value: str) -> str:
        return require_filename(value, field="structure_name")


class ComputeRequestDocument(InlineStructureDocument):
    """A structure plus the calculation to prepare for it."""

    code: CodeName = Field(
        "quantum_espresso", description="The simulation code to write input for."
    )
    task: CalcTask = Field(
        "scf_single_point",
        description="The calculation task. capabilities lists each task's steps.",
    )
    hpc: str | None = Field(
        None,
        description=(
            "An HPC profile id from capabilities' hpc_profiles. May be omitted "
            "when exactly one profile is installed."
        ),
    )
    overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Settings to pin, as {key: value} with JSON values. Keys are "
            "capabilities' settings[].key and the overridable facts[].key. A "
            "key applies to every step of the task that reads it; "
            "'steps.<step>.<key>' sets one step only; a key that no step of "
            "the task reads is refused. null leaves a key to the automatic "
            "choice. Every value here is recorded as the user's own choice "
            "(source 'human')."
        ),
    )


class RunRequestDocument(ComputeRequestDocument):
    """A run request over HTTP, which can return the bundle as a ZIP."""

    respond_with: Literal["json", "archive"] = Field(
        "json",
        description=(
            "'json' returns the file list, decisions and warnings; 'archive' "
            "returns the whole input bundle as a ZIP file."
        ),
    )


class MagneticOrderingsRequestDocument(InlineStructureDocument):
    """A structure whose candidate magnetic orderings to list."""


class MagneticOrderingsToolDocument(MagneticOrderingsRequestDocument):
    """A structure whose candidate magnetic orderings to list, and optionally
    rank."""

    rank_with_mmace: bool = Field(
        False,
        description=(
            "Also relax every candidate on the mMACE potential and mark the "
            "lowest energy per atom as recommended. Takes seconds for a small "
            "cell and minutes for a large one."
        ),
    )


class CapabilitiesRequestDocument(BaseModel):
    """Optionally narrows capabilities to one task."""

    model_config = _STRICT

    task: CalcTask | None = Field(
        None,
        description=(
            "Return only this task, the codes that offer it and the settings "
            "it reads. Omit for every task."
        ),
    )


class ExplainResponse(TypedDict):
    """Every decision for the task, the warnings to relay, the findings run
    would refuse on (empty when run would generate), and each record's
    place."""

    records: dict[str, ResolvedField[Any]]
    warnings: list[RecordWarning]
    blocking: list[FindingDocument]
    layout: RecordLayout


class RunResponse(TypedDict):
    """The generated files' paths within the bundle, with the same records,
    warnings and layout as explain."""

    files: list[str]
    records: dict[str, ResolvedField[Any]]
    warnings: list[RecordWarning]
    layout: RecordLayout


CapabilitiesResponse = Capabilities
InspectResponse = StructureInspection
MagneticOrderingsResponse = MagneticOrderingsDocument
RankingResponse = RankingStatus


class ErrorBody(BaseModel):
    """What failed."""

    kind: str = Field(
        description=(
            "A stable name for the failure, e.g. invalid_request, "
            "invalid_setting, advice_incomplete, asset_not_installed."
        )
    )
    message: str = Field(description="What failed, written to show the user.")
    details: dict[str, Any] | None = Field(
        None,
        description=(
            "Data for this kind: validation_errors for invalid_request, "
            "findings (explain's blocking findings) for advice_incomplete."
        ),
    )


class ErrorResponse(BaseModel):
    """The body of every failure, over HTTP and in MCP tool errors."""

    error: ErrorBody


_INVALID_REQUEST_MESSAGE = "The request does not match the transport contract."


def invalid_request(
    errors: Iterable[Mapping[str, Any]], *, skip: int = 0
) -> dict[str, Any]:
    """The ``invalid_request`` failure for pydantic validation ``errors``.

    ``skip`` drops the leading location parts the transport adds (FastAPI's
    ``body``, the MCP wrapper's ``document``). Only each error's own message
    is kept: the raw ``ctx`` is not JSON-safe and the rejected value may be a
    whole structure file."""
    return {
        "error": {
            "kind": "invalid_request",
            "message": _INVALID_REQUEST_MESSAGE,
            "details": {
                "validation_errors": [
                    {
                        "path": ".".join(str(part) for part in item["loc"][skip:]),
                        "type": item["type"],
                        "message": item["msg"],
                    }
                    for item in errors
                ]
            },
        }
    }
