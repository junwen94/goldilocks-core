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
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from goldilocks_core.inputs.structure import require_filename
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
