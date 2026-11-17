"""MCP stdio transport over ``server/_handlers.py``, the same functions the
HTTP transport calls, so validation and dispatch cannot drift between them.

- **Flat arguments.** Each tool's input schema is its request document's own
  schema (``server/documents.py``), so a call is ``{"structure_content": ...,
  "task": ...}``. The arguments are wrapped into the tool function's single
  ``document`` parameter before the SDK validates them, so the pydantic
  documents stay the one contract; ``extra="forbid"`` refuses unknown keys.
- **Results.** A tool returns its document as structured content checked
  against the response type (the tool's ``outputSchema``), plus the same JSON
  as compact text.
- **Errors.** A refused request returns ``isError`` with ``{"error": {kind,
  message, details}}``, the body HTTP sends; an argument that does not match
  the document is ``invalid_request``.
- ``run`` returns the file list, decisions and warnings, never file contents.
  With a publish root, set by whoever starts the server, ``run`` also takes
  a ``bundle_name`` and publishes the whole bundle as a new directory under
  that root; a caller names the bundle, never a path
  (goldilocks-agent-design.md S9.1).
- Transports: stdio, or streamable HTTP at ``/mcp`` on a host and port.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from functools import partial
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
from pydantic import BaseModel, ValidationError

from goldilocks_core.bundle import resolve_publish_root
from goldilocks_core.capabilities import capabilities as build_capabilities
from goldilocks_core.failures import ExpectedFailure
from goldilocks_core.server import _handlers
from goldilocks_core.server.documents import (
    CapabilitiesRequestDocument,
    CapabilitiesResponse,
    ComputeRequestDocument,
    ExplainResponse,
    InlineStructureDocument,
    InspectResponse,
    MagneticOrderingsResponse,
    MagneticOrderingsToolDocument,
    PublishRequestDocument,
    RunResponse,
    invalid_request,
)

try:
    from mcp.server.mcpserver import Context, MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp_types import CallToolResult, TextContent, ToolAnnotations
except ImportError as error:
    raise ImportError(
        'The MCP transport requires the mcp extra: install "goldilocks-core[mcp]".'
    ) from error

__all__ = ["create_server", "serve"]

_INSTRUCTIONS = """\
goldilocks-core recommends DFT settings for a crystal structure and writes \
Quantum ESPRESSO input files. It does not run DFT.
Call capabilities (with the task, for the settings it reads), then \
inspect_structure, then explain, then run.
Send a structure as file text in structure_content, never as a path.
Relay every entry of a result's warnings array to the user as it is.
When explain lists blocking findings, show them to the user. Each remedy is \
an overrides patch that resolves its finding (null clears a key): apply the \
one the user chooses, then explain again.
Every value in overrides is recorded as the user's own choice: send only \
values the user gave.
A failure returns {"error": {"kind", "message", "details"}}: act on kind.\
"""

_PUBLISHING_INSTRUCTIONS = """
This server publishes run's whole bundle when you pass a new bundle_name; \
the result's path is on the server's machine.\
"""

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
_PUBLISHING = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)
_RUN_DESCRIPTION = (
    "Generate the task's input files and submission script. Returns the file "
    "list with the same records, warnings and layout as explain. If run's "
    "checks refuse the request it fails with kind advice_incomplete and "
    "details.findings, explain's blocking findings."
)

type Transport = Literal["stdio", "streamable-http"]


class _DocumentServer(MCPServer):
    """An MCP server whose every tool takes one request document, published
    with the document's fields as the tool's own arguments."""

    def __init__(self, **settings: Any) -> None:
        super().__init__(**settings)
        self._documents: dict[str, type[BaseModel]] = {}

    def document_tool(
        self,
        document: type[BaseModel],
        *,
        name: str,
        title: str,
        description: str,
        annotations: ToolAnnotations = _READ_ONLY,
    ) -> Callable[[Any], Any]:
        """Registers the decorated function, whose one argument is
        ``document: <document>``, as the tool ``name``."""
        self._documents[name] = document
        return self.tool(
            name=name, title=title, description=description, annotations=annotations
        )

    async def list_tools(self) -> list[Any]:
        tools = await super().list_tools()
        for tool in tools:
            tool.input_schema = self._documents[tool.name].model_json_schema()
        return tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Any | None = None
    ) -> Any:
        try:
            return await super().call_tool(name, {"document": arguments}, context)
        except ToolError as error:
            if isinstance(error.__cause__, ValidationError):
                return _error_result(invalid_request(error.__cause__.errors(), skip=1))
            raise


def create_server(
    *, name: str = "goldilocks-core", publish_root: str | Path | None = None
) -> MCPServer:
    """The MCP server. ``publish_root``, an existing directory, lets ``run``
    publish bundles under it; it is resolved once, here."""
    root = None if publish_root is None else resolve_publish_root(publish_root)
    server = _DocumentServer(
        name=name,
        version=package_version("goldilocks-core"),
        instructions=_INSTRUCTIONS + ("" if root is None else _PUBLISHING_INSTRUCTIONS),
    )

    @server.document_tool(
        CapabilitiesRequestDocument,
        name="capabilities",
        title="Capabilities",
        description=(
            "What this server can prepare: codes, tasks and their steps, "
            "settings (key, type, unit, default, allowed values), facts, "
            "pseudopotential tables, HPC profiles, ML models and the warnings "
            "catalogue. Pass task to get only the settings that task reads."
        ),
    )
    async def capabilities(
        document: CapabilitiesRequestDocument,
    ) -> Annotated[CallToolResult, CapabilitiesResponse]:
        return await _call(build_capabilities, document.task)

    @server.document_tool(
        InlineStructureDocument,
        name="inspect_structure",
        title="Inspect structure",
        description=(
            "Parse a structure and report its source, composition, lattice and "
            "sites, with a canonical CIF. explain, run and magnetic_orderings "
            "refuse a structure with partial occupancies; order it first."
        ),
    )
    async def inspect_structure(
        document: InlineStructureDocument,
    ) -> Annotated[CallToolResult, InspectResponse]:
        return await _call(_handlers.inspect, document)

    @server.document_tool(
        MagneticOrderingsToolDocument,
        name="magnetic_orderings",
        title="List magnetic orderings",
        description=(
            "List candidate magnetic orderings: ferromagnetic, plus every "
            "antiferromagnetic candidate enumlib finds. rank_with_mmace also "
            "ranks them by mMACE energy per atom and marks the lowest as "
            "recommended, reporting progress; without mMACE on this server the "
            "listing comes back unranked, with a warning. Writes nothing: to "
            "prepare inputs for a candidate, send its overrides with the same "
            "structure to explain or run."
        ),
    )
    async def magnetic_orderings(
        document: MagneticOrderingsToolDocument, ctx: Context
    ) -> Annotated[CallToolResult, MagneticOrderingsResponse]:
        def report(done: int, total: int) -> None:
            anyio.from_thread.run(ctx.report_progress, done, total)

        return await _call(_handlers.magnetic_orderings_tool, document, report)

    @server.document_tool(
        ComputeRequestDocument,
        name="explain",
        title="Explain DFT settings",
        description=(
            "Decide every setting of a task for the structure, without writing "
            "files. Returns records (each decision: value, source, what it was "
            "derived from), warnings to relay as they are (each names its "
            "record), blocking (the findings run would refuse on, empty when "
            "run would generate; each has a message to show and remedies) and "
            "layout (each record's tier or step). Records of a later step "
            "carry its prefix, e.g. nscf_ for dos and bands_ for band; "
            "capabilities lists each task's steps."
        ),
    )
    async def explain(
        document: ComputeRequestDocument,
    ) -> Annotated[CallToolResult, ExplainResponse]:
        return await _call(_handlers.explain, document)

    if root is None:
        _register_summary_run(server)
    else:
        _register_publishing_run(server, root)
    return server


def _register_summary_run(server: _DocumentServer) -> None:
    @server.document_tool(
        ComputeRequestDocument,
        name="run",
        title="Generate DFT inputs",
        description=(
            _RUN_DESCRIPTION + " This server writes nothing and returns no "
            "file contents."
        ),
    )
    async def run(
        document: ComputeRequestDocument,
    ) -> Annotated[CallToolResult, RunResponse]:
        return await _call(_run_summary, document)


def _register_publishing_run(server: _DocumentServer, root: Path) -> None:
    @server.document_tool(
        PublishRequestDocument,
        name="run",
        title="Generate DFT inputs",
        description=(
            _RUN_DESCRIPTION + " With bundle_name, the whole bundle is also "
            "published as a new directory in this server's publish directory, "
            "and path gives its location on the server's machine; an existing "
            "name fails with kind bundle_exists. Returns no file contents."
        ),
        annotations=_PUBLISHING,
    )
    async def run(
        document: PublishRequestDocument,
    ) -> Annotated[CallToolResult, RunResponse]:
        return await _call(_handlers.publish_run, document, root)


def _run_summary(document: ComputeRequestDocument) -> RunResponse:
    summary, _bundle_input = _handlers.run(document)
    return summary


async def _call(handler: Callable[..., Any], *args: Any) -> CallToolResult:
    """``handler(*args)`` on a worker thread, as a tool result."""
    try:
        document = await anyio.to_thread.run_sync(partial(handler, *args))
    except ExpectedFailure as error:
        return _error_result({"error": error.public_error()})
    return _result(document)


def _result(document: Mapping[str, Any]) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=_compact_json(document))],
        structured_content=dict(document),
    )


def _error_result(document: Mapping[str, Any]) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=_compact_json(document))],
        structured_content=dict(document),
        is_error=True,
    )


def _compact_json(document: Mapping[str, Any]) -> str:
    return json.dumps(document, allow_nan=False, separators=(",", ":"))


def serve(
    *,
    transport: Transport = "stdio",
    host: str = "127.0.0.1",
    port: int = 8001,
    publish_root: str | Path | None = None,
) -> None:
    """Serve over stdio, or over streamable HTTP at ``http://host:port/mcp``."""
    server = create_server(publish_root=publish_root)
    if transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=host, port=port)
