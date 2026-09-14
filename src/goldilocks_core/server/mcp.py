"""MCP stdio transport over the v2 ``service``/``capabilities``/
``set_overrides`` stack (v2 epic 8, #8).

Every tool calls ``server/_handlers.py``'s free functions -- the exact
same path ``server/http.py`` calls -- so validation/dispatch can never
drift between the two transports (this epic's own "one shared
request-validation path" scope item).

Four tools, not v1's three: ``capabilities``, ``inspect_structure``,
``explain`` (new: the tri-state "diagnosis always available" half of
the promise, with no v1 precedent at the transport layer), and ``run``
(replaces v1's ``compute``). ``run`` never returns raw bundle bytes --
only the file list, every decision, and ``warnings`` -- matching MCP's
own JSON/text idiom; fetching actual file contents has no MCP-shaped
answer yet (out of scope, tracked with web/agent integration, not this
epic's ``/run archive`` HTTP route).

Fixes issue #8's own evidence #3: v1 hardcoded ``version="0.1.0"``,
disagreeing with ``capabilities.py``'s real
``importlib.metadata.version("goldilocks-core")``. Now the same call.
"""

from __future__ import annotations

import asyncio
from importlib.metadata import version as package_version
from typing import Any

from goldilocks_core.capabilities import capabilities as build_capabilities
from goldilocks_core.failures import ExpectedFailure
from goldilocks_core.server import _handlers
from goldilocks_core.server.documents import (
    ComputeRequestDocument,
    InlineStructureDocument,
)

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError as error:
    raise ImportError(
        "The MCP transport requires goldilocks-core[mcp]. "
        "Install it with `uv sync --extra mcp`."
    ) from error

__all__ = ["create_server", "serve"]


class _StrictMCPServer(MCPServer):
    """``additionalProperties: false`` on every tool's schema, and every
    call re-checked against that same schema -- reused unchanged from
    v1 (issue #8's own "v1 code leaned on as-is" section): the one hard
    gate at the LLM/core boundary."""

    async def list_tools(self) -> list[Any]:
        tools = await super().list_tools()
        for tool in tools:
            tool.input_schema["additionalProperties"] = False
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> Any:
        tools = await self.list_tools()
        tool = next((candidate for candidate in tools if candidate.name == name), None)
        if tool is not None:
            allowed = set(tool.input_schema.get("properties", {}))
            unknown = sorted(set(arguments) - allowed)
            if unknown:
                raise ToolError(f"Unknown {name} arguments: {', '.join(unknown)}")
        return await super().call_tool(name, arguments, context)


def create_server(*, name: str = "goldilocks-core") -> MCPServer:
    server = _StrictMCPServer(
        name=name,
        version=package_version("goldilocks-core"),
        instructions=(
            "Inspect structures and generate Goldilocks Core DFT inputs. "
            "structure_content must be file text (e.g. CIF), never a path."
        ),
    )

    @server.tool(
        description="Describe available codes, tasks, settings, facts, and assets."
    )
    async def capabilities() -> dict[str, Any]:
        return await asyncio.to_thread(build_capabilities)

    @server.tool(description="Normalize and inspect an inline structure.")
    async def inspect_structure(document: InlineStructureDocument) -> dict[str, Any]:
        return await _call(_handlers.inspect, document)

    @server.tool(
        description=(
            "Run analysis and advisors only, without generating any files. "
            "Returns every decision (with its source) and a warnings array "
            "the caller must relay verbatim. task='dos' explains all three "
            "of the density-of-states task's steps (scf, nscf, dos.x); "
            "its nscf-step decisions are returned under nscf_-prefixed keys."
        )
    )
    async def explain(document: ComputeRequestDocument) -> dict[str, Any]:
        return await _call(_handlers.explain, document)

    @server.tool(
        description=(
            "Generate a runnable input. Returns the published file list, "
            "every decision, and a warnings array the caller must relay "
            "verbatim -- fails if any required field is unavailable or "
            "blocked. task='dos' generates all three steps (scf, nscf, "
            "dos.x) sharing one submission script, not just a single scf."
        )
    )
    async def run(document: ComputeRequestDocument) -> dict[str, Any]:
        summary, _bundle_input = await _call(_handlers.run, document)
        return summary

    return server


async def _call(handler: Any, document: Any) -> Any:
    try:
        return await asyncio.to_thread(handler, document)
    except ExpectedFailure as error:
        raise ToolError(str(error)) from error


def serve() -> None:
    create_server().run("stdio")
