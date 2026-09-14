"""Tests for goldilocks_core.server.mcp (v2 epic 8, #8).

Deliberately thin on business-logic scenarios (test_handlers.py already
covers those) -- this file is about what's MCP-specific: the strict
tool schema, the real package version (issue #8's own evidence #3),
and ToolError propagation.

No ``pytest-asyncio`` in this project's dependencies -- every test
below runs its coroutine through a plain ``asyncio.run()`` instead of a
native ``async def`` test function.
"""

from __future__ import annotations

import asyncio
from importlib.metadata import version as package_version

import pytest

pytest.importorskip("mcp")

from mcp.server.mcpserver.exceptions import ToolError

from goldilocks_core.server.mcp import create_server


class TestServerIdentity:
    def test_version_matches_the_installed_package_not_a_hardcoded_string(self) -> None:
        """Issue #8's own evidence #3: v1 hardcoded version="0.1.0",
        disagreeing with capabilities.py's real
        importlib.metadata.version call. (This package's own current
        version happens to also read "0.1.0" -- the point is that this
        now comes from the same live call capabilities.py uses, not a
        second, independently-hardcoded literal that could drift.)"""
        from goldilocks_core.capabilities import capabilities

        server = create_server()

        assert server.version == package_version("goldilocks-core")
        assert server.version == capabilities()["core_version"]


class TestToolCatalogue:
    def test_exposes_exactly_four_tools(self) -> None:
        server = create_server()

        tools = asyncio.run(server.list_tools())

        assert {tool.name for tool in tools} == {
            "capabilities",
            "inspect_structure",
            "explain",
            "run",
        }

    def test_every_tool_forbids_additional_properties(self) -> None:
        server = create_server()

        tools = asyncio.run(server.list_tools())

        for tool in tools:
            assert tool.input_schema["additionalProperties"] is False, tool.name


class TestToolCalls:
    def test_capabilities_tool_returns_the_real_contract(self) -> None:
        from goldilocks_core.capabilities import capabilities

        server = create_server()

        result = asyncio.run(server.call_tool("capabilities", {}))

        assert result.is_error is False
        assert result.structured_content == capabilities()

    def test_inspect_structure_tool(self, silicon_cif: str) -> None:
        server = create_server()

        result = asyncio.run(
            server.call_tool(
                "inspect_structure", {"document": {"structure_content": silicon_cif}}
            )
        )

        assert result.is_error is False
        assert result.structured_content["structure"]["reduced_formula"] == "Si"

    def test_run_tool_end_to_end(self, real_assets, silicon_cif: str) -> None:
        server = create_server()

        result = asyncio.run(
            server.call_tool(
                "run", {"document": {"structure_content": silicon_cif, "hpc": "scarf"}}
            )
        )

        assert result.is_error is False
        assert "scf.in" in result.structured_content["files"]

    def test_unknown_top_level_argument_raises_tool_error(
        self, silicon_cif: str
    ) -> None:
        server = create_server()

        with pytest.raises(ToolError, match="Unknown run arguments"):
            asyncio.run(
                server.call_tool(
                    "run", {"document": {"structure_content": silicon_cif}, "bogus": 1}
                )
            )

    def test_unknown_setting_raises_tool_error(
        self, real_assets, silicon_cif: str
    ) -> None:
        server = create_server()

        with pytest.raises(ToolError, match="did you mean: ecutwfc_ry"):
            asyncio.run(
                server.call_tool(
                    "run",
                    {
                        "document": {
                            "structure_content": silicon_cif,
                            "hpc": "scarf",
                            "overrides": {"ecutwf_ry": 30},
                        }
                    },
                )
            )

    def test_path_shaped_structure_content_raises_tool_error(self) -> None:
        server = create_server()

        with pytest.raises(ToolError, match="not a path"):
            asyncio.run(
                server.call_tool(
                    "inspect_structure",
                    {"document": {"structure_content": "/etc/passwd"}},
                )
            )
