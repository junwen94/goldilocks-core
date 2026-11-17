"""Tests for goldilocks_core.server.mcp.

Deliberately thin on business-logic scenarios (test_handlers.py already
covers those) -- this file is about what's MCP-specific: the flat tool
schemas and their output schemas and annotations, the real package version,
progress, and the structured errors HTTP also sends (#299).

No ``pytest-asyncio`` in this project's dependencies -- every test
below runs its coroutine through a plain ``asyncio.run()`` instead of a
native ``async def`` test function.
"""

from __future__ import annotations

import asyncio
import json
import math
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

from mcp import Client
from support import INTERNAL_TEXT, strict_json

from goldilocks_core.bundle import InvalidPublishRoot
from goldilocks_core.server import _handlers
from goldilocks_core.server.documents import (
    CapabilitiesRequestDocument,
    ComputeRequestDocument,
    InlineStructureDocument,
    MagneticOrderingsToolDocument,
    PublishRequestDocument,
)
from goldilocks_core.server.mcp import create_server

_SILICON_GAMMA_INTERVAL = [round(2 * math.pi / 5.431, 8), None]
"""Rung 1 of the bundled Si.cif's ladder (cubic, a = 5.431 A): every
k-distance above |b| = 2 pi / a, with no upper bound."""


def _call(tool: str, arguments: dict[str, Any]) -> Any:
    return asyncio.run(create_server().call_tool(tool, arguments))


def _validation_paths(result: Any) -> list[str]:
    error = result.structured_content["error"]
    assert error["kind"] == "invalid_request"
    return [item["path"] for item in error["details"]["validation_errors"]]


def _gamma_only_call(server: Any, tool: str, silicon_cif: str) -> Any:
    return asyncio.run(
        server.call_tool(
            tool,
            {
                "structure_content": silicon_cif,
                "hpc": "scarf",
                "overrides": {"k_grid": [1, 1, 1]},
            },
        )
    )


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
    def test_exposes_exactly_five_tools(self) -> None:
        server = create_server()

        tools = asyncio.run(server.list_tools())

        assert {tool.name for tool in tools} == {
            "capabilities",
            "inspect_structure",
            "magnetic_orderings",
            "explain",
            "run",
        }

    def test_every_tool_forbids_additional_properties(self) -> None:
        server = create_server()

        tools = asyncio.run(server.list_tools())

        for tool in tools:
            assert tool.input_schema["additionalProperties"] is False, tool.name


_TOOL_DOCUMENTS = {
    "capabilities": CapabilitiesRequestDocument,
    "inspect_structure": InlineStructureDocument,
    "magnetic_orderings": MagneticOrderingsToolDocument,
    "explain": ComputeRequestDocument,
    "run": ComputeRequestDocument,
}
_TOOL_OUTPUT_KEYS = {
    "capabilities": {"codes", "tasks", "settings", "facts"},
    "inspect_structure": {"source", "structure", "canonical_cif"},
    "magnetic_orderings": {"ranked", "candidates", "warnings"},
    "explain": {"records", "warnings", "blocking", "layout"},
    "run": {"files", "records", "warnings", "layout"},
}


class TestToolContract:
    """What a client's model reads before calling (#299)."""

    def test_each_tool_takes_its_document_fields_as_arguments(self) -> None:
        tools = asyncio.run(create_server().list_tools())

        assert {tool.name: tool.input_schema for tool in tools} == {
            name: document.model_json_schema()
            for name, document in _TOOL_DOCUMENTS.items()
        }

    def test_each_tool_publishes_the_shape_it_returns(self) -> None:
        tools = asyncio.run(create_server().list_tools())

        for tool in tools:
            properties = set(tool.output_schema["properties"])
            assert _TOOL_OUTPUT_KEYS[tool.name] <= properties, tool.name

    def test_every_tool_is_titled_read_only_and_closed_world(self) -> None:
        tools = asyncio.run(create_server().list_tools())

        for tool in tools:
            assert tool.title, tool.name
            assert tool.annotations.read_only_hint is True, tool.name
            assert tool.annotations.open_world_hint is False, tool.name

    def test_schema_text_reads_as_user_text(self) -> None:
        server = create_server()
        tools = asyncio.run(server.list_tools())
        texts = [server.instructions or ""]
        for tool in tools:
            texts.append(tool.description or "")
            texts.extend(_descriptions(tool.input_schema))
            texts.extend(_descriptions(tool.output_schema))

        assert [text for text in texts if INTERNAL_TEXT.search(text)] == []

    def test_capabilities_narrows_to_one_task(self) -> None:
        result = _call("capabilities", {"task": "dos"})

        assert result.is_error is False
        assert [task["id"] for task in result.structured_content["tasks"]] == ["dos"]
        assert all("dos" in s["tasks"] for s in result.structured_content["settings"])

    def test_a_ranking_reports_progress_to_the_client(self, monkeypatch) -> None:
        def ranked(document: Any, on_progress: Any) -> dict[str, Any]:
            on_progress(0, 2)
            on_progress(2, 2)
            return {"ranked": True, "candidates": [], "warnings": []}

        monkeypatch.setattr(_handlers, "magnetic_orderings_tool", ranked)
        seen: list[tuple[float, float | None]] = []

        async def progress(done: float, total: float | None, message: Any) -> None:
            seen.append((done, total))

        async def call() -> Any:
            async with Client(create_server()) as client:
                return await client.call_tool(
                    "magnetic_orderings",
                    {
                        "structure_content": "line one\nline two",
                        "rank_with_mmace": True,
                    },
                    progress_callback=progress,
                )

        result = asyncio.run(call())

        assert result.is_error is False
        assert seen == [(0, 2), (2, 2)]


def _descriptions(schema: Any) -> list[str]:
    if isinstance(schema, dict):
        own = (
            [schema["description"]]
            if isinstance(schema.get("description"), str)
            else []
        )
        return own + [
            text for value in schema.values() for text in _descriptions(value)
        ]
    if isinstance(schema, list):
        return [text for value in schema for text in _descriptions(value)]
    return []


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
            server.call_tool("inspect_structure", {"structure_content": silicon_cif})
        )

        assert result.is_error is False
        assert result.structured_content["structure"]["reduced_formula"] == "Si"

    def test_magnetic_orderings_tool_lists_the_fm_candidate_unranked(
        self, silicon_cif: str
    ) -> None:
        server = create_server()

        result = asyncio.run(
            server.call_tool("magnetic_orderings", {"structure_content": silicon_cif})
        )

        assert result.is_error is False
        assert result.structured_content["ranked"] is False
        assert result.structured_content["candidates"][0]["label"] == "fm"

    def test_run_tool_end_to_end(self, real_assets, silicon_cif: str) -> None:
        server = create_server()

        result = asyncio.run(
            server.call_tool("run", {"structure_content": silicon_cif, "hpc": "scarf"})
        )

        assert result.is_error is False
        assert "scf.in" in result.structured_content["files"]

    def test_run_tool_dos_task_generates_all_three_steps(
        self, real_assets, silicon_cif: str
    ) -> None:
        """v2 epic 9 (#9, #28): the MCP `run` tool used to silently
        ignore `task`, always running the scf-only pipeline -- this
        confirms it actually routes to advise_dos/generate_dos through
        the real MCP transport (the same shared _handlers.run HTTP
        calls, so this is really pinning that the document.task field
        reaches it, not re-testing the routing logic itself)."""
        server = create_server()

        result = asyncio.run(
            server.call_tool(
                "run",
                {
                    "structure_content": silicon_cif,
                    "hpc": "scarf",
                    "task": "dos",
                },
            )
        )

        assert result.is_error is False
        files = result.structured_content["files"]
        assert "scf.in" in files
        assert "nscf.in" in files
        assert "dos.in" in files

    def test_run_tool_band_task_generates_every_step(
        self, silicon_fixture_table, silicon_cif: str
    ) -> None:
        """#118: task='band' reaches the shared handler through the real MCP
        transport, which publishes the bands step's path as k_path and its
        other decisions under bands_ names."""
        server = create_server()

        result = asyncio.run(
            server.call_tool(
                "run",
                {
                    "structure_content": silicon_cif,
                    "hpc": "scarf",
                    "task": "band",
                },
            )
        )

        assert result.is_error is False
        files = result.structured_content["files"]
        for name in ("scf.in", "bands.in", "bands_x.in", "submit.sh"):
            assert name in files, name
        assert {"k_path", "bands_nbnd"} <= set(result.structured_content["records"])

    @pytest.mark.parametrize(
        ("task", "first_step"),
        [("phonon", "scf"), ("relax-phonon", "relax"), ("vc-relax-phonon", "vc-relax")],
    )
    def test_explain_and_run_tools_route_each_phonon_task(
        self, silicon_fixture_table, silicon_cif: str, task: str, first_step: str
    ) -> None:
        """#215: each phonon task reaches the shared handlers through the real
        MCP transport: explain finds nothing to refuse in Si set to a
        non-metal, which a phonon task needs on smearing without a model to
        decide it, and run generates the first pw.x step, ph.x, q2r.x and
        both matdyn.x inputs, with the decisions laid out under the ph, q2r,
        phonon_dos and dispersion steps as explain lays them out."""
        server = create_server()
        document = {
            "structure_content": silicon_cif,
            "hpc": "scarf",
            "task": task,
            "overrides": {"is_metal": "non_metal"},
        }

        explained = asyncio.run(server.call_tool("explain", document))
        result = asyncio.run(server.call_tool("run", document))

        assert (explained.is_error, result.is_error) == (False, False)
        assert explained.structured_content["blocking"] == []
        files = result.structured_content["files"]
        for name in (
            f"{first_step}.in",
            "ph.in",
            "q2r.in",
            "phonon_dos.in",
            "dispersion.in",
        ):
            assert name in files, name
        layout = result.structured_content["layout"]
        assert layout == explained.structured_content["layout"]
        assert [step["name"] for step in layout["steps"]] == [
            first_step,
            "ph",
            "q2r",
            "phonon_dos",
            "dispersion",
        ]

    def test_run_tool_relax_task_generates_relax_in(
        self, real_assets, silicon_cif: str
    ) -> None:
        """v2 epic 10 (#10): confirms task='relax' actually routes
        through the real MCP transport, same shape as the dos check
        above."""
        server = create_server()

        result = asyncio.run(
            server.call_tool(
                "run",
                {
                    "structure_content": silicon_cif,
                    "hpc": "scarf",
                    "task": "relax",
                },
            )
        )

        assert result.is_error is False
        assert "relax.in" in result.structured_content["files"]

    def test_explain_tool_answers_a_gamma_only_mesh_with_strict_json(
        self, silicon_cif: str
    ) -> None:
        """The tool's text content carried Infinity for rung 1's open
        k-distance bound, while its structured content read null (#137)."""
        result = _gamma_only_call(create_server(), "explain", silicon_cif)

        assert result.is_error is False
        records = strict_json(result.content[0].text)["records"]
        interval = records["k_sampling"]["value"]["k_distance_interval"]
        assert interval == _SILICON_GAMMA_INTERVAL
        structured = result.structured_content["records"]["k_sampling"]["value"]
        assert structured["k_distance_interval"] == _SILICON_GAMMA_INTERVAL

    def test_run_tool_answers_a_gamma_only_mesh_with_strict_json(
        self, silicon_fixture_table, silicon_cif: str
    ) -> None:
        result = _gamma_only_call(create_server(), "run", silicon_cif)

        assert result.is_error is False
        records = strict_json(result.content[0].text)["records"]
        interval = records["k_sampling"]["value"]["k_distance_interval"]
        assert interval == _SILICON_GAMMA_INTERVAL

    def test_explain_tool_returns_the_reasons_the_run_tool_refuses_with(
        self, silicon_cif: str
    ) -> None:
        """#165: no real assets, so the isolated asset root's uninstalled
        default table blocks both tools."""
        server = create_server()
        arguments = {"structure_content": silicon_cif, "hpc": "scarf"}

        explained = asyncio.run(server.call_tool("explain", arguments))

        assert explained.is_error is False
        blocking = explained.structured_content["blocking"]
        assert len(blocking) == 1
        assert blocking[0]["code"] == "advice.blocked"
        assert "is not installed" in blocking[0]["message"]
        refused = asyncio.run(server.call_tool("run", arguments))
        assert refused.is_error is True
        error = refused.structured_content["error"]
        assert error["kind"] == "advice_incomplete"
        assert error["details"]["findings"] == blocking

    @pytest.mark.parametrize(
        ("extra", "path"),
        [({"bogus": 1}, "bogus"), ({"fetch_missing": True}, "fetch_missing")],
    )
    def test_an_unknown_argument_is_an_invalid_request(
        self, silicon_cif: str, extra: dict[str, Any], path: str
    ) -> None:
        result = _call("run", {"structure_content": silicon_cif, **extra})

        assert result.is_error is True
        assert _validation_paths(result) == [path]

    def test_the_old_document_wrapper_is_refused(self, silicon_cif: str) -> None:
        """Arguments are the document's own fields (#299)."""
        result = _call("explain", {"document": {"structure_content": silicon_cif}})

        assert result.is_error is True
        assert _validation_paths(result) == ["structure_content", "document"]

    def test_unknown_setting_is_a_structured_error(
        self, real_assets, silicon_cif: str
    ) -> None:
        result = _call(
            "run",
            {
                "structure_content": silicon_cif,
                "hpc": "scarf",
                "overrides": {"ecutwf_ry": 30},
            },
        )

        assert result.is_error is True
        error = result.structured_content["error"]
        assert error["kind"] == "invalid_setting"
        assert "did you mean: ecutwfc_ry" in error["message"]

    def test_path_shaped_structure_content_is_an_invalid_request(self) -> None:
        result = _call("inspect_structure", {"structure_content": "/etc/passwd"})

        assert result.is_error is True
        assert _validation_paths(result) == ["structure_content"]
        assert "not a path" in json.dumps(result.structured_content)

    def test_an_error_matches_the_http_body_for_the_same_request(
        self, silicon_cif: str
    ) -> None:
        """One error contract on both transports (#299)."""
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from goldilocks_core.server.http import create_app

        request = {"structure_content": silicon_cif, "hpc": "no-such-machine"}
        result = _call("explain", request)
        response = TestClient(create_app()).post("/explain", json=request)

        assert result.is_error is True
        assert result.structured_content == response.json()
        assert json.loads(result.content[0].text) == response.json()


class TestPublishRoot:
    """``goldilocks serve mcp --publish-root DIR`` (#299): run publishes the
    bundle under a name the caller gives, never a path."""

    def _publish(self, root: Path, silicon_cif: str, name: str) -> Any:
        server = create_server(publish_root=root)
        arguments = {"structure_content": silicon_cif, "bundle_name": name}
        return asyncio.run(server.call_tool("run", arguments))

    def test_run_publishes_the_whole_bundle_once_under_its_name(
        self, silicon_fixture_table, silicon_cif: str, tmp_path: Path
    ) -> None:
        published = self._publish(tmp_path, silicon_cif, "si-scf")
        again = self._publish(tmp_path, silicon_cif, "si-scf")

        assert published.is_error is False
        target = tmp_path.resolve() / "si-scf"
        assert published.structured_content["path"] == str(target)
        assert (target / ".complete").is_file()
        assert (target / "scf.in").is_file()
        assert again.structured_content["error"]["kind"] == "bundle_exists"

    @pytest.mark.parametrize("name", ["../escape", "a/b", ".hidden", ""])
    def test_a_bundle_name_is_one_new_name_not_a_path(
        self, silicon_cif: str, tmp_path: Path, name: str
    ) -> None:
        result = self._publish(tmp_path, silicon_cif, name)

        assert _validation_paths(result) == ["bundle_name"]
        assert list(tmp_path.iterdir()) == []

    def test_only_a_publishing_run_is_marked_as_writing(self, tmp_path: Path) -> None:
        def run_tool(server: Any) -> Any:
            tools = asyncio.run(server.list_tools())
            return next(tool for tool in tools if tool.name == "run")

        summary = run_tool(create_server())
        publishing_server = create_server(publish_root=tmp_path)
        publishing = run_tool(publishing_server)

        assert summary.annotations.read_only_hint is True
        assert "bundle_name" not in summary.input_schema["properties"]
        assert publishing.annotations.read_only_hint is False
        assert publishing.annotations.destructive_hint is False
        assert publishing.input_schema == PublishRequestDocument.model_json_schema()
        assert "bundle_name" in (publishing_server.instructions or "")

    def test_a_missing_publish_root_stops_the_server_starting(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(InvalidPublishRoot):
            create_server(publish_root=tmp_path / "missing")


class TestPerStepOverrides:
    """#154 over MCP: `steps.nscf.k_grid` raised "unknown setting", dos.x
    keys were unknown, and press on an scf was dropped silently."""

    def _call(
        self, tool: str, silicon_cif: str, task: str, overrides: dict[str, Any]
    ) -> Any:
        document = {
            "structure_content": silicon_cif,
            "hpc": "scarf",
            "task": task,
            "overrides": overrides,
        }
        return asyncio.run(create_server().call_tool(tool, document))

    def test_steps_nscf_k_grid_changes_only_the_nscf_mesh(
        self, silicon_cif: str
    ) -> None:
        default = self._call("explain", silicon_cif, "dos", {})
        result = self._call(
            "explain", silicon_cif, "dos", {"steps.nscf.k_grid": [30, 30, 30]}
        )

        assert result.is_error is False
        records = result.structured_content["records"]
        assert records["nscf_k_sampling"]["value"]["mesh"] == [30, 30, 30]
        assert (
            records["k_sampling"] == default.structured_content["records"]["k_sampling"]
        )

    def test_a_shift_keeps_the_protocol_nscf_mesh(self, silicon_cif: str) -> None:
        default = self._call("explain", silicon_cif, "dos", {})
        result = self._call("explain", silicon_cif, "dos", {"shift": [1, 1, 1]})

        nscf = result.structured_content["records"]["nscf_k_sampling"]["value"]
        protocol = default.structured_content["records"]["nscf_k_sampling"]["value"]
        assert (nscf["k_distance"], nscf["mesh"]) == (0.1, protocol["mesh"])

    def test_a_dos_x_setting_reaches_the_dos_record(
        self, silicon_fixture_table, silicon_cif: str
    ) -> None:
        result = self._call("run", silicon_cif, "dos", {"delta_e": 0.05})

        assert result.is_error is False
        dos = result.structured_content["records"]["dos"]
        assert (dos["value"]["delta_e"], dos["source"]) == (0.05, "human")

    @pytest.mark.parametrize("tool", ["explain", "run"])
    def test_press_on_an_scf_raises_tool_error(
        self, silicon_cif: str, tool: str
    ) -> None:
        result = self._call(tool, silicon_cif, "scf_single_point", {"press": 10})

        assert result.is_error is True
        assert result.structured_content["error"]["message"].endswith(
            "'press' does not apply to task 'scf_single_point'; "
            "it applies to tasks vc-relax, vc-relax-phonon"
        )
