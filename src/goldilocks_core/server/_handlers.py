"""Transport-agnostic request handling shared by HTTP and MCP (v2 epic
8, #8) -- one implementation of "structure in, advice/bundle out",
called identically by both transports, directly answering this epic's
own scope item ("one shared request-validation path... invoked
identically by CLI, HTTP, and MCP before dispatch to Service"). The
CLI's own equivalent (``cli/_run.py``/``cli/_explain.py``) can't share
this exact code since it resolves structure from a local path, not
inline content -- but everything past structure resolution is the same
shape here as there.
"""

from __future__ import annotations

from typing import Any

from goldilocks_core.assets.store import AssetStore
from goldilocks_core.bundle import BundleInput, bundle_files
from goldilocks_core.inputs.hpc import resolve_hpc_profile
from goldilocks_core.inputs.structure import InlineStructureSource, normalize_structure
from goldilocks_core.resolution import FieldState, ResolvedField
from goldilocks_core.server.documents import (
    ComputeRequestDocument,
    InlineStructureDocument,
)
from goldilocks_core.server.readiness import AssetReadiness
from goldilocks_core.service import (
    advise,
    check,
    generate,
    render_submission,
    to_bundle_input,
)
from goldilocks_core.set_overrides import build_overrides
from goldilocks_core.steps import default_shared_context

_HPC_FIELD_NAME = "hpc"


def build_readiness() -> AssetReadiness:
    """The one place HTTP/MCP construct their readiness checker, so
    neither transport needs to import ``AssetStore``/``AssetReadiness``
    directly -- keeps both transports' own import surface inside their
    tight, transport-specific complexity ceilings
    (``scripts/check_complexity.py``)."""
    return AssetReadiness(AssetStore())


def _structure_source(document: InlineStructureDocument) -> InlineStructureSource:
    return InlineStructureSource(
        name=document.structure_name,
        content=document.structure_content,
        format=document.structure_format,
    )


def _records_document(records: dict[str, FieldState[object]]) -> dict[str, Any]:
    """Plain ``dict``s, not ``ResolvedField`` instances: some callers
    (``server/http.py``'s ``/run`` route, wrapping this in ``JSONResponse``
    directly for its "archive" branch) bypass FastAPI's own pydantic
    -aware response encoder, which is otherwise what turns a returned
    model into JSON. Dumping here once makes every caller's output
    JSON-safe regardless of how it gets serialized downstream."""
    return {
        name: ResolvedField.from_state(state).model_dump()
        for name, state in records.items()
    }


def inspect(document: InlineStructureDocument) -> dict[str, Any]:
    return normalize_structure(_structure_source(document)).inspection


def explain(document: ComputeRequestDocument) -> dict[str, Any]:
    structure = normalize_structure(_structure_source(document)).structure
    hpc = resolve_hpc_profile(document.hpc, field=_HPC_FIELD_NAME)
    overrides = build_overrides(document.overrides)
    advice = advise(
        structure,
        code=document.code,
        hpc=hpc,
        overrides=overrides,
        fetch_missing=document.fetch_missing,
    )
    return {
        "records": _records_document(advice.records()),
        "warnings": advice.warnings(),
    }


def run(document: ComputeRequestDocument) -> tuple[dict[str, Any], BundleInput]:
    """Returns both the JSON-ready summary and the ``BundleInput`` it
    was built from -- HTTP's archive response mode reuses the latter to
    build the zip without recomputing the pipeline; MCP only ever needs
    the former."""
    structure = normalize_structure(_structure_source(document)).structure
    hpc = resolve_hpc_profile(document.hpc, field=_HPC_FIELD_NAME)
    overrides = build_overrides(document.overrides)
    advice = advise(
        structure,
        code=document.code,
        hpc=hpc,
        overrides=overrides,
        fetch_missing=document.fetch_missing,
    )
    report = check(advice)
    steps = generate(advice, report)  # raises AdviceIncomplete if report is not ok
    ctx = default_shared_context()
    script = render_submission(advice, hpc, document.code, ctx, steps)
    bundle_input = to_bundle_input(advice, steps, script, ctx)
    summary = {
        "files": [file["path"] for file in bundle_files(bundle_input)],
        "records": _records_document(advice.records()),
        "warnings": advice.warnings(),
    }
    return summary, bundle_input
