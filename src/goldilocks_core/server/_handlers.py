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
from goldilocks_core.resolution import records_document
from goldilocks_core.server.documents import (
    ComputeRequestDocument,
    InlineStructureDocument,
    MagneticOrderingsRequestDocument,
)
from goldilocks_core.server.readiness import AssetReadiness
from goldilocks_core.service import (
    advise,
    advise_dos,
    advise_relax,
    check,
    check_dos,
    check_relax,
    generate,
    generate_dos,
    generate_relax,
    list_magnetic_orderings,
    render_submission,
    render_submission_dos,
    render_submission_relax,
    report_to_json,
    to_bundle_input,
    to_bundle_input_dos,
    to_bundle_input_relax,
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


def inspect(document: InlineStructureDocument) -> dict[str, Any]:
    return normalize_structure(_structure_source(document)).inspection


def magnetic_orderings(document: MagneticOrderingsRequestDocument) -> dict[str, Any]:
    structure = normalize_structure(_structure_source(document)).structure
    report = list_magnetic_orderings(
        structure, rank_with_mmace=document.rank_with_mmace
    )
    return report_to_json(report)


def explain(document: ComputeRequestDocument) -> dict[str, Any]:
    """Every decision plus the verdict ``run`` reaches before generating
    (#165): ``blocking`` holds the distinct reasons the task's check
    refuses generation for, in first-seen order, and is empty when ``run``
    would go on to generate."""
    structure = normalize_structure(_structure_source(document)).structure
    hpc = resolve_hpc_profile(document.hpc, field=_HPC_FIELD_NAME)
    overrides = build_overrides(document.overrides, task=document.task)
    if document.task == "dos":
        advice = advise_dos(
            structure,
            code=document.code,
            hpc=hpc,
            overrides=overrides,
            fetch_missing=document.fetch_missing,
        )
        report = check_dos(advice)
    elif document.task in ("relax", "vc-relax"):
        advice = advise_relax(
            structure,
            calculation=document.task,
            code=document.code,
            hpc=hpc,
            overrides=overrides,
            fetch_missing=document.fetch_missing,
        )
        report = check_relax(advice)
    else:
        advice = advise(
            structure,
            code=document.code,
            hpc=hpc,
            overrides=overrides,
            fetch_missing=document.fetch_missing,
        )
        report = check(advice)
    return {
        "records": records_document(advice.records()),
        "warnings": advice.warnings(),
        "blocking": list(dict.fromkeys(report.blocking)),
    }


def run(document: ComputeRequestDocument) -> tuple[dict[str, Any], BundleInput]:
    """Returns both the JSON-ready summary and the ``BundleInput`` it
    was built from -- HTTP's archive response mode reuses the latter to
    build the zip without recomputing the pipeline; MCP only ever needs
    the former."""
    structure = normalize_structure(_structure_source(document)).structure
    hpc = resolve_hpc_profile(document.hpc, field=_HPC_FIELD_NAME)
    overrides = build_overrides(document.overrides, task=document.task)
    ctx = default_shared_context()
    if document.task == "dos":
        dos_advice = advise_dos(
            structure,
            code=document.code,
            hpc=hpc,
            overrides=overrides,
            fetch_missing=document.fetch_missing,
        )
        dos_report = check_dos(dos_advice)
        steps = generate_dos(dos_advice, dos_report, ctx=ctx)
        script = render_submission_dos(dos_advice, hpc, document.code, steps)
        bundle_input = to_bundle_input_dos(dos_advice, steps, script, ctx)
        records, adv_warnings = dos_advice.records(), dos_advice.warnings()
    elif document.task in ("relax", "vc-relax"):
        relax_advice = advise_relax(
            structure,
            calculation=document.task,
            code=document.code,
            hpc=hpc,
            overrides=overrides,
            fetch_missing=document.fetch_missing,
        )
        relax_report = check_relax(relax_advice)
        # raises AdviceIncomplete if not ok
        steps = generate_relax(relax_advice, relax_report, ctx=ctx)
        script = render_submission_relax(relax_advice, hpc, document.code, steps)
        bundle_input = to_bundle_input_relax(relax_advice, steps, script, ctx)
        records, adv_warnings = relax_advice.records(), relax_advice.warnings()
    else:
        advice = advise(
            structure,
            code=document.code,
            hpc=hpc,
            overrides=overrides,
            fetch_missing=document.fetch_missing,
        )
        report = check(advice)
        steps = generate(advice, report)  # raises AdviceIncomplete if not ok
        script = render_submission(advice, hpc, document.code, steps)
        bundle_input = to_bundle_input(advice, steps, script, ctx)
        records, adv_warnings = advice.records(), advice.warnings()
    summary = {
        "files": [file["path"] for file in bundle_files(bundle_input)],
        "records": records_document(records),
        "warnings": adv_warnings,
    }
    return summary, bundle_input
