"""Transport-agnostic request handling shared by HTTP and MCP (v2 epic
8, #8) -- one implementation of "structure in, advice/bundle out",
called identically by both transports, directly answering this epic's
own scope item ("one shared request-validation path... invoked
identically by CLI, HTTP, and MCP before dispatch to Service"). The
CLI's own equivalent (``cli/_run.py``/``cli/_explain.py``) can't share
this exact code since it resolves structure from a local path, not
inline content -- but everything past structure resolution is the same
shape here as there: both look the task's service functions up in
``service.TASK_PHASES`` (#118).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from goldilocks_core.assets.store import AssetStore
from goldilocks_core.bundle import BundleInput, DirectoryOutput, bundle_files, publish
from goldilocks_core.inputs.hpc import resolve_hpc_profile
from goldilocks_core.inputs.structure import (
    InlineStructureSource,
    normalize_structure,
    require_ordered,
)
from goldilocks_core.resolution import records_document
from goldilocks_core.server.documents import (
    ComputeRequestDocument,
    ExplainResponse,
    InlineStructureDocument,
    InspectResponse,
    MagneticOrderingsRequestDocument,
    MagneticOrderingsResponse,
    MagneticOrderingsToolDocument,
    PublishRequestDocument,
    RunResponse,
)
from goldilocks_core.server.ranking_jobs import RankingQueue, RankingStatus
from goldilocks_core.server.readiness import AssetReadiness
from goldilocks_core.service import (
    TASK_PHASES,
    list_magnetic_orderings,
    report_to_json,
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


def inspect(document: InlineStructureDocument) -> InspectResponse:
    return normalize_structure(_structure_source(document)).inspection


def magnetic_orderings(
    document: MagneticOrderingsRequestDocument,
) -> MagneticOrderingsResponse:
    """The candidate orderings, unranked."""
    structure = normalize_structure(_structure_source(document)).structure
    return report_to_json(list_magnetic_orderings(structure))


def magnetic_orderings_tool(
    document: MagneticOrderingsToolDocument,
    on_progress: Callable[[int, int], None] | None = None,
) -> MagneticOrderingsResponse:
    """The candidate orderings, ranked in this call if the tool was asked to;
    ``on_progress(done, total)`` reports the relaxations of a ranking."""
    structure = normalize_structure(_structure_source(document)).structure
    report = list_magnetic_orderings(
        structure, rank_with_mmace=document.rank_with_mmace, on_progress=on_progress
    )
    return report_to_json(report)


def build_ranking_queue() -> RankingQueue:
    """The HTTP transport's queue of ranking jobs (#261)."""
    return RankingQueue()


def submit_ranking(
    queue: RankingQueue, document: MagneticOrderingsRequestDocument
) -> RankingStatus:
    """Queue a ranking of the document's structure; returns the job's
    status. A structure that cannot be listed is refused here, before it
    is queued."""
    structure = normalize_structure(_structure_source(document)).structure
    require_ordered(structure)
    return queue.submit(structure)


def explain(document: ComputeRequestDocument) -> ExplainResponse:
    """Every decision plus the verdict ``run`` reaches before generating
    (#165): ``blocking`` holds the document of each finding the task's
    check refuses generation for (#151), one per root cause in first-seen
    order, and is empty when ``run`` would go on to generate. ``layout``
    names each record's tier and step (#187, ``service.RecordTiers``)."""
    structure = normalize_structure(_structure_source(document)).structure
    hpc = resolve_hpc_profile(document.hpc, field=_HPC_FIELD_NAME)
    overrides = build_overrides(document.overrides, task=document.task)
    phases = TASK_PHASES[document.task]
    advice = phases.advise(
        structure,
        code=document.code,
        hpc=hpc,
        overrides=overrides,
    )
    report = phases.check(advice)
    tiers = advice.tiers()
    return {
        "records": records_document(tiers.records()),
        "warnings": advice.warnings(),
        "blocking": [finding.to_document() for finding in report.blocking],
        "layout": tiers.layout(),
    }


def run(document: ComputeRequestDocument) -> tuple[RunResponse, BundleInput]:
    """Returns both the JSON-ready summary and the ``BundleInput`` it
    was built from -- HTTP's archive response mode reuses the latter to
    build the zip without recomputing the pipeline; MCP only ever needs
    the former. The bundle records the request (#176): the body's
    ``overrides`` as sent and the submitted structure's identity. The
    summary's ``layout`` is ``explain``'s (#187)."""
    normalized = normalize_structure(_structure_source(document))
    hpc = resolve_hpc_profile(document.hpc, field=_HPC_FIELD_NAME)
    overrides = build_overrides(document.overrides, task=document.task)
    ctx = default_shared_context()
    phases = TASK_PHASES[document.task]
    advice = phases.advise(
        normalized.structure,
        code=document.code,
        hpc=hpc,
        overrides=overrides,
    )
    report = phases.check(advice)
    # raises AdviceIncomplete if not ok
    steps = phases.generate(advice, report, ctx=ctx)
    script = phases.render_submission(advice, hpc, document.code, steps)
    bundle_input = phases.to_bundle_input(
        advice,
        steps,
        script,
        ctx,
        code=document.code,
        hpc=hpc,
        assignments=document.overrides,
        source=normalized.source,
    )
    tiers = advice.tiers()
    summary: RunResponse = {
        "files": [file["path"] for file in bundle_files(bundle_input)],
        "records": records_document(tiers.records()),
        "warnings": advice.warnings(),
        "layout": tiers.layout(),
    }
    return summary, bundle_input


def publish_run(document: PublishRequestDocument, root: Path) -> RunResponse:
    """``run``'s summary; given a ``bundle_name``, the bundle is also published
    as the new directory ``root / bundle_name``, and its kind and path join
    the summary. An existing name raises ``bundle.BundleExists``."""
    summary, bundle_input = run(document)
    if document.bundle_name is None:
        return summary
    publication = publish(bundle_input, DirectoryOutput(root / document.bundle_name))
    return {**summary, "kind": "directory", "path": publication["path"]}
