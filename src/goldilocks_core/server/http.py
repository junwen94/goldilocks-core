"""FastAPI transport over the v2 ``service``/``capabilities``/
``set_overrides`` stack (v2 epic 8, #8).

Replaces v1's ``Service``/``Runtime``-backed transport: every route now
calls ``server/_handlers.py``'s free functions, the exact same path
``server/mcp.py`` calls -- one shared request-validation/dispatch
implementation for both transports, not two.

**Workbench static-file mount.** Three ways to reach it, in precedence
order: an explicit ``static_root`` given to ``create_app()``/``serve()``;
``GOLDILOCKS_WORKBENCH_STATIC_ROOT``; a ``webapp/`` directory bundled
inside this installed package (#122) -- present only when this package
was built from a release that ran ``npm run build`` first (see
``.github/workflows/release.yml``), absent from a plain source checkout
or dev install, so ``uv build``/``uv sync`` here never needs a frontend
build lying around either. Whichever is found is mounted at ``/`` --
same origin as the API, no CORS (v2 epic 12, #12). ``cli/_serve.py``'s
own ``--static-root`` flag (#59) is the other, CLI-facing way to reach
the first tier -- the Docker image uses the env var instead since it has
no CLI invocation to add a flag to. A relative root is resolved against
the cwd once, when the app is created, so later requests never depend on
the process's cwd (#136).

**Strict JSON documents.** Every route returns a ``JSONResponse`` rather
than letting FastAPI serialize the document: ``JSONResponse`` raises on a
non-finite float (a 500), where FastAPI's serializer writes it as ``null``
(#137), and a TypedDict ``response_model`` would drop undeclared keys. The
``response_model``s therefore only document the responses in OpenAPI.

Behind the optional ``[http]`` extra; importing ``goldilocks_core``
never imports FastAPI (P7).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Annotated, Any

from goldilocks_core.bundle import archive_bytes
from goldilocks_core.capabilities import capabilities
from goldilocks_core.failures import ExpectedFailure
from goldilocks_core.server import _handlers
from goldilocks_core.server.documents import (
    CapabilitiesRequestDocument,
    CapabilitiesResponse,
    ComputeRequestDocument,
    ErrorResponse,
    ExplainResponse,
    InlineStructureDocument,
    InspectResponse,
    MagneticOrderingsRequestDocument,
    MagneticOrderingsResponse,
    RankingResponse,
    RunRequestDocument,
    RunResponse,
    invalid_request,
)

try:
    # Module-level, not deferred into create_app() like the rest of
    # FastAPI: with `from __future__ import annotations` in effect,
    # route return-type hints are strings resolved against *this
    # module's* __globals__ when FastAPI builds the OpenAPI schema
    # (#31). A name bound only inside create_app() never satisfies
    # that lookup, so .openapi() dies on a stale forward ref. The
    # fallback keeps the module importable without the [http] extra --
    # create_app()'s own try/except still raises _MISSING_HTTP_EXTRA
    # for anyone who actually calls it.
    from fastapi import Query
    from fastapi.responses import JSONResponse, Response
except ImportError:
    JSONResponse = Response = Query = Any  # type: ignore[assignment,misc]

__all__ = ["create_app", "serve"]

_MISSING_HTTP_EXTRA = (
    'The HTTP transport requires the http extra: install "goldilocks-core[http]".'
)
_STATIC_ROOT_ENV = "GOLDILOCKS_WORKBENCH_STATIC_ROOT"
_STATUS_BY_CATEGORY = {"input": 422, "dependency": 424, "local": 500}
_FAILURES: dict[int | str, dict[str, Any]] = {
    422: {
        "model": ErrorResponse,
        "description": (
            "The request was refused: invalid_request, invalid_setting, "
            "invalid_structure, invalid_hpc_profile, advice_incomplete "
            "(details.findings), generation_error or submission_error."
        ),
    },
    424: {
        "model": ErrorResponse,
        "description": (
            "A runtime asset on the server is missing or damaged: "
            "asset_not_installed or asset_corrupt."
        ),
    },
    500: {
        "model": ErrorResponse,
        "description": "A failure on the server, e.g. pseudo_import_error.",
    },
}
_UNKNOWN_RANKING: dict[int | str, dict[str, Any]] = {
    422: _FAILURES[422],
    404: {
        "model": ErrorResponse,
        "description": "unknown_ranking: this server holds no such job.",
    },
}
_ZIP: dict[int | str, dict[str, Any]] = {
    200: {
        "description": (
            "respond_with 'json': the summary below; 'archive': the whole "
            "bundle as a ZIP file."
        ),
        "content": {
            "application/zip": {"schema": {"type": "string", "format": "binary"}}
        },
    }
}


def create_app(*, static_root: str | Path | None = None) -> Any:
    try:
        from fastapi import FastAPI
    except ImportError as error:
        raise ImportError(_MISSING_HTTP_EXTRA) from error

    readiness = _handlers.build_readiness()
    rankings = _handlers.build_ranking_queue()

    @asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        # Warm the first ranking job's worker as the server starts (#285);
        # an app that is never started, as in tests, warms none.
        rankings.start()
        yield

    app = FastAPI(
        title="goldilocks-core",
        version=package_version("goldilocks-core"),
        description=(
            "Recommends DFT settings for a crystal structure and writes Quantum "
            "ESPRESSO input files. Every failure is "
            '{"error": {"kind", "message", "details"}}.'
        ),
        lifespan=lifespan,
    )
    app.state.asset_readiness = readiness
    _register_error_handlers(app)
    _register_operational_routes(app, readiness)

    @app.get(
        "/capabilities",
        tags=["discovery"],
        response_model=CapabilitiesResponse,
        responses=_FAILURES,
    )
    def get_capabilities(
        query: Annotated[CapabilitiesRequestDocument, Query()],
    ) -> Any:
        """What this server can prepare: codes, tasks and their steps,
        settings, facts, pseudopotential tables, HPC profiles, ML models and
        the warnings catalogue. ``?task=`` keeps only that task's settings."""
        return JSONResponse(capabilities(query.task))

    @app.post(
        "/inspect",
        tags=["structures"],
        response_model=InspectResponse,
        responses=_FAILURES,
    )
    def inspect(body: InlineStructureDocument) -> Any:
        """Parse a structure: its source, composition, lattice and sites,
        with a canonical CIF."""
        return JSONResponse(_handlers.inspect(body))

    @app.post(
        "/explain",
        tags=["inputs"],
        response_model=ExplainResponse,
        responses=_FAILURES,
    )
    def explain(body: ComputeRequestDocument) -> Any:
        """Every decision for the task, without generating files. A request
        run would refuse is still a 200 here, with its findings in
        ``blocking``."""
        return JSONResponse(_handlers.explain(body))

    @app.post(
        "/magnetic-orderings",
        tags=["structures"],
        response_model=MagneticOrderingsResponse,
        responses=_FAILURES,
    )
    def magnetic_orderings(body: MagneticOrderingsRequestDocument) -> Any:
        """The structure's candidate magnetic orderings, unranked. Rank them
        with ``POST /magnetic-orderings/rank``."""
        return JSONResponse(_handlers.magnetic_orderings(body))

    _register_ranking_routes(app, rankings)

    @app.post(
        "/run",
        tags=["inputs"],
        response_model=RunResponse,
        responses={**_FAILURES, **_ZIP},
    )
    def run(body: RunRequestDocument) -> Response:
        """Generate the task's input files and submission script, in memory:
        nothing is written on the server. A refused request is a 422
        ``advice_incomplete`` with ``details.findings``."""
        summary, bundle_input = _handlers.run(body)
        if body.respond_with == "archive":
            return Response(archive_bytes(bundle_input), media_type="application/zip")
        return JSONResponse(summary)

    _mount_workbench(app, static_root)
    return app


def _bundled_static_root() -> Path | None:
    """A ``webapp/`` directory shipped inside this installed package, if a
    release build put one there (#122) -- ``None`` for a plain source
    checkout or dev install, where nothing ever creates it."""
    from importlib import resources

    candidate = resources.files("goldilocks_core") / "webapp"
    if candidate.is_dir() and (candidate / "index.html").is_file():
        return Path(str(candidate))
    return None


def _mount_workbench(app: Any, static_root: str | Path | None) -> None:
    static_root = (
        static_root or os.environ.get(_STATIC_ROOT_ENV) or _bundled_static_root()
    )
    if not static_root:
        return
    from fastapi.staticfiles import StaticFiles

    # Mounted last, after every API route above: Starlette matches routes
    # in registration order, so the explicit routes still win even though
    # this Mount's prefix ("/") would otherwise swallow everything.
    # Resolved once, here: StaticFiles joins a relative directory with
    # whatever the cwd is on every request (#136). `check_dir` (default
    # True) still fails fast with a RuntimeError if it doesn't exist.
    app.mount(
        "/",
        StaticFiles(directory=Path(static_root).resolve(), html=True),
        name="workbench",
    )


def _register_error_handlers(app: Any) -> None:
    from fastapi import Request
    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content=invalid_request(error.errors()))

    @app.exception_handler(ExpectedFailure)
    async def expected_failure_handler(
        _request: Request, error: ExpectedFailure
    ) -> JSONResponse:
        status = _STATUS_BY_CATEGORY.get(error.category, 500)
        return JSONResponse(status_code=status, content={"error": error.public_error()})


def _register_ranking_routes(app: Any, queue: Any) -> None:
    """Ranking jobs (#261): submit, poll, cancel. A job id this server does
    not hold -- unknown, cancelled, long finished, or from before a restart
    -- is a 404."""

    @app.post(
        "/magnetic-orderings/rank",
        status_code=202,
        tags=["structures"],
        response_model=RankingResponse,
        responses=_FAILURES,
    )
    def submit_ranking(body: MagneticOrderingsRequestDocument) -> Any:
        """Queue a ranking of the structure's candidate orderings by mMACE
        energy per atom; returns the job's status."""
        return JSONResponse(_handlers.submit_ranking(queue, body), status_code=202)

    @app.get(
        "/magnetic-orderings/rank/{job_id}",
        tags=["structures"],
        response_model=RankingResponse,
        responses=_UNKNOWN_RANKING,
    )
    def ranking_status(job_id: str) -> Any:
        """The job's status. Poll at least every 120 s: a queued or running
        job nobody polls for 120 s is dropped. A finished job is kept until
        64 newer jobs finish, and a restart forgets every job."""
        status = queue.status(job_id)
        return _unknown_ranking(job_id) if status is None else JSONResponse(status)

    @app.delete(
        "/magnetic-orderings/rank/{job_id}",
        status_code=204,
        response_model=None,
        tags=["structures"],
        responses=_UNKNOWN_RANKING,
    )
    def cancel_ranking(job_id: str) -> Any:
        """Drop the job, stopping its ranking if it runs."""
        if not queue.cancel(job_id):
            return _unknown_ranking(job_id)
        return Response(status_code=204)


def _unknown_ranking(job_id: str) -> Any:
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "kind": "unknown_ranking",
                "message": (
                    f"No ranking job {job_id} on this server: it was cancelled, "
                    "finished long ago, or the server restarted. Run the "
                    "ranking again."
                ),
            }
        },
    )


def _register_operational_routes(app: Any, readiness: Any) -> None:
    @app.get("/health", tags=["operations"])
    async def health() -> dict[str, str]:
        """The process is up."""
        return {"status": "ok"}

    @app.get(
        "/ready",
        tags=["operations"],
        responses={
            503: {
                "model": ErrorResponse,
                "description": "assets_unavailable: a runtime asset is missing.",
            }
        },
    )
    def ready() -> dict[str, Any]:
        """Every registered runtime asset is installed and intact."""
        report = readiness.check()
        if report.ready:
            return {"status": "ready", "asset_count": report.asset_count}
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "kind": "assets_unavailable",
                    "message": (
                        f"Required runtime asset {report.asset_id}@{report.version} "
                        f"is {report.state}."
                    ),
                    "details": {
                        "asset_id": report.asset_id,
                        "version": report.version,
                        "state": report.state,
                    },
                }
            },
        )


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    static_root: str | Path | None = None,
) -> None:
    try:
        import uvicorn
    except ImportError as error:
        raise ImportError(_MISSING_HTTP_EXTRA) from error
    uvicorn.run(create_app(static_root=static_root), host=host, port=port)
