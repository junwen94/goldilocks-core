"""FastAPI transport over the v2 ``service``/``capabilities``/
``set_overrides`` stack (v2 epic 8, #8).

Replaces v1's ``Service``/``Runtime``-backed transport: every route now
calls ``server/_handlers.py``'s free functions, the exact same path
``server/mcp.py`` calls -- one shared request-validation/dispatch
implementation for both transports, not two.

**No Workbench static-file mount.** v1's ``create_app`` also served the
built frontend off ``/`` when a static root was configured. Wiring the
Workbench back up to this epic's shapes (regenerated TypeScript types,
new tri-state/warnings rendering) is v2 epic 12 (#12)'s explicit scope,
not this one's -- see this epic's own issue, "downstream" section.

Behind the optional ``[http]`` extra; importing ``goldilocks_core``
never imports FastAPI (P7).
"""

from __future__ import annotations

from typing import Any

from goldilocks_core.bundle import archive_bytes
from goldilocks_core.capabilities import capabilities
from goldilocks_core.failures import ExpectedFailure
from goldilocks_core.server import _handlers
from goldilocks_core.server.documents import (
    ComputeRequestDocument,
    InlineStructureDocument,
    RunRequestDocument,
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
    from fastapi.responses import JSONResponse, Response
except ImportError:
    JSONResponse = Response = Any  # type: ignore[assignment,misc]

__all__ = ["create_app", "serve"]

_MISSING_HTTP_EXTRA = (
    "The HTTP transport requires goldilocks-core[http]. "
    "Install it with `uv sync --extra http`."
)
_STATUS_BY_CATEGORY = {"input": 422, "dependency": 424, "local": 500}


def create_app() -> Any:
    try:
        from fastapi import FastAPI
    except ImportError as error:
        raise ImportError(_MISSING_HTTP_EXTRA) from error

    readiness = _handlers.build_readiness()
    app = FastAPI(title="goldilocks-core")
    app.state.asset_readiness = readiness
    _register_error_handlers(app)
    _register_operational_routes(app, readiness)

    @app.get("/capabilities")
    def get_capabilities() -> Any:
        return capabilities()

    @app.post("/inspect")
    def inspect(body: InlineStructureDocument) -> Any:
        return _handlers.inspect(body)

    @app.post("/explain")
    def explain(body: ComputeRequestDocument) -> Any:
        return _handlers.explain(body)

    @app.post("/run")
    def run(body: RunRequestDocument) -> Response:
        summary, bundle_input = _handlers.run(body)
        if body.respond_with == "archive":
            return Response(archive_bytes(bundle_input), media_type="application/zip")
        return JSONResponse(summary)

    return app


def _register_error_handlers(app: Any) -> None:
    from fastapi import Request
    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        # pydantic's own error.errors() embeds the raw exception object
        # in ctx for custom validators (e.g. this module's own
        # _reject_path_shaped_content) -- not JSON-safe. Keep only the
        # string fields, same sanitization v1's handler already did.
        validation_errors = [
            {
                "path": ".".join(str(part) for part in item["loc"]),
                "type": item["type"],
                "message": item["msg"],
            }
            for item in error.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "kind": "invalid_request",
                    "message": "The request does not match the transport contract.",
                    "details": {"validation_errors": validation_errors},
                }
            },
        )

    @app.exception_handler(ExpectedFailure)
    async def expected_failure_handler(
        _request: Request, error: ExpectedFailure
    ) -> JSONResponse:
        status = _STATUS_BY_CATEGORY.get(error.category, 500)
        return JSONResponse(status_code=status, content={"error": error.public_error()})


def _register_operational_routes(app: Any, readiness: Any) -> None:
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> Any:
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


def serve(*, host: str = "127.0.0.1", port: int = 8000) -> None:
    try:
        import uvicorn
    except ImportError as error:
        raise ImportError(_MISSING_HTTP_EXTRA) from error
    uvicorn.run(create_app(), host=host, port=port)
