"""``goldilocks serve http|mcp``: now boots the v2-backed transports
(v2 epic 8, #8) -- ``server/http.py``/``server/mcp.py`` are rewired to
the ``service``/``capabilities``/``set_overrides`` stack in this same
epic. ``--static-root`` (#59) restores the Workbench static-file mount
CLI flag, once epic 12 (#12) actually wired the mount itself back up in
``server/http.py``'s ``create_app()`` -- the CLI-facing counterpart to
``GOLDILOCKS_WORKBENCH_STATIC_ROOT`` the Docker image sets instead
(see ``server/http.py``'s own docstring). P7's "optional deps imported
inside functions" applies here too: neither transport's extra needs to
be installed to run ``goldilocks run``/``explain``/etc.
"""

from __future__ import annotations

import argparse

from goldilocks_core.failures import ExpectedFailure


class ServeOptionError(ExpectedFailure, ValueError):
    """Options that do not go together for the chosen transport."""

    kind = "invalid_serve_option"
    category = "local"


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    serve = subparsers.add_parser("serve", help="Run an HTTP or MCP transport.")
    transports = serve.add_subparsers(dest="transport", required=True)
    http = transports.add_parser("http", help="Run the HTTP transport.")
    http.add_argument("--host", default="127.0.0.1")
    http.add_argument("--port", type=int, default=8000)
    http.add_argument(
        "--static-root",
        default=None,
        help=(
            "Serve the built Workbench frontend (a web/dist directory) at /. "
            "Defaults to the frontend bundled into this install, if this "
            "package was installed from a release build (#122)."
        ),
    )
    mcp = transports.add_parser(
        "mcp",
        help="Run the MCP transport, over stdio or streamable HTTP.",
        description=(
            "Serve the MCP tools over stdio, for a client that launches this "
            "command, or over streamable HTTP at http://HOST:PORT/mcp. "
            "Streamable HTTP has no authentication: keep the default local "
            "host, or put an authenticated proxy in front."
        ),
    )
    mcp.add_argument(
        "--transport",
        dest="mcp_transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="stdio (default) or streamable-http.",
    )
    mcp.add_argument(
        "--host",
        default=None,
        help="Streamable HTTP only: the address to listen on (default 127.0.0.1).",
    )
    mcp.add_argument(
        "--port",
        type=int,
        default=None,
        help="Streamable HTTP only: the port to listen on (default 8001).",
    )
    mcp.add_argument(
        "--publish-root",
        default=None,
        metavar="DIR",
        help=(
            "An existing directory where the run tool publishes bundles under "
            "the names callers give. Without it, run writes nothing."
        ),
    )


def run(args: argparse.Namespace) -> None:
    from goldilocks_core.advisors.magnetic_ordering_ml import mmace_device

    # Read before either transport starts, so a bad GOLDILOCKS_MMACE_DEVICE
    # stops the server and a cuda fallback to the CPU is logged now (#268).
    mmace_device()
    if args.transport == "http":
        from goldilocks_core.server.http import serve

        serve(host=args.host, port=args.port, static_root=args.static_root)
        return

    if args.mcp_transport == "stdio" and (args.host or args.port):
        raise ServeOptionError("--host and --port apply to --transport streamable-http")
    from goldilocks_core.server.mcp import serve

    serve(
        transport=args.mcp_transport,
        host=args.host or "127.0.0.1",
        port=args.port or 8001,
        publish_root=args.publish_root,
    )
