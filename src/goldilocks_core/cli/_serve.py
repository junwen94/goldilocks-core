"""``goldilocks serve http|mcp``: unchanged from v1 for now (v2 epic 8,
#8) -- ``server/http.py``/``server/mcp.py`` are rewired to the v2
``Service``/``capabilities``/``set_overrides`` stack in this epic's own
later modules, not this one (step 5's own sub-order: "CLI -> HTTP ->
MCP"). Until then this boots the still-live v1 transports, exactly as
v1's CLI did -- P7's "optional deps imported inside functions" applies
doubly here: neither transport's extra needs to be installed to run
``goldilocks run``/``explain``/etc.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    serve = subparsers.add_parser("serve", help="Run an HTTP or MCP transport.")
    transports = serve.add_subparsers(dest="transport", required=True)
    http = transports.add_parser("http", help="Run the HTTP transport.")
    http.add_argument("--host", default="127.0.0.1")
    http.add_argument("--port", type=int, default=8000)
    http.add_argument(
        "--static-root",
        type=Path,
        help="Directory containing the built Workbench.",
    )
    transports.add_parser("mcp", help="Run the MCP stdio transport.")


def run(args: argparse.Namespace) -> None:
    if args.transport == "http":
        from goldilocks_core.server.http import serve

        serve(host=args.host, port=args.port, static_root=args.static_root)
        return

    from goldilocks_core.server.mcp import serve

    serve()
