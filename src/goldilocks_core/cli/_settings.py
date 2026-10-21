"""``goldilocks settings``: list every ``--set``-able key, its default,
and where its value can come from (v2 epic 8, #8) -- design point (1)-b's
own illustrative table ("k_index integer human . ml(cgcnn-v2) . heuristic").
"""

from __future__ import annotations

import argparse
from typing import TextIO

from goldilocks_core.capabilities import capabilities
from goldilocks_core.cli._output import write_json


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "settings", help="List every --set-able key, its type, and its sources."
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")


def run(args: argparse.Namespace, out: TextIO) -> None:
    caps = capabilities()
    if args.json:
        write_json(out, caps["settings"])
        return
    for setting in sorted(caps["settings"], key=lambda item: item["key"]):
        default = setting.get("default")
        default_text = "" if default is None else f" (default {default!r})"
        sources = " . ".join(setting["approaches"])
        print(
            f"{setting['key']} [{setting['group']}] {setting['type']}{default_text}",
            file=out,
        )
        print(f"  sources: {sources}", file=out)
        if setting["description"]:
            print(f"  {setting['description']}", file=out)
