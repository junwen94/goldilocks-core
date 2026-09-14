"""``goldilocks settings``: list every ``--set``-able key, its default,
and where its value can come from (v2 epic 8, #8) -- design point (1)-b's
own illustrative table ("k_index integer human . ml(cgcnn-v2) . heuristic").
"""

from __future__ import annotations

import argparse
import json

from goldilocks_core.capabilities import capabilities


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "settings", help="List every --set-able key, its type, and its sources."
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")


def run(args: argparse.Namespace) -> None:
    caps = capabilities()
    if args.json:
        print(json.dumps(caps["settings"], indent=2, sort_keys=True))
        return
    for setting in sorted(caps["settings"], key=lambda item: item["key"]):
        default = setting.get("default")
        default_text = "" if default is None else f" (default {default!r})"
        sources = " . ".join(setting["approaches"])
        print(f"{setting['key']} [{setting['group']}] {setting['type']}{default_text}")
        print(f"  sources: {sources}")
        if setting["description"]:
            print(f"  {setting['description']}")
