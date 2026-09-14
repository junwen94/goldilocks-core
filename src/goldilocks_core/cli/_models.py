"""``goldilocks models list|show``: what ml models are installed (v2
epic 8, #8). Always empty today -- ml integration is deliberately last
(v2 epic 11, #11); this command exists now so scripts/docs referencing
it don't need to wait for that epic, and so the empty state is honest
rather than the command not existing at all.
"""

from __future__ import annotations

import argparse
import json

from goldilocks_core.capabilities import capabilities


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("models", help="List installed ml models.")
    model_commands = parser.add_subparsers(dest="models_command", required=True)
    model_commands.add_parser("list", help="List every installed model.")
    show = model_commands.add_parser("show", help="Show one model's details.")
    show.add_argument(
        "target", help="The ml_target to show (see 'goldilocks settings')."
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")


def run(args: argparse.Namespace) -> None:
    models = capabilities()["models"]
    if args.models_command == "show":
        matches = [model for model in models if model.get("target") == args.target]
        if args.json:
            print(json.dumps(matches, indent=2, sort_keys=True))
            return
        if not matches:
            print(f"no installed model for target {args.target!r}")
            return
        for model in matches:
            print(model)
        return

    if args.json:
        print(json.dumps(models, indent=2, sort_keys=True))
        return
    if not models:
        print("no ml models installed (ml integration lands in a later epic)")
        return
    for model in models:
        print(model)
