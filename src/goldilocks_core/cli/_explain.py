"""``goldilocks explain``: analysis+advisors only, printing every
decision and its source (v2 epic 8, #8). Always returns -- per the
"diagnosis is always available" promise, `service.advise()` never
raises for a scientifically-incomplete request; only a genuinely
malformed structure file does.
"""

from __future__ import annotations

import argparse
import json

from goldilocks_core.cli._common import (
    add_fetch_missing_argument,
    add_run_arguments,
    format_field_state,
    records_to_json,
    resolve_hpc,
    resolve_overrides,
    resolve_structure,
)
from goldilocks_core.service import advise


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "explain",
        help="Run analysis+advisors only; print every decision and its source.",
    )
    add_run_arguments(parser)
    add_fetch_missing_argument(parser)


def run(args: argparse.Namespace) -> None:
    structure = resolve_structure(args.structure)
    hpc = resolve_hpc(args.hpc)
    overrides = resolve_overrides(args)
    advice = advise(
        structure,
        code=args.code,
        hpc=hpc,
        overrides=overrides,
        fetch_missing=args.fetch_missing,
    )
    records = advice.records()
    if args.json:
        print(json.dumps(records_to_json(records), indent=2, sort_keys=True))
        return
    for name, state in sorted(records.items()):
        print(f"{name}: {format_field_state(state)}")
