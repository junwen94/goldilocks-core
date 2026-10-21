"""``goldilocks explain``: analysis+advisors only, printing every
decision and its source (v2 epic 8, #8). A scientifically incomplete
request still prints: ``service.advise()`` marks what it cannot decide
unavailable or blocked instead of raising.
"""

from __future__ import annotations

import argparse
from typing import TextIO

from goldilocks_core.cli._common import (
    add_fetch_missing_argument,
    add_run_arguments,
    format_field_state,
    records_to_json,
    resolve_hpc,
    resolve_overrides,
    resolve_structure,
)
from goldilocks_core.cli._output import write_json
from goldilocks_core.service import advise, advise_dos, advise_relax


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "explain",
        help="Run analysis+advisors only; print every decision and its source.",
    )
    add_run_arguments(parser)
    add_fetch_missing_argument(parser)


def run(args: argparse.Namespace, out: TextIO) -> None:
    structure = resolve_structure(args.structure)
    hpc = resolve_hpc(args.hpc)
    overrides = resolve_overrides(args)
    if args.task == "dos":
        advice = advise_dos(
            structure,
            code=args.code,
            hpc=hpc,
            overrides=overrides,
            fetch_missing=args.fetch_missing,
        )
    elif args.task in ("relax", "vc-relax"):
        advice = advise_relax(
            structure,
            calculation=args.task,
            code=args.code,
            hpc=hpc,
            overrides=overrides,
            fetch_missing=args.fetch_missing,
        )
    else:
        advice = advise(
            structure,
            code=args.code,
            hpc=hpc,
            overrides=overrides,
            fetch_missing=args.fetch_missing,
        )
    records = advice.records()
    if args.json:
        write_json(
            out,
            {"records": records_to_json(records), "warnings": advice.warnings()},
        )
        return
    for name, state in sorted(records.items()):
        print(f"{name}: {format_field_state(state)}", file=out)
