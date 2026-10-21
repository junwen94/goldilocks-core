"""``goldilocks explain``: analysis, advisors and ``run``'s check, without
generating anything (v2 epic 8, #8). Prints every decision with its source,
then one ``blocked:`` line per distinct reason ``run`` would refuse to
generate an input for (#165). A scientifically incomplete or blocked request
still prints and exits 0: ``service.advise()`` marks what it cannot decide
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
from goldilocks_core.service import (
    advise,
    advise_dos,
    advise_relax,
    check,
    check_dos,
    check_relax,
)


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "explain",
        help=(
            "Run analysis+advisors without generating files; print every "
            "decision, its source, and the blocking reasons run would report."
        ),
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
        report = check_dos(advice)
    elif args.task in ("relax", "vc-relax"):
        advice = advise_relax(
            structure,
            calculation=args.task,
            code=args.code,
            hpc=hpc,
            overrides=overrides,
            fetch_missing=args.fetch_missing,
        )
        report = check_relax(advice)
    else:
        advice = advise(
            structure,
            code=args.code,
            hpc=hpc,
            overrides=overrides,
            fetch_missing=args.fetch_missing,
        )
        report = check(advice)
    records = advice.records()
    blocking = list(dict.fromkeys(report.blocking))
    if args.json:
        write_json(
            out,
            {
                "records": records_to_json(records),
                "warnings": advice.warnings(),
                "blocking": blocking,
            },
        )
        return
    for name, state in sorted(records.items()):
        print(f"{name}: {format_field_state(state)}", file=out)
    for reason in blocking:
        print(f"blocked: {reason}", file=out)
