"""``goldilocks run``: advise -> generate -> submission script -> bundle,
published if ``-o/--out`` is given, otherwise a memory-only preview
(v2 epic 8, #8).

Memory-only is the default, not an auto-picked output directory: P8's
"never a silent download" extends in spirit to "never a silent write" --
``goldilocks run si.cif`` (the design doc's own "99% of the time"
example, with no ``-o``) previews exactly what *would* be published
without touching disk, via ``bundle.bundle_files`` (a pure function).
"""

from __future__ import annotations

import argparse
import json
import sys

from goldilocks_core.bundle import ArchiveOutput, DirectoryOutput, bundle_files, publish
from goldilocks_core.cli._common import (
    add_fetch_missing_argument,
    add_run_arguments,
    records_to_json,
    resolve_hpc,
    resolve_overrides,
    resolve_structure,
)
from goldilocks_core.service import (
    advise,
    advise_dos,
    check,
    check_dos,
    generate,
    generate_dos,
    render_submission,
    render_submission_dos,
    to_bundle_input,
    to_bundle_input_dos,
)
from goldilocks_core.steps import default_shared_context


def _print_blocked_and_exit(blocking: tuple[str, ...]) -> None:
    # Many fields can share one root cause (report.blocking has one
    # entry per blocked field, not per distinct cause); dedupe for a
    # human, not for a machine reading --json elsewhere.
    for reason in dict.fromkeys(blocking):
        print(f"blocked: {reason}", file=sys.stderr)
    raise SystemExit(2)


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "run", help="Generate a runnable input; publish it if -o/--out is given."
    )
    add_run_arguments(parser)
    add_fetch_missing_argument(parser)
    parser.add_argument(
        "-o",
        "--out",
        help="Directory (or a .zip path) to publish the bundle to; omit for a "
        "memory-only preview.",
    )


def run(args: argparse.Namespace) -> None:
    structure = resolve_structure(args.structure)
    hpc = resolve_hpc(args.hpc)
    overrides = resolve_overrides(args)
    ctx = default_shared_context()
    if args.task == "dos":
        dos_advice = advise_dos(
            structure,
            code=args.code,
            hpc=hpc,
            overrides=overrides,
            fetch_missing=args.fetch_missing,
        )
        dos_report = check_dos(dos_advice)
        if not dos_report.ok:
            _print_blocked_and_exit(dos_report.blocking)
        steps = generate_dos(dos_advice, dos_report, ctx=ctx)
        script = render_submission_dos(dos_advice, hpc, args.code, ctx, steps)
        bundle_input = to_bundle_input_dos(dos_advice, steps, script, ctx)
        advice_warnings = dos_advice.warnings()
    else:
        advice = advise(
            structure,
            code=args.code,
            hpc=hpc,
            overrides=overrides,
            fetch_missing=args.fetch_missing,
        )
        report = check(advice)
        if not report.ok:
            _print_blocked_and_exit(report.blocking)
        steps = generate(advice, report)
        script = render_submission(advice, hpc, args.code, ctx, steps)
        bundle_input = to_bundle_input(advice, steps, script, ctx)
        advice_warnings = advice.warnings()

    if args.out is not None:
        output = (
            ArchiveOutput(args.out)
            if args.out.endswith(".zip")
            else DirectoryOutput(args.out)
        )
        publication = publish(bundle_input, output)
        if args.json:
            print(json.dumps(publication, indent=2, sort_keys=True))
            return
        print(f"published {publication['kind']} to {publication['path']}")
        for file_path in publication["files"]:
            print(f"  {file_path}")
        return

    files = bundle_files(bundle_input)
    if args.json:
        print(
            json.dumps(
                {
                    "files": [file["path"] for file in files],
                    "records": records_to_json(bundle_input.records),
                    "warnings": advice_warnings,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print("memory-only preview (pass -o/--out to publish):")
    for file in files:
        print(f"  {file['path']} ({len(file['content'])} bytes)")
