"""``goldilocks magnetic-orderings``: list candidate magnetic orderings for
a structure, optionally ranked by mMACE-relaxed energy (v2, #87).

Layer 3 (CLI) of #87's three-layer design (enumerate -> optionally rank
with mMACE -> generate). Deliberately its own command, not a ``run``/
``explain`` flag: this lists candidates for a human/agent to choose among
*before* generating any input file, the same reason
``service.list_magnetic_orderings`` sits outside ``advise()``/``generate()``
entirely. Generating input files for one or more chosen candidates is a
separate, later piece of this design (not yet wired to any existing
override) -- this command only lists and, optionally, ranks.
"""

from __future__ import annotations

import argparse
import json

from goldilocks_core.cli._common import resolve_structure
from goldilocks_core.service import list_magnetic_orderings, report_to_json


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "magnetic-orderings",
        help="List candidate magnetic orderings for a structure.",
    )
    parser.add_argument("structure", help="Path to the input structure file.")
    parser.add_argument(
        "--rank-with-mmace",
        action="store_true",
        help="Relax every candidate on the mMACE potential energy surface and "
        "recommend the lowest energy-per-atom one (needs GOLDILOCKS_MACE_BACKBONE "
        "and goldilocks-ml's magnetism extra; degrades to an unranked listing, "
        "with a warning, if either is unavailable).",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")


def run(args: argparse.Namespace) -> None:
    structure = resolve_structure(args.structure)
    report = list_magnetic_orderings(structure, rank_with_mmace=args.rank_with_mmace)

    if args.json:
        print(json.dumps(report_to_json(report), indent=2, sort_keys=True))
        return

    header = (
        f"{'label':10s} {'formula':10s} {'natoms':>6s} {'E/atom (eV)':>14s}  status"
    )
    print(header)
    for candidate in report.candidates:
        energy = (
            f"{candidate.energy_per_atom_ev:14.6f}"
            if candidate.energy_per_atom_ev is not None
            else " " * 14
        )
        marker = " *" if candidate.is_recommended else ""
        status = candidate.status or ("-" if not report.ranked else "")
        print(
            f"{candidate.label:10s} "
            f"{candidate.structure.composition.reduced_formula:10s} "
            f"{candidate.natoms:6d} {energy}  {status}{marker}"
        )
    for warning in report.warnings:
        print(f"warning: {warning.message}")
