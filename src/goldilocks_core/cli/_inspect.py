"""``goldilocks inspect``: structure info only, no analysis/advisors
(v2 epic 8, #8).
"""

from __future__ import annotations

import argparse
import json

from goldilocks_core.inputs.structure import PathStructureSource, normalize_structure


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("inspect", help="Inspect a structure file.")
    parser.add_argument("structure", help="Path to the input structure file.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")


def run(args: argparse.Namespace) -> None:
    inspection = normalize_structure(
        PathStructureSource(path=args.structure)
    ).inspection
    if args.json:
        print(json.dumps(inspection, indent=2, sort_keys=True))
        return
    structure_doc = inspection["structure"]
    elements = sorted(
        {
            species["symbol"]
            for site in structure_doc["sites"]
            for species in site["species"]
        }
    )
    print(f"formula: {structure_doc['reduced_formula']}")
    print(f"elements: {', '.join(elements)}")
    print(f"sites: {structure_doc['site_count']}")
    print(f"volume (A^3): {structure_doc['lattice']['volume_angstrom3']:.4f}")
