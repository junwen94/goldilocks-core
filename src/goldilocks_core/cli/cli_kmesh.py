"""CLI entry point for k-mesh recommendation."""

from __future__ import annotations

import argparse

from goldilocks_core.advise.pipeline import build_qe_parameter_set
from goldilocks_core.analyse.structure import analyze_structure
from goldilocks_core.intent import CalculationIntent
from goldilocks_core.io.structures import load_structure


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for k-mesh recommendation."""
    parser = argparse.ArgumentParser(
        prog="goldilocks-kmesh",
        description="Recommend a k-point mesh for a structure.",
    )
    parser.add_argument(
        "structure",
        help="Path to the input structure file.",
    )
    parser.add_argument(
        "--accuracy",
        choices=["fast", "balanced", "accurate"],
        default="accurate",
        help="Accuracy tier (default: accurate).",
    )
    return parser


def main() -> None:
    """Run the k-mesh recommendation CLI."""
    parser = build_parser()
    args = parser.parse_args()

    structure = load_structure(args.structure)
    analysis = analyze_structure(structure)
    intent = CalculationIntent(structure=structure, accuracy=args.accuracy)
    params = build_qe_parameter_set(analysis, intent)

    print(f"recommended mesh:  {params.kpoints_grid}")
    print(f"shift:             {params.kpoints_shift}")
    print(f"protocol:          {params.kpoints_decision.rationale}")


if __name__ == "__main__":
    main()
