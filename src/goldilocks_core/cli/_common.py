"""Shared plumbing for the v2 CLI commands (v2 epic 8, #8).

Split out of ``core.py`` (and ``core.py`` itself split into one file per
command below) for the same reason ``service/`` is a package: this
project's own import-surface ceiling (``scripts/check_complexity.py``)
sets ``cli.core`` a *tighter* limit (8 origins/16 symbols) than the
default, precisely so the CLI stays a thin dispatcher and real logic
lives in dedicated modules -- v1's own ``cli/core.py`` already sat right
at that ceiling before this rewrite touched it.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import get_args

from pymatgen.core import Structure

from goldilocks_core.inputs.hpc import HpcProfile, resolve_hpc_profile
from goldilocks_core.inputs.structure import PathStructureSource, normalize_structure
from goldilocks_core.resolution import Blocked, FieldState, Resolved, ResolvedField
from goldilocks_core.service import RunOverrides
from goldilocks_core.set_overrides import (
    build_overrides,
    coerce_cli_assignments,
    parse_set_flags,
)
from goldilocks_core.types import CalcTask


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """The 7 flags ``run``/``explain`` share (design doc S6: "``run``
    only has 7 flags")."""
    parser.add_argument("structure", help="Path to the input structure file.")
    parser.add_argument(
        "--code", default="quantum_espresso", help="Code to generate input for."
    )
    parser.add_argument(
        "--task",
        default="scf_single_point",
        choices=get_args(CalcTask),
        help="Task to run.",
    )
    parser.add_argument(
        "--hpc",
        help="HPC profile name; required unless exactly one profile is installed.",
    )
    parser.add_argument(
        "--set",
        dest="set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one setting (repeatable). See 'goldilocks settings'.",
    )
    parser.add_argument(
        "--config", type=Path, help="TOML file of KEY=VALUE overrides, same as --set."
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")


def add_fetch_missing_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fetch-missing",
        action="store_true",
        help="Install a missing pseudopotential table instead of failing (P8: "
        "never a silent download).",
    )


def resolve_structure(path: str) -> Structure:
    return normalize_structure(PathStructureSource(path=path)).structure


def resolve_hpc(name: str | None) -> HpcProfile:
    return resolve_hpc_profile(name, field="--hpc")


def resolve_overrides(args: argparse.Namespace) -> RunOverrides:
    assignments: dict[str, object] = {}
    if args.config is not None:
        with args.config.open("rb") as source:
            assignments.update(tomllib.load(source))
    assignments.update(coerce_cli_assignments(parse_set_flags(args.set)))
    return build_overrides(assignments)


def format_field_state(state: FieldState[object]) -> str:
    """Tri-state-aware rendering -- directly fixes this epic's evidence
    #2 (v1's print layer collapsed "not computed", "not applicable",
    and "genuinely off" into one falsy-string fallback like
    ``'unresolved'``/``'none'``/``'any'``)."""
    if isinstance(state, Resolved):
        return f"{state.value!r} (source={state.source})"
    if isinstance(state, Blocked):
        return f"blocked by: {state.root_cause()}"
    return f"unavailable: {state.reason}"


def records_to_json(records: dict[str, FieldState[object]]) -> dict[str, object]:
    return {
        name: ResolvedField.from_state(state).model_dump()
        for name, state in sorted(records.items())
    }
