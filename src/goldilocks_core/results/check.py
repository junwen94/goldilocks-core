"""Consistency validation: compare DFT results against the goldilocks manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from goldilocks_core.results.types import RelaxResult, SCFResult, ValidationReport

_DEFAULT_MAX_ITER = 100     # QE default electron_maxstep
_HIGH_ITER_FRACTION = 0.8   # warn if n_iter > 80 % of max
_MAG_THRESHOLD = 0.5        # μB: unexpected magnetisation threshold


def validate(
    result: "SCFResult | RelaxResult",
    manifest_path: str | Path | None = None,
    max_iterations: int = _DEFAULT_MAX_ITER,
) -> "ValidationReport":
    """Check DFT result against the goldilocks manifest and internal guardrails.

    Args:
        result: Parsed SCF or Relax result.
        manifest_path: Path to ``goldilocks_manifest.json`` written by
            ``write_qe_inputs()``.  Pass None to skip manifest checks.
        max_iterations: Maximum SCF iteration count used in the calculation
            (default 100, matching QE's ``electron_maxstep``).

    Returns:
        ValidationReport with ``passed`` items and ``warnings`` list.
    """
    from goldilocks_core.results.types import (
        RelaxResult,
        SCFResult,
        ValidationReport,
        ValidationWarning,
    )

    passed: list[str] = []
    warnings: list[ValidationWarning] = []

    manifest: dict | None = None
    if manifest_path is not None:
        p = Path(manifest_path)
        if p.exists():
            manifest = json.loads(p.read_text())

    # ── SCF-specific checks ──────────────────────────────────────────────────
    if isinstance(result, SCFResult):
        if result.converged:
            passed.append("scf_convergence")
            if result.n_iterations > max_iterations * _HIGH_ITER_FRACTION:
                warnings.append(ValidationWarning(
                    level="warning",
                    parameter="scf_iterations",
                    message=(
                        f"SCF converged in {result.n_iterations} / {max_iterations} iterations "
                        f"({result.n_iterations / max_iterations:.0%} of limit)"
                    ),
                    suggestion="Consider increasing mixing_beta or switching mixing_mode",
                ))
        else:
            warnings.append(ValidationWarning(
                level="error",
                parameter="scf_convergence",
                message="SCF did not converge",
                suggestion=(
                    "Increase electron_maxstep, adjust mixing_beta, "
                    "or review the starting geometry/magnetisation"
                ),
            ))

        # Magnetism consistency against manifest
        if manifest is not None:
            spin_treatment = (
                manifest.get("parameters", {}).get("spin", {}).get("treatment")
            )
            if (
                spin_treatment == "non_magnetic"
                and result.total_magnetization is not None
                and abs(result.total_magnetization) > _MAG_THRESHOLD
            ):
                warnings.append(ValidationWarning(
                    level="warning",
                    parameter="magnetism_consistency",
                    message=(
                        f"Non-magnetic run produced total_magnetization = "
                        f"{result.total_magnetization:.2f} μB — unexpected spin polarisation"
                    ),
                    suggestion="Re-run with hint spin_treatment=collinear",
                ))
            else:
                passed.append("magnetism_consistency")

    # ── Relax-specific checks ────────────────────────────────────────────────
    elif isinstance(result, RelaxResult):
        if result.converged:
            passed.append("relax_convergence")
        else:
            warnings.append(ValidationWarning(
                level="error",
                parameter="relax_convergence",
                message="Geometry relaxation did not converge",
                suggestion="Increase nstep or tighten ion_dynamics thresholds",
            ))

    return ValidationReport(passed=passed, warnings=warnings, manifest=manifest)
