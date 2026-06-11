"""Heuristic phonon setup adviser.

Main entry point: ``advise_ph_setup(structure, analysis, accuracy, base_kgrid)``

Knowledge basis
---------------
Phonon accuracy depends on a full convergence chain:
  structure relax → SCF conv_thr → k-grid → smearing → q-grid → ASR → NAC

Key rules encoded here:

q-grid sizing (``advise_qgrid``)
    nq_i = max(min_nq, ceil(target_range / a_i))
    Target range: 8/10/12 Å for fast/balanced/accurate.
    Metal: ×1.2 multiplier (Fermi-surface oscillations).
    2D slab: vacuum direction forced to nq=1.

Phonon k-grid (``_phonon_kgrid``)
    Must be commensurate with the q-grid: nk_i mod nq_i == 0.
    Metal: mult = 3/4/5 × nq (fast/balanced/accurate).
    Insulator: mult = 2/3/3 × nq.
    Commensurability is the hard constraint; density is the quality target.

SCF convergence
    conv_thr = 1e-10 (tighter than the default 1e-8 used for total energies).

ph.x convergence
    tr2_ph = 1e-14 (standard recommendation).

LO-TO splitting (epsil)
    Polar insulators (is_polar AND insulating/likely_insulating) need
    Born effective charges and a dielectric tensor.  Set epsil=.true. in ph.x,
    then apply non-analytical correction in matdyn.x.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pymatgen.core import Structure

    from goldilocks_core.analyse.structure import StructureAnalysis

from goldilocks_core.advise.types import PhononSetupAdvice, QGridAdvice

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_RANGE: dict[str, float] = {
    "fast":     8.0,
    "balanced": 10.0,
    "accurate": 12.0,
}
_METAL_MULTIPLIER   = 1.2
_METAL_MIN_NQ       = 3
_DEFAULT_MIN_NQ     = 2

_METAL_KMULT:     dict[str, int] = {"fast": 3, "balanced": 4, "accurate": 5}
_INSULATOR_KMULT: dict[str, int] = {"fast": 2, "balanced": 3, "accurate": 3}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def advise_ph_setup(
    structure: "Structure",
    analysis: "StructureAnalysis",
    accuracy: Literal["fast", "balanced", "accurate"],
    base_kgrid: tuple[int, int, int],
) -> PhononSetupAdvice:
    """Return a comprehensive heuristic setup for a ph.x phonon run.

    Args:
        structure:   Crystal structure (for lattice parameters).
        analysis:    StructureAnalysis (metallicity, dimensionality, polarity, etc.).
        accuracy:    Accuracy tier from CalculationIntent.
        base_kgrid:  SCF k-grid already recommended for this material.  Used as
                     a lower bound when computing the phonon k-grid.

    Returns:
        ``PhononSetupAdvice`` — all parameters needed to generate phonon-ready
        ``goldilocks.in`` and ``ph.in`` files, plus advisory warnings.
    """
    q_grid      = advise_qgrid(structure, analysis, accuracy)
    phonon_kg   = _phonon_kgrid(q_grid.nq, analysis, accuracy, base_kgrid)
    needs_epsil = analysis.is_polar and analysis.metallicity in (
        "insulating", "likely_insulating"
    )
    warnings    = _build_warnings(analysis, needs_epsil)

    return PhononSetupAdvice(
        q_grid        = q_grid,
        phonon_kgrid  = phonon_kg,
        scf_conv_thr  = 1e-10,
        tr2_ph        = 1e-14,
        needs_epsil   = needs_epsil,
        warnings      = warnings,
        provenance    = "heuristic",
    )


def advise_qgrid(
    structure: "Structure",
    analysis: "StructureAnalysis",
    accuracy: Literal["fast", "balanced", "accurate"],
) -> QGridAdvice:
    """Return a heuristic q-point grid for a ph.x calculation.

    Args:
        structure: Crystal structure (for lattice parameters).
        analysis:  StructureAnalysis (metallicity, dimensionality, pbc).
        accuracy:  Accuracy tier.

    Returns:
        ``QGridAdvice`` with ``nq``, ``target_range_aa``, and ``rationale``.
    """
    a = structure.lattice.a
    b = structure.lattice.b
    c = structure.lattice.c

    base     = _BASE_RANGE[accuracy]
    is_metal = analysis.metallicity in ("metallic", "likely_metallic")

    if analysis.dimensionality == "2d" or analysis.is_slab:
        mult     = _METAL_MULTIPLIER if is_metal else 1.0
        target_a = base * mult
        target_b = base * mult
        target_c = 0.0            # vacuum direction → nq=1
        dim_tag  = "2D/slab (vacuum direction → nq=1)"
    elif analysis.dimensionality == "1d":
        mult     = _METAL_MULTIPLIER if is_metal else 1.0
        target_a = base * mult if analysis.pbc[0] else 0.0
        target_b = base * mult if analysis.pbc[1] else 0.0
        target_c = base * mult if analysis.pbc[2] else 0.0
        dim_tag  = "1D wire (non-periodic directions → nq=1)"
    else:
        mult     = _METAL_MULTIPLIER if is_metal else 1.0
        target_a = target_b = target_c = base * mult
        dim_tag  = None

    effective_range = base * mult
    min_nq = _METAL_MIN_NQ if is_metal else _DEFAULT_MIN_NQ

    def _nq(target: float, length: float) -> int:
        return 1 if target == 0.0 else max(min_nq, math.ceil(target / length))

    nq1, nq2, nq3 = _nq(target_a, a), _nq(target_b, b), _nq(target_c, c)

    return QGridAdvice(
        nq              = (nq1, nq2, nq3),
        target_range_aa = effective_range,
        provenance      = "heuristic",
        rationale       = _qgrid_rationale(
            accuracy, base, is_metal, mult, effective_range,
            a, b, c, nq1, nq2, nq3, dim_tag,
        ),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _phonon_kgrid(
    nq: tuple[int, int, int],
    analysis: "StructureAnalysis",
    accuracy: Literal["fast", "balanced", "accurate"],
    base_kgrid: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Phonon k-grid: commensurate with q-grid and denser than normal SCF.

    Commensurability constraint: nk_i must be a multiple of nq_i.
    Rule: nk_i = max(base_kgrid_i, mult × nq_i) rounded up to the next
    multiple of nq_i.
    """
    is_metal = analysis.metallicity in ("metallic", "likely_metallic")
    mult = _METAL_KMULT[accuracy] if is_metal else _INSULATOR_KMULT[accuracy]

    result: list[int] = []
    for nq_i, nk_base in zip(nq, base_kgrid):
        if nq_i <= 1:
            # Vacuum / non-periodic direction: keep SCF k-grid as-is.
            result.append(nk_base)
        else:
            target = max(nk_base, mult * nq_i)
            # Round up to next multiple of nq_i for commensurability.
            nk = math.ceil(target / nq_i) * nq_i
            result.append(nk)
    return (result[0], result[1], result[2])


def _build_warnings(
    analysis: "StructureAnalysis",
    needs_epsil: bool,
) -> list[str]:
    warnings: list[str] = []

    # Magnetic materials: wrong spin state → wrong phonons
    if analysis.magnetic_elements:
        els = ", ".join(analysis.magnetic_elements)
        warnings.append(
            f"Magnetic material ({els}): confirm the magnetic ground state "
            "(FM / AFM / ferrimagnetic) before running phonons — "
            "the wrong spin configuration gives wrong force constants."
        )

    # Polar insulator: LO-TO splitting
    if needs_epsil:
        warnings.append(
            "Polar insulator: epsil=.true. applied to ph.in → Born effective charges "
            "+ dielectric tensor will be computed.  After q2r.x, use "
            "matdyn.x with non-analytical correction for correct LO-TO splitting."
        )

    # Metal: additional convergence tests recommended
    if analysis.metallicity in ("metallic", "likely_metallic"):
        warnings.append(
            "Metal phonon: test both k-grid and smearing convergence. "
            "Try degauss = 0.01, 0.02, 0.03 Ry; if phonon frequencies shift "
            "significantly, the electronic integration is not yet converged."
        )

    return warnings


def _qgrid_rationale(
    accuracy: str,
    base: float,
    is_metal: bool,
    mult: float,
    effective_range: float,
    a: float, b: float, c: float,
    nq1: int, nq2: int, nq3: int,
    dim_tag: str | None,
) -> str:
    parts: list[str] = [f"{accuracy} tier → base IFC range {base:.0f} Å"]
    if is_metal:
        parts.append(
            f"metal ×{_METAL_MULTIPLIER} → {effective_range:.0f} Å "
            "(Fermi-surface oscillations)"
        )
    if dim_tag:
        parts.append(dim_tag)
    if dim_tag is None:
        parts.append(
            f"lattice {a:.2f}×{b:.2f}×{c:.2f} Å → "
            f"ceil({effective_range:.0f}/{a:.2f})={nq1}, "
            f"ceil({effective_range:.0f}/{b:.2f})={nq2}, "
            f"ceil({effective_range:.0f}/{c:.2f})={nq3}"
        )
    else:
        parts.append(f"lattice {a:.2f}×{b:.2f}×{c:.2f} Å → nq = {nq1} {nq2} {nq3}")
    parts.append(
        "Verify: compare phonon dispersion / lowest frequency at successive q-grids "
        + (f"{nq1}×{nq2}×{nq3} → larger." if a >= 4 else "4×4×4 → 6×6×6 → 8×8×8.")
    )
    return "  ".join(parts)
