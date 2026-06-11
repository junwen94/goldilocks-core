from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_Provenance = Literal["heuristic", "ML", "MLIP", "user_hint"]


@dataclass(frozen=True, slots=True)
class Protocol:
    """Coupled smearing-width + k-distance sampling protocol."""

    name: Literal["stringent", "balanced", "fast"]
    smearing_width_ry: float
    smearing_width_ev: float
    k_distance: float          # Å⁻¹


# ---------------------------------------------------------------------------
# Layer 1: code-agnostic advisory decisions (internal to advise/)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SmearingDecision:
    """Physics decision about occupation broadening.

    method uses physics names; per-code Layer 2 translates:
      marzari_vanderbilt → QE "mv"; VASP fallback → ISMEAR=1
    """

    use_smearing: bool
    method: Literal["marzari_vanderbilt", "methfessel_paxton", "fermi_dirac", "gaussian"] | None
    width_ev: float | None  # eV; None when use_smearing=False
    provenance: _Provenance
    rationale: str


@dataclass(frozen=True, slots=True)
class KPointsDecision:
    """K-point sampling decision (code-agnostic; Monkhorst-Pack grid)."""

    grid: tuple[int, int, int]
    shift: tuple[int, int, int]  # 0 or 1 per axis
    provenance: _Provenance
    rationale: str


@dataclass(frozen=True, slots=True)
class SpinDecision:
    """Spin and magnetic treatment decision.

    treatment values:
      non_magnetic      — nspin=1
      collinear         — nspin=2
      non_collinear     — noncolin=T, lspinorb=F (frustrated magnets, no SOC)
      non_collinear_soc — noncolin=T, lspinorb=T (heavy elements + non-collinear)

    initial_magnetization: element → initial moment in μB.
      Collinear: only magnetic species included; QE default (0) applies to others.
      Non-collinear: ALL species included; non-magnetic species use 0.1 μB
        (aiida-quantumespresso convention) so QE doesn't collapse to zero.
      Generate stage converts to QE starting_magnetization via magmom / z_val.

    angle1 / angle2: polar / azimuthal angle (degrees) per element.
      None for collinear. For non-collinear defaults to 0.0 / 0.0 (FM along z).
    """

    treatment: Literal["non_magnetic", "collinear", "non_collinear", "non_collinear_soc"]
    initial_magnetization: dict[str, float] | None  # μB; None = non-magnetic
    angle1: dict[str, float] | None                 # degrees; None for collinear
    angle2: dict[str, float] | None                 # degrees; None for collinear
    provenance: _Provenance
    rationale: str


@dataclass(frozen=True, slots=True)
class CutoffDecision:
    """Plane-wave cutoff decision (eV, code-agnostic)."""

    wavefunction_cutoff_ev: float
    density_cutoff_ev: float
    provenance: _Provenance
    rationale: str


# ---------------------------------------------------------------------------
# Universal
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PseudoSelection:
    """Pseudopotential selected for one element (UPF-based, multi-code)."""

    element: str
    family: str            # aiida-pseudo label
    filename: str          # e.g. "Fe.upf"
    path: Path | None      # local UPF path; None when using aiida-pseudo
    wavefunction_cutoff_ev: float
    density_cutoff_ev: float
    provenance: _Provenance


# ---------------------------------------------------------------------------
# Layer 2: QE-specific parameter set
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class QEParameterSet:
    """QE input parameters ready for pw.x. Provenance lives in the decision fields."""

    # Smearing (SYSTEM card)
    occupations: Literal["smearing", "fixed"]
    smearing: str | None      # "mv" | "mp" | "fd" | "gauss"; None when fixed
    degauss: float             # Ry

    # K-points
    kpoints_grid: tuple[int, int, int]
    kpoints_shift: tuple[int, int, int]

    # Pseudos and cutoffs (SYSTEM card)
    pseudos: list[PseudoSelection]
    ecutwfc: float             # Ry
    ecutrho: float             # Ry

    # Spin (SYSTEM card)
    nspin: Literal[1, 2, 4]
    noncolin: bool
    lspinorb: bool
    starting_magnetization: dict[str, float] | None  # μB; Generate stage divides by z_val
    angle1: dict[str, float] | None                  # degrees; non-None for noncolin
    angle2: dict[str, float] | None                  # degrees; non-None for noncolin

    # Provenance (code-agnostic decisions that produced the above)
    smearing_decision: SmearingDecision
    kpoints_decision: KPointsDecision
    cutoff_decision: CutoffDecision
    spin_decision: SpinDecision
