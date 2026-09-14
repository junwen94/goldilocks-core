"""size: resource_estimate, a pure arithmetic memory/workload estimate --
no four-tier resolution, no human/llm input, unlike every other advisor
in this codebase.

New in v2 (v2 epic 7, #7); no v1 precedent (v1 never estimates memory at
all). Per goldilocks-core-design.md:1637-1646: "purely arithmetic,
reproduces QE's own memory_report logic... +-10-20% precision. Not a
four-way choice, advisors and generation can both read it. The thing
that actually needs ml is not the memory estimate, it's walltime and
the optimal parallel layout" -- confirmed by reading QE's own
``PW/src/memory_report.f90`` directly (2026-09-14) rather than guessing
a formula.

**`npw`/`ngm`, QE's own formula, not approximated further**:
``memory_report.f90``'s ``npwx_g = NINT(fpi/3 * SQRT(ecutwfc)**3 /
(tpi**3/omega) / g_fact)`` is exactly ``npw = omega * ecutwfc**1.5 /
(6*pi**2)`` once ``fpi=4*pi``/``tpi=2*pi`` are expanded (``omega`` in
Bohr^3, ``ecutwfc`` in Ry) -- the volume of the reciprocal-space cutoff
sphere divided by the Brillouin-zone-cell volume. The same formula with
``ecutrho`` gives ``ngm``, the dense-grid G-vector count. ``g_fact``
(halves ``npw`` for a real Gamma-only wavefunction) is not modelled:
this codebase's ``advisors/k_sampling.py`` does not yet decide when a
calculation is Gamma-only-optimized, so this is a documented, not a
silent, simplification -- the estimate is conservative (slightly high)
without it, not wrong in the dangerous direction.

**What this deliberately does NOT reproduce from `memory_report.f90`**:
nonlocal-pseudopotential projector storage (needs per-species angular
momentum channel counts this codebase does not track), hybrid-functional
EXX buffers, Hubbard-projector buffers, and SCF-mixing history --
each is a real, sometimes-not-negligible contribution QE's own routine
accounts for, but every one needs data this codebase's advisors do not
yet produce. Only the two dominant, always-present terms are modelled:
wavefunction storage and charge-density/potential storage (``rho``,
``v``, ``vnew`` -- ``memory_report.f90``'s own ``scf_type_size`` times
3). Given the design doc's own +-10-20% accuracy bar, the two dominant
terms are treated as sufficient; the gap is documented, not silently
absorbed into a fudge factor.

**Wavefunction storage does not scale with the number of k-points.**
``memory_report.f90``'s own ``nk = 1 IF (io_level > 0 OR nks == 1) ELSE
nks + 1`` -- with QE's default disk I/O settings, only the *current*
k-point's wavefunctions are held in memory at a time, streamed to disk
between k-points, not all of them multiplied together. Assuming
otherwise would overestimate wavefunction memory by a factor of
``n_irr_k``, often the single largest term in a naive estimate.

**The dense real-space FFT grid (`nnr`) is a documented rule of
thumb, not QE's own value.** QE rounds each FFT grid dimension up to a
"5-smooth" length for transform efficiency, which this module cannot
reproduce without re-implementing QE's own dimension-search routine.
The real-space grid needs to be at least large enough to circumscribe
the reciprocal-space cutoff sphere -- a cube of side `2*sqrt(ecutrho)`
against a sphere of radius `sqrt(ecutrho)`, giving a volume ratio of
`6/pi ~= 1.91` -- and QE's own rounding adds a further, structure
-dependent margin on top. This module uses a flat factor of 2.0 as a
readable middle estimate; it is not calibrated against real QE runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pymatgen.core import Structure

_BOHR_PER_ANGSTROM = 1.8897259886
_NPW_CONSTANT = 1.0 / (6.0 * math.pi**2)
_FFT_GRID_TO_NGM_RATIO = 2.0
"""Documented rule of thumb, not QE's own value -- see module docstring."""
_COMPLEX_BYTES = 16
_REAL_BYTES = 8
_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    npw: int
    ngm: int
    fft_grid_points: int
    ram_mb: float


def resource_estimate(
    structure: Structure,
    ecutwfc_ry: float,
    ecutrho_ry: float,
    nbnd: int,
    nspin: int = 1,
    noncollinear: bool = False,
) -> ResourceEstimate:
    """Pure arithmetic; no `FieldState`, no human/llm input (design doc:
    "not a four-way choice"). Every argument here is a plain, already
    -resolved value -- a caller with anything still `Unavailable`/
    `Blocked` should not call this yet.
    """
    volume_bohr3 = structure.lattice.volume * _BOHR_PER_ANGSTROM**3

    npw = math.ceil(_NPW_CONSTANT * volume_bohr3 * ecutwfc_ry**1.5)
    ngm = math.ceil(_NPW_CONSTANT * volume_bohr3 * ecutrho_ry**1.5)
    fft_grid_points = math.ceil(_FFT_GRID_TO_NGM_RATIO * ngm)

    npol = 2 if noncollinear else 1
    wavefunction_bytes = _COMPLEX_BYTES * nbnd * npol * npw

    # rho, v, vnew (memory_report.f90's scf_type_size, x3), each holding
    # a reciprocal-space (ngm) and a real-space (fft_grid_points) part.
    scf_type_bytes = (_COMPLEX_BYTES * ngm + _REAL_BYTES * fft_grid_points) * nspin
    density_bytes = 3 * scf_type_bytes

    ram_mb = (wavefunction_bytes + density_bytes) / _BYTES_PER_MB
    return ResourceEstimate(
        npw=npw, ngm=ngm, fft_grid_points=fft_grid_points, ram_mb=ram_mb
    )


def envelope(estimates: list[ResourceEstimate]) -> ResourceEstimate:
    """Combine several steps' estimates into one envelope for job sizing:
    the largest single-step footprint along every axis, not a sum --
    steps run one after another within a task, never concurrently,
    so nothing is ever held in memory across two steps at once.
    """
    if not estimates:
        raise ValueError("envelope() requires at least one estimate")
    return ResourceEstimate(
        npw=max(e.npw for e in estimates),
        ngm=max(e.ngm for e in estimates),
        fft_grid_points=max(e.fft_grid_points for e in estimates),
        ram_mb=max(e.ram_mb for e in estimates),
    )
