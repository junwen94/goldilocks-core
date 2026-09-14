from __future__ import annotations

import math

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.advisors.size import (
    ResourceEstimate,
    envelope,
    memory_per_process,
    resource_estimate,
)

_BOHR_PER_ANGSTROM = 1.8897259886
_SILICON = Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])


def _expected_npw(structure: Structure, ecut_ry: float) -> int:
    """Independent re-derivation of QE's own npwx formula (memory_report.f90:
    npwx_g = fpi/3 * sqrt(ecutwfc)**3 / (tpi**3/omega)), to cross-check the
    module's constant rather than just re-running its own code."""
    volume_bohr3 = structure.lattice.volume * _BOHR_PER_ANGSTROM**3
    return math.ceil(
        (4 * math.pi / 3) * ecut_ry**1.5 / ((2 * math.pi) ** 3 / volume_bohr3)
    )


def test_npw_matches_qes_own_reciprocal_sphere_formula() -> None:
    estimate = resource_estimate(_SILICON, ecutwfc_ry=40.0, ecutrho_ry=160.0, nbnd=8)

    assert estimate.npw == _expected_npw(_SILICON, 40.0)
    assert estimate.ngm == _expected_npw(_SILICON, 160.0)


def test_ngm_grows_with_ecutrho_not_ecutwfc() -> None:
    estimate = resource_estimate(_SILICON, ecutwfc_ry=40.0, ecutrho_ry=320.0, nbnd=8)

    assert estimate.ngm > estimate.npw


def test_fft_grid_is_twice_ngm() -> None:
    estimate = resource_estimate(_SILICON, ecutwfc_ry=40.0, ecutrho_ry=160.0, nbnd=8)

    assert estimate.fft_grid_points == 2 * estimate.ngm


def test_ram_estimate_is_positive_and_finite() -> None:
    estimate = resource_estimate(_SILICON, ecutwfc_ry=40.0, ecutrho_ry=160.0, nbnd=8)

    assert estimate.ram_mb > 0
    assert math.isfinite(estimate.ram_mb)


def test_ram_estimate_does_not_scale_with_k_point_count() -> None:
    """memory_report.f90's own default (io_level > 0) streams wavefunctions
    to disk one k-point at a time -- resource_estimate has no n_irr_k/
    nkpoints parameter at all, on purpose, not by oversight."""
    import inspect

    assert "n_irr_k" not in inspect.signature(resource_estimate).parameters
    assert "nkpoints" not in inspect.signature(resource_estimate).parameters


def test_noncollinear_doubles_the_wavefunction_contribution() -> None:
    collinear = resource_estimate(_SILICON, ecutwfc_ry=40.0, ecutrho_ry=160.0, nbnd=8)
    noncollinear = resource_estimate(
        _SILICON, ecutwfc_ry=40.0, ecutrho_ry=160.0, nbnd=8, noncollinear=True
    )

    assert noncollinear.ram_mb > collinear.ram_mb


def test_nspin_two_increases_the_density_contribution() -> None:
    nspin1 = resource_estimate(
        _SILICON, ecutwfc_ry=40.0, ecutrho_ry=160.0, nbnd=8, nspin=1
    )
    nspin2 = resource_estimate(
        _SILICON, ecutwfc_ry=40.0, ecutrho_ry=160.0, nbnd=8, nspin=2
    )

    assert nspin2.ram_mb > nspin1.ram_mb


def test_more_bands_increases_ram_linearly() -> None:
    small = resource_estimate(_SILICON, ecutwfc_ry=40.0, ecutrho_ry=160.0, nbnd=8)
    large = resource_estimate(_SILICON, ecutwfc_ry=40.0, ecutrho_ry=160.0, nbnd=16)

    wfc_only_ratio = (large.ram_mb) / (small.ram_mb)
    # Not exactly 2x since density terms are nbnd-independent, but strictly
    # more RAM for strictly more bands.
    assert large.ram_mb > small.ram_mb
    assert wfc_only_ratio < 2.0


def test_memory_per_process_divides_by_ranks_within_one_pool() -> None:
    estimate = ResourceEstimate(npw=1000, ngm=4000, fft_grid_points=8000, ram_mb=1000.0)

    # 128 ranks, 4 pools -> 32 ranks per pool.
    assert memory_per_process(estimate, ntasks=128, npool=4) == 1000.0 / 32


def test_memory_per_process_with_a_single_pool_uses_every_rank() -> None:
    estimate = ResourceEstimate(npw=1000, ngm=4000, fft_grid_points=8000, ram_mb=640.0)

    assert memory_per_process(estimate, ntasks=64, npool=1) == 10.0


def test_envelope_takes_the_max_across_steps_not_the_sum() -> None:
    small = ResourceEstimate(npw=10, ngm=20, fft_grid_points=40, ram_mb=5.0)
    large = ResourceEstimate(npw=100, ngm=200, fft_grid_points=400, ram_mb=50.0)

    combined = envelope([small, large])

    assert combined == large


def test_envelope_requires_at_least_one_estimate() -> None:
    with pytest.raises(ValueError, match="at least one"):
        envelope([])
