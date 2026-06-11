"""Tests for results/local/qe.py — QE output file parsing."""

from __future__ import annotations

import pytest

from goldilocks_core.results.local.qe import parse_relax, parse_scf

_SAMPLE_SCF_CONVERGED = """\
     Self-consistent Calculation

     iteration #  1     ewald energy      =    -100.0
     iteration #  5     ewald energy      =    -200.0
     convergence has been achieved in   5 iterations

     !    total energy              =    -308.76543210 Ry

     The Fermi energy is    12.3456 ev

     total magnetization       =     2.20 Bohr mag/cell
"""

_SAMPLE_SCF_NOT_CONVERGED = """\
     Self-consistent Calculation

     iteration #  1     ewald energy      =    -100.0
     convergence NOT achieved

     !    total energy              =    -100.12345678 Ry
"""

_SAMPLE_SCF_INSULATOR = """\
     Self-consistent Calculation

     iteration #  1     ewald energy      =    -100.0
     convergence has been achieved in   8 iterations

     !    total energy              =    -150.00000000 Ry
"""

_SAMPLE_RELAX_CONVERGED = """\
     BFGS Geometry Optimization

     BFGS step number   1
     convergence has been achieved in   7 iterations
     !    total energy              =    -308.10000000 Ry

     BFGS step number   3
     convergence has been achieved in   6 iterations
     !    total energy              =    -308.76543210 Ry

     End of BFGS Geometry Optimization
"""

_RY_TO_EV = 13.605693122994


def _write_file(tmp_path, content: str, name: str = "pw.out"):
    p = tmp_path / name
    p.write_text(content)
    return p


class TestParseScf:
    def test_converged_true(self, tmp_path) -> None:
        f = _write_file(tmp_path, _SAMPLE_SCF_CONVERGED)
        result = parse_scf(f)
        assert result.converged is True

    def test_n_iterations(self, tmp_path) -> None:
        f = _write_file(tmp_path, _SAMPLE_SCF_CONVERGED)
        result = parse_scf(f)
        assert result.n_iterations == 5

    def test_energy_ev(self, tmp_path) -> None:
        f = _write_file(tmp_path, _SAMPLE_SCF_CONVERGED)
        result = parse_scf(f)
        assert result.energy_ev == pytest.approx(-308.76543210 * _RY_TO_EV)

    def test_fermi_energy(self, tmp_path) -> None:
        f = _write_file(tmp_path, _SAMPLE_SCF_CONVERGED)
        result = parse_scf(f)
        assert result.fermi_energy_ev == pytest.approx(12.3456)

    def test_total_magnetization(self, tmp_path) -> None:
        f = _write_file(tmp_path, _SAMPLE_SCF_CONVERGED)
        result = parse_scf(f)
        assert result.total_magnetization == pytest.approx(2.20)

    def test_not_converged(self, tmp_path) -> None:
        f = _write_file(tmp_path, _SAMPLE_SCF_NOT_CONVERGED)
        result = parse_scf(f)
        assert result.converged is False

    def test_no_fermi_for_insulator(self, tmp_path) -> None:
        f = _write_file(tmp_path, _SAMPLE_SCF_INSULATOR)
        result = parse_scf(f)
        assert result.fermi_energy_ev is None
        assert result.total_magnetization is None


class TestParseRelax:
    def test_converged(self, tmp_path) -> None:
        f = _write_file(tmp_path, _SAMPLE_RELAX_CONVERGED)
        result = parse_relax(f)
        assert result.converged is True

    def test_final_energy(self, tmp_path) -> None:
        f = _write_file(tmp_path, _SAMPLE_RELAX_CONVERGED)
        result = parse_relax(f)
        # Last energy in the file
        assert result.final_energy_ev == pytest.approx(-308.76543210 * _RY_TO_EV)

    def test_ionic_steps(self, tmp_path) -> None:
        f = _write_file(tmp_path, _SAMPLE_RELAX_CONVERGED)
        result = parse_relax(f)
        assert result.n_ionic_steps == 3
