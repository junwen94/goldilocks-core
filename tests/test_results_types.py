"""Unit tests for results/types.py dataclasses."""

from __future__ import annotations

import pytest

from goldilocks_core.results.types import (
    RelaxResult,
    SCFResult,
    ValidationReport,
    ValidationWarning,
)


def test_scf_result_instantiation() -> None:
    r = SCFResult(
        converged=True,
        energy_ev=-4200.5,
        fermi_energy_ev=12.3,
        total_magnetization=2.2,
        n_iterations=15,
        warnings=[],
    )
    assert r.converged is True
    assert r.energy_ev == pytest.approx(-4200.5)
    assert r.fermi_energy_ev == pytest.approx(12.3)
    assert r.total_magnetization == pytest.approx(2.2)
    assert r.n_iterations == 15


def test_scf_result_non_magnetic() -> None:
    r = SCFResult(
        converged=True,
        energy_ev=-100.0,
        fermi_energy_ev=None,
        total_magnetization=None,
        n_iterations=8,
        warnings=[],
    )
    assert r.fermi_energy_ev is None
    assert r.total_magnetization is None


def test_relax_result_instantiation() -> None:
    r = RelaxResult(
        converged=True,
        final_structure=None,
        final_energy_ev=-4201.0,
        n_ionic_steps=12,
        warnings=["some warning"],
    )
    assert r.converged is True
    assert r.n_ionic_steps == 12
    assert len(r.warnings) == 1


def test_validation_warning_instantiation() -> None:
    w = ValidationWarning(
        level="error",
        parameter="scf_convergence",
        message="SCF did not converge",
        suggestion="Increase electron_maxstep",
    )
    assert w.level == "error"
    assert w.suggestion is not None


def test_validation_warning_no_suggestion() -> None:
    w = ValidationWarning(level="info", parameter="test", message="ok")
    assert w.suggestion is None


def test_validation_report_instantiation() -> None:
    w = ValidationWarning(level="warning", parameter="mag", message="unexpected mag")
    rep = ValidationReport(passed=["scf_convergence"], warnings=[w], manifest=None)
    assert "scf_convergence" in rep.passed
    assert rep.manifest is None
    assert len(rep.warnings) == 1


def test_results_are_frozen() -> None:
    r = SCFResult(
        converged=True, energy_ev=-100.0, fermi_energy_ev=None,
        total_magnetization=None, n_iterations=5, warnings=[],
    )
    with pytest.raises((AttributeError, TypeError)):
        r.converged = False  # type: ignore[misc]
