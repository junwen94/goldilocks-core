"""Tests for results/check.py — validation logic."""

from __future__ import annotations

import json

from goldilocks_core.results.check import validate
from goldilocks_core.results.types import RelaxResult, SCFResult, ValidationReport


def _scf(converged=True, n_iter=10, total_mag=None):
    return SCFResult(
        converged=converged,
        energy_ev=-4200.0,
        fermi_energy_ev=12.0,
        total_magnetization=total_mag,
        n_iterations=n_iter,
        warnings=[],
    )


def _relax(converged=True):
    return RelaxResult(
        converged=converged,
        final_structure=None,
        final_energy_ev=-4201.0,
        n_ionic_steps=5,
        warnings=[],
    )


class TestSCFValidation:
    def test_converged_scf_passes(self) -> None:
        rep = validate(_scf())
        assert "scf_convergence" in rep.passed
        assert not any(w.level == "error" for w in rep.warnings)

    def test_not_converged_is_error(self) -> None:
        rep = validate(_scf(converged=False))
        errors = [w for w in rep.warnings if w.level == "error"]
        assert any("convergence" in w.parameter for w in errors)

    def test_high_iterations_warning(self) -> None:
        rep = validate(_scf(n_iter=85), max_iterations=100)
        warn_params = [w.parameter for w in rep.warnings]
        assert "scf_iterations" in warn_params

    def test_low_iterations_no_warning(self) -> None:
        rep = validate(_scf(n_iter=20), max_iterations=100)
        assert not any(w.parameter == "scf_iterations" for w in rep.warnings)


class TestManifestChecks:
    def _make_manifest(self, spin_treatment: str) -> dict:
        return {
            "parameters": {
                "spin": {"treatment": spin_treatment, "provenance": "heuristic"}
            }
        }

    def test_unexpected_magnetisation_warns(self, tmp_path) -> None:
        manifest = self._make_manifest("non_magnetic")
        mp = tmp_path / "goldilocks_manifest.json"
        mp.write_text(json.dumps(manifest))

        rep = validate(_scf(total_mag=1.5), manifest_path=mp)
        assert any(w.parameter == "magnetism_consistency" for w in rep.warnings)

    def test_expected_non_magnetic_passes(self, tmp_path) -> None:
        manifest = self._make_manifest("non_magnetic")
        mp = tmp_path / "goldilocks_manifest.json"
        mp.write_text(json.dumps(manifest))

        rep = validate(_scf(total_mag=0.01), manifest_path=mp)
        assert "magnetism_consistency" in rep.passed

    def test_no_manifest_skips_manifest_checks(self) -> None:
        rep = validate(_scf())
        # Without a manifest, magnetism_consistency is not checked
        assert "magnetism_consistency" not in rep.passed


class TestRelaxValidation:
    def test_converged_relax_passes(self) -> None:
        rep = validate(_relax())
        assert "relax_convergence" in rep.passed

    def test_not_converged_relax_is_error(self) -> None:
        rep = validate(_relax(converged=False))
        errors = [w for w in rep.warnings if w.level == "error"]
        assert any("relax" in w.parameter for w in errors)


class TestReturnType:
    def test_returns_validation_report(self) -> None:
        rep = validate(_scf())
        assert isinstance(rep, ValidationReport)

    def test_manifest_none_when_not_provided(self) -> None:
        rep = validate(_scf())
        assert rep.manifest is None

    def test_manifest_loaded_when_provided(self, tmp_path) -> None:
        mp = tmp_path / "goldilocks_manifest.json"
        mp.write_text(json.dumps({"goldilocks_version": "0.1.0"}))
        rep = validate(_scf(), manifest_path=mp)
        assert rep.manifest is not None
        assert rep.manifest["goldilocks_version"] == "0.1.0"
