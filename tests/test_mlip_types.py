"""Tests for mlip/types.py — MLIPPreview can be imported without mace-torch."""

from __future__ import annotations

import pytest

from goldilocks_core.mlip.types import MLIPPreview


def test_mlip_preview_relax_only() -> None:
    preview = MLIPPreview(
        relaxed_structure=None,
        final_energy_ev=-4200.5,
        phonon_stable=None,
        imaginary_frequencies=[],
        model_used="mace-mp-medium",
        tasks_run=["relax"],
        warnings=[],
    )
    assert preview.final_energy_ev == pytest.approx(-4200.5)
    assert preview.phonon_stable is None
    assert "relax" in preview.tasks_run


def test_mlip_preview_with_phonon() -> None:
    preview = MLIPPreview(
        relaxed_structure=None,
        final_energy_ev=-4200.5,
        phonon_stable=False,
        imaginary_frequencies=[-0.5, -1.2],
        model_used="mace-mp-medium",
        tasks_run=["relax", "phonon"],
        warnings=["Phonon instability detected: 2 imaginary mode(s)"],
    )
    assert preview.phonon_stable is False
    assert len(preview.imaginary_frequencies) == 2
    assert len(preview.warnings) == 1


def test_mlip_preview_defaults() -> None:
    preview = MLIPPreview(
        relaxed_structure=None,
        final_energy_ev=None,
        phonon_stable=None,
    )
    assert preview.model_used == ""
    assert preview.tasks_run == []
    assert preview.warnings == []
    assert preview.imaginary_frequencies == []


def test_mlip_preview_is_frozen() -> None:
    preview = MLIPPreview(
        relaxed_structure=None,
        final_energy_ev=None,
        phonon_stable=None,
    )
    with pytest.raises((AttributeError, TypeError)):
        preview.phonon_stable = True  # type: ignore[misc]


def test_relax_missing_raises_import_error() -> None:
    """relax_structure raises ImportError when mace-torch is absent (or just importable)."""
    try:
        import mace  # noqa: F401
        pytest.skip("mace-torch is installed; skip graceful-degradation test")
    except ImportError:
        pass

    with pytest.raises(ImportError, match="mace-torch"):
        from pymatgen.core import Lattice, Structure

        from goldilocks_core.mlip.relax import relax_structure

        relax_structure(
            Structure(Lattice.cubic(3.0), ["Si"], [[0, 0, 0]])
        )
