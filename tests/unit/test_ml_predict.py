"""ml.predict: the registered-model call seam, and its is_magnetic-only
backbone-artifact wiring.

Confirmed empirically (2026-09-21) against the real published is_magnetic
record (1g8rw-q8128): ``goldilocks_ml.inference.load_model`` requires its
mMACE backbone passed explicitly as ``artifacts={"mace_backbone": path}``
-- there is no automatic PSDI download for it. These tests exercise that
wiring at the seam (``goldilocks_ml.inference.load_model`` stubbed out),
not the real mace stack -- see
``tests/unit/test_advisors_magnetic_ordering_ml.py`` for the
skipif-guarded tests against a real checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.ml.predict import MlModelUnavailable, predict

pytestmark = pytest.mark.usefixtures("real_assets")

_IRON = Structure(Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]])


class _FakeModel:
    def __init__(self) -> None:
        self.structures: list[Structure] = []

    def predict(self, structure: Structure) -> str:
        self.structures.append(structure)
        return "a prediction"


def _stub_load_model(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    def fake_load_model(
        directory: Path, *, artifacts: Any = None, **_: Any
    ) -> _FakeModel:
        captured["directory"] = directory
        captured["artifacts"] = artifacts
        return _FakeModel()

    monkeypatch.setattr("goldilocks_ml.inference.load_model", fake_load_model)


def test_is_magnetic_passes_the_configured_backbone_as_an_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "backbone.model"
    checkpoint.write_bytes(b"stub")
    monkeypatch.setenv("GOLDILOCKS_MACE_BACKBONE", str(checkpoint))
    captured: dict[str, Any] = {}
    _stub_load_model(monkeypatch, captured)

    result = predict("is_magnetic", _IRON)

    assert result == "a prediction"
    assert captured["artifacts"] == {"mace_backbone": checkpoint}


def test_is_magnetic_degrades_when_the_backbone_is_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOLDILOCKS_MACE_BACKBONE", raising=False)

    with pytest.raises(MlModelUnavailable, match="GOLDILOCKS_MACE_BACKBONE"):
        predict("is_magnetic", _IRON)


def test_is_magnetic_degrades_when_the_configured_backbone_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GOLDILOCKS_MACE_BACKBONE", str(tmp_path / "missing.model"))

    with pytest.raises(MlModelUnavailable, match="does not exist"):
        predict("is_magnetic", _IRON)


def test_is_metal_needs_no_backbone_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only is_magnetic needs the shared mMACE checkpoint -- is_metal's own
    CGCNN classifier must not be affected by whether it is configured."""
    monkeypatch.delenv("GOLDILOCKS_MACE_BACKBONE", raising=False)
    captured: dict[str, Any] = {}
    _stub_load_model(monkeypatch, captured)

    result = predict("is_metal", _IRON)

    assert result == "a prediction"
    assert captured["artifacts"] is None
