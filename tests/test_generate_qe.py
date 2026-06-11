"""Golden-file tests for generate/qe.py.

Each test runs the full advise pipeline → write_qe_inputs() and asserts that
key QE namelist parameters are present in the generated file.  No external
data required — pseudos come from the bundled PseudoDojo subset.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.advise.pipeline import advise
from goldilocks_core.analyse.structure import analyze_structure
from goldilocks_core.generate.qe import write_qe_inputs
from goldilocks_core.intent import CalculationIntent, ParameterHints
from goldilocks_core.select.qe import build_qe_parameter_set

_SR_FAMILY = "PseudoDojo/0.4/PBEsol/SR/standard/upf"


def _run_pipeline(
    structure: Structure,
    task: str = "scf",
    hints: dict | None = None,
    accuracy: str = "balanced",
) -> tuple:
    analysis = analyze_structure(structure)
    intent = CalculationIntent(
        structure=structure,
        task=task,
        pseudo_family=_SR_FAMILY,
        accuracy=accuracy,
        hints=ParameterHints.from_dict(hints or {}),
    )
    params = build_qe_parameter_set(advise(analysis, intent))
    return params, structure, intent


def _input_text(
    tmp_path: Path,
    structure: Structure,
    task: str = "scf",
    hints: dict | None = None,
    accuracy: str = "balanced",
) -> str:
    params, structure, intent = _run_pipeline(structure, task, hints, accuracy)
    result = write_qe_inputs(params, structure, intent, output_dir=tmp_path)
    return result["input_file"].read_text()


# ---------------------------------------------------------------------------
# BCC Fe — collinear spin
# ---------------------------------------------------------------------------

@pytest.fixture
def fe_bcc() -> Structure:
    return Structure(
        lattice=Lattice.cubic(2.87),
        species=["Fe"],
        coords=[[0, 0, 0]],
    )


def test_fe_nspin2(tmp_path: Path, fe_bcc: Structure) -> None:
    text = _input_text(tmp_path, fe_bcc)
    # ASE writes "nspin            = 2" with variable spacing
    assert any("nspin" in line and "2" in line for line in text.splitlines())


def test_fe_starting_magnetization_present(tmp_path: Path, fe_bcc: Structure) -> None:
    text = _input_text(tmp_path, fe_bcc)
    assert "starting_magnetization" in text


def test_fe_no_noncolin(tmp_path: Path, fe_bcc: Structure) -> None:
    text = _input_text(tmp_path, fe_bcc)
    # collinear Fe should not set noncolin=.true.
    assert "noncolin = .true." not in text and "noncolin=.true." not in text


def test_fe_smearing_present(tmp_path: Path, fe_bcc: Structure) -> None:
    text = _input_text(tmp_path, fe_bcc)
    assert "smearing" in text
    assert "degauss" in text


# ---------------------------------------------------------------------------
# Si diamond — non-magnetic insulator
# ---------------------------------------------------------------------------

@pytest.fixture
def si_diamond() -> Structure:
    return Structure(
        lattice=Lattice.cubic(5.43),
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )


def test_si_nspin1(tmp_path: Path, si_diamond: Structure) -> None:
    text = _input_text(tmp_path, si_diamond)
    # nspin=1 is the QE default; ASE may omit it, but smearing should be absent
    assert "nspin = 2" not in text and "nspin=2" not in text


def test_si_no_starting_magnetization(tmp_path: Path, si_diamond: Structure) -> None:
    text = _input_text(tmp_path, si_diamond)
    assert "starting_magnetization" not in text


def test_si_no_spin_polarization(tmp_path: Path, si_diamond: Structure) -> None:
    text = _input_text(tmp_path, si_diamond)
    # Si has no magnetic elements → no spin polarization, no starting_magnetization
    assert not any("nspin" in line and "2" in line for line in text.splitlines())
    assert "starting_magnetization" not in text


# ---------------------------------------------------------------------------
# Fe with vdW hint
# ---------------------------------------------------------------------------

def test_fe_vdw_hint_writes_vdw_corr(tmp_path: Path, fe_bcc: Structure) -> None:
    text = _input_text(tmp_path, fe_bcc, hints={"use_vdw": True, "vdw_method": "d3bj"})
    assert "vdw_corr" in text
    assert "grimme-d3bj" in text


def test_fe_no_vdw_by_default(tmp_path: Path, fe_bcc: Structure) -> None:
    text = _input_text(tmp_path, fe_bcc)
    assert "vdw_corr" not in text


# ---------------------------------------------------------------------------
# Collinear hint on non-magnetic structure
# ---------------------------------------------------------------------------

def test_si_collinear_hint_writes_nspin2(tmp_path: Path, si_diamond: Structure) -> None:
    text = _input_text(tmp_path, si_diamond, hints={"spin_treatment": "collinear"})
    assert any("nspin" in line and "2" in line for line in text.splitlines())


# ---------------------------------------------------------------------------
# Phonon calculation — both pw.x and ph.x files written
# ---------------------------------------------------------------------------

def test_si_phonon_writes_ph_file(tmp_path: Path, si_diamond: Structure) -> None:
    from goldilocks_core.generate.qe import write_ph_inputs

    params, structure, intent = _run_pipeline(si_diamond, task="ph")
    result = write_qe_inputs(params, structure, intent, output_dir=tmp_path)
    ph_result = write_ph_inputs(output_dir=tmp_path, nq=(2, 2, 2))

    assert result["input_file"].exists()
    assert ph_result["ph_file"].exists()
    ph_text = ph_result["ph_file"].read_text()
    assert "nq1" in ph_text or "&inputph" in ph_text.lower() or "ph.x" in ph_text.lower() or "nq" in ph_text


def test_si_phonon_scf_has_tight_conv_thr(tmp_path: Path, si_diamond: Structure) -> None:
    params, structure, intent = _run_pipeline(si_diamond, task="ph")
    result = write_qe_inputs(params, structure, intent, output_dir=tmp_path, conv_thr=1e-10)
    text = result["input_file"].read_text()
    assert "conv_thr" in text
    assert "1e-10" in text or "1.0e-10" in text or "1E-10" in text


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def test_writes_manifest_json(tmp_path: Path, fe_bcc: Structure) -> None:
    import json

    from goldilocks_core.analyse.structure import analyze_structure

    analysis = analyze_structure(fe_bcc)
    params, structure, intent = _run_pipeline(fe_bcc)
    result = write_qe_inputs(
        params, structure, intent, output_dir=tmp_path, analysis=analysis
    )
    manifest_file = result["manifest_file"]
    assert manifest_file.exists()
    doc = json.loads(manifest_file.read_text())
    assert "goldilocks_version" in doc
    assert "generated_at" in doc
    assert doc["intent"]["task"] == "scf"
    assert doc["structure"]["formula"] == "Fe"
    assert doc["analysis"]["metallicity"] == "likely_metallic"
    assert doc["parameters"]["spin"]["treatment"] == "collinear"
    assert doc["parameters"]["kpoints"]["grid"] != [0, 0, 0]
