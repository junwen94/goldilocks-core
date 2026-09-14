from __future__ import annotations

import pytest

from goldilocks_core.generation.errors import GenerationError
from goldilocks_core.generation.quantum_espresso.namelists import render_namelist


def test_renders_only_sections_that_received_a_keyword() -> None:
    text = render_namelist(
        {
            "calculation": "scf",
            "ecutwfc": 40.0,
            "conv_thr": 1e-08,
        }
    )

    assert "&CONTROL" in text
    assert "&SYSTEM" in text
    assert "&ELECTRONS" in text
    assert "&IONS" not in text
    assert "&CELL" not in text
    assert "&FCP" not in text
    assert "&RISM" not in text


def test_preserves_indexed_magnetic_keywords_verbatim() -> None:
    text = render_namelist(
        {
            "nspin": 2,
            "starting_magnetization(1)": 0.55,
            "angle1(1)": 90.0,
        }
    )

    assert "starting_magnetization(1) = 0.55" in text
    assert "angle1(1)" in text and "90.0" in text


def test_never_rederives_magnetization_from_atoms_magic() -> None:
    """The exact bug this module exists to avoid: ASE's write_espresso_in
    silently overwrites starting_magnetization from atoms.get_initial_
    magnetic_moments() whenever nspin==2. render_namelist never touches
    Atoms at all, so a resolved non-zero value must survive unchanged."""
    text = render_namelist({"nspin": 2, "starting_magnetization(1)": 0.55})

    assert "starting_magnetization(1) = 0.55" in text
    assert "0.0" not in text


def test_booleans_render_as_qe_fortran_literals() -> None:
    text = render_namelist({"tprnfor": True, "tstress": False})

    assert "tprnfor" in text and ".true." in text
    assert "tstress" in text and ".false." in text


def test_unrecognized_keyword_raises_generation_error_not_silent_drop() -> None:
    with pytest.raises(GenerationError, match="unrecognized"):
        render_namelist({"ecutwfc": 40.0, "totally_bogus_keyword": 1})


def test_empty_keywords_raises_generation_error() -> None:
    with pytest.raises(GenerationError, match="no Quantum ESPRESSO"):
        render_namelist({})
