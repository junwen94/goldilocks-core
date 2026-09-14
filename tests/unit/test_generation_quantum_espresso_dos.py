from __future__ import annotations

import pytest

from goldilocks_core.generation.errors import GenerationError
from goldilocks_core.generation.quantum_espresso.dos import write_qe_dos
from goldilocks_core.step_settings import DosSettings
from goldilocks_core.steps import SharedContext


def _ctx() -> SharedContext:
    return SharedContext(prefix="silicon", outdir="./out", pseudo_dir="./pseudo")


def test_writes_a_minimal_dos_namelist() -> None:
    step = DosSettings(delta_e=0.01, ngauss=0)

    steps = write_qe_dos(step, _ctx())

    assert len(steps) == 1
    dos_step = steps[0]
    assert dos_step.name == "dos"
    assert dos_step.executable == "dos.x"
    assert dos_step.args == ("-in", "dos.in")
    assert dos_step.stdout == "dos.out"
    content = dos_step.files["dos.in"]
    assert "&DOS" in content
    assert "prefix           = 'silicon'" in content
    assert "deltae           = 0.01" in content
    assert "ngauss           = 0" in content
    # Nothing invented: no degauss/emin/emax when none were resolved.
    assert "degauss" not in content
    assert "emin" not in content
    assert "emax" not in content


def test_writes_broadening_and_energy_window_when_resolved() -> None:
    step = DosSettings(delta_e=0.01, ngauss=0, broadening=0.02, emin=-5.0, emax=5.0)

    content = write_qe_dos(step, _ctx())[0].files["dos.in"]

    assert "degauss          = 0.02" in content
    assert "emin             = -5.0" in content
    assert "emax             = 5.0" in content


def test_missing_delta_e_is_a_generation_error_not_a_silent_default() -> None:
    step = DosSettings(delta_e=None, ngauss=0)

    with pytest.raises(GenerationError, match="delta_e"):
        write_qe_dos(step, _ctx())


def test_missing_ngauss_is_a_generation_error() -> None:
    step = DosSettings(delta_e=0.01, ngauss=None)

    with pytest.raises(GenerationError, match="ngauss"):
        write_qe_dos(step, _ctx())


def test_no_parallelisation_args_dos_x_has_no_such_concept_here() -> None:
    step = DosSettings(delta_e=0.01, ngauss=0)

    args = write_qe_dos(step, _ctx())[0].args

    assert "-npool" not in args
