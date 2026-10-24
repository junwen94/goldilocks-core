from __future__ import annotations

from goldilocks_core.plan import PlannedStep, expand_task


def test_scf_single_point_expands_to_exactly_one_step() -> None:
    steps = expand_task("scf_single_point")

    assert steps == (PlannedStep(purpose="scf", program="pw.x"),)


def test_dos_expands_to_scf_then_nscf_then_dos_in_order() -> None:
    steps = expand_task("dos")

    assert steps == (
        PlannedStep(purpose="scf", program="pw.x"),
        PlannedStep(purpose="nscf", program="pw.x"),
        PlannedStep(purpose="dos", program="dos.x"),
    )


def test_relax_expands_to_exactly_one_step() -> None:
    steps = expand_task("relax")

    assert steps == (PlannedStep(purpose="relax", program="pw.x"),)


def test_vc_relax_expands_to_exactly_one_step() -> None:
    steps = expand_task("vc-relax")

    assert steps == (PlannedStep(purpose="vc-relax", program="pw.x"),)
