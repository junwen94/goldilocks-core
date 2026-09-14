from __future__ import annotations

from goldilocks_core.advisors.relax import RelaxOptions, VcRelaxOptions


def test_relax_options_default_to_qes_own_documented_values() -> None:
    options = RelaxOptions()

    assert options.ion_dynamics == "bfgs"
    assert options.forc_conv_thr == 1.0e-3
    assert options.nstep == 50
    assert options.trust_radius_max == 0.8
    assert options.trust_radius_min == 1.0e-3
    assert options.trust_radius_ini == 0.5
    assert options.remove_rigid_rot is False


def test_vc_relax_extends_relax_with_cell_specific_fields() -> None:
    options = VcRelaxOptions()

    assert options.ion_dynamics == "bfgs"  # inherited
    assert options.cell_dofree == "all"
    assert options.press == 0.0
    assert options.press_conv_thr == 0.5
    assert options.cell_factor == 2.0


def test_cell_dynamics_is_always_derived_from_ion_dynamics() -> None:
    """Audit finding B5: QE requires cell_dynamics == ion_dynamics for
    vc-relax -- a derived property means the invalid combination cannot
    be constructed, rather than being caught only by QE at runtime."""
    options = VcRelaxOptions(ion_dynamics="bfgs")

    assert options.cell_dynamics == "bfgs"


def test_relax_options_are_frozen() -> None:
    options = RelaxOptions()

    try:
        options.ion_dynamics = "damp"  # type: ignore[misc]
        raised = False
    except AttributeError:
        raised = True
    assert raised
