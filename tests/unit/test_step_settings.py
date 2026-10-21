from __future__ import annotations

import dataclasses

import pytest

from goldilocks_core.advisors.convergence import ConvergenceDecision
from goldilocks_core.advisors.k_sampling import KSamplingDecision
from goldilocks_core.advisors.nbnd import NbndDecision
from goldilocks_core.advisors.occupations import OccupationsDecision
from goldilocks_core.step_settings import DosSettings, PhSettings, PwSettings


def _pw_settings() -> PwSettings:
    return PwSettings(
        occupations=OccupationsDecision(
            occupations="smearing", smearing_type="cold", degauss=0.01
        ),
        k_sampling=KSamplingDecision(mesh=(4, 4, 4), shift=(0, 0, 0), k_distance=0.15),
        n_irr_k=10,
        nbnd=NbndDecision(nbnd=9),
        convergence=ConvergenceDecision(
            conv_thr=2e-9,
            etot_conv_thr=1e-4,
            mixing_beta=0.4,
            electron_maxstep=80,
            mixing_mode="plain",
            mixing_fixed_ns=None,
        ),
    )


def test_pw_settings_holds_every_per_step_advisors_decision() -> None:
    settings = _pw_settings()

    assert settings.occupations is not None
    assert settings.occupations.occupations == "smearing"
    assert settings.k_sampling is not None
    assert settings.k_sampling.mesh == (4, 4, 4)
    assert settings.n_irr_k == 10
    assert settings.nbnd is not None
    assert settings.nbnd.nbnd == 9
    assert settings.convergence is not None
    assert settings.convergence.conv_thr == 2e-9


def test_pw_settings_has_no_producer_fields_default_to_none() -> None:
    settings = _pw_settings()

    assert settings.parallel is None
    assert settings.relax is None


@pytest.mark.parametrize("settings", [PwSettings, DosSettings, PhSettings])
@pytest.mark.parametrize("field", ["size", "mem_per_proc"])
def test_no_step_settings_carry_a_memory_estimate(settings: type, field: str) -> None:
    """#148: no writer reads a memory estimate; the job record carries
    pw.x's per-process and total memory instead."""
    assert field not in {item.name for item in dataclasses.fields(settings)}
    with pytest.raises(TypeError, match=field):
        settings(**{field: None})


@pytest.mark.parametrize("settings", [PwSettings, DosSettings, PhSettings])
def test_no_step_settings_carry_disk_io(settings: type) -> None:
    """#129: disk_io='none' would leave pw.x no status file to check
    (punch.f90:60-63), so no settings can request it."""
    assert "disk_io" not in {field.name for field in dataclasses.fields(settings)}
    with pytest.raises(TypeError, match="disk_io"):
        settings(disk_io="none")


def test_constructing_pw_settings_with_no_arguments_is_allowed() -> None:
    """Nothing is required yet -- every field is reserved shape, not a
    forced fabrication, until its producer exists."""
    settings = PwSettings()

    assert settings.occupations is None
    assert settings.nbnd is None


def test_dos_settings_has_its_own_broadening_field_not_degauss() -> None:
    """The deferred v2 epic 2 physics test: pw.x's occupation-broadening
    degauss and dos.x's DOS-histogram broadening are physically unrelated
    numbers QE happens to spell the same way -- DosSettings existing as
    its own type, with its own field name, is what prevents one value
    from silently doing double duty across the two programs."""
    dos = DosSettings(emin=-5.0, emax=5.0, delta_e=0.01, broadening=0.02, ngauss=0)

    assert dos.broadening == 0.02
    assert not hasattr(dos, "degauss")
    assert not hasattr(dos, "occupations")
    assert not hasattr(dos, "k_sampling")
    assert not hasattr(dos, "nbnd")
    assert not hasattr(dos, "convergence")


def test_pw_settings_has_no_dos_specific_fields() -> None:
    pw = _pw_settings()

    assert not hasattr(pw, "broadening")
    assert not hasattr(pw, "ngauss")
    assert not hasattr(pw, "emin")


def test_ph_settings_is_its_own_distinct_type_too() -> None:
    ph = PhSettings(tr2_ph=1e-14, epsil=True)

    assert ph.tr2_ph == 1e-14
    assert ph.epsil is True
    assert not hasattr(ph, "occupations")
    assert not hasattr(ph, "broadening")


def test_step_settings_variants_are_distinguishable_by_isinstance() -> None:
    variants: list[object] = [_pw_settings(), DosSettings(), PhSettings()]

    kinds = [
        "pw"
        if isinstance(v, PwSettings)
        else "dos"
        if isinstance(v, DosSettings)
        else "ph"
        for v in variants
    ]

    assert kinds == ["pw", "dos", "ph"]
