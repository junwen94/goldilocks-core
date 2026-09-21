from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
from pymatgen.core import Lattice, Species, Structure

from goldilocks_core.advisors.magnetic_ordering_ml import (
    CHECKPOINT_ENV,
    MagneticOrderingMlUnavailable,
    checkpoint_path,
    rank_orderings,
)

_IRON = Structure(Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]])
_IRON_PAIR = Structure(
    Lattice.cubic(2.87), ["Fe", "Fe"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
)


def test_checkpoint_path_requires_the_environment_variable(monkeypatch) -> None:
    monkeypatch.delenv(CHECKPOINT_ENV, raising=False)

    with pytest.raises(MagneticOrderingMlUnavailable, match=CHECKPOINT_ENV):
        checkpoint_path()


def test_checkpoint_path_rejects_a_missing_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(CHECKPOINT_ENV, str(tmp_path / "missing.model"))

    with pytest.raises(MagneticOrderingMlUnavailable, match="does not exist"):
        checkpoint_path()


def test_checkpoint_path_accepts_a_real_file(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "backbone.model"
    checkpoint.write_bytes(b"stub")
    monkeypatch.setenv(CHECKPOINT_ENV, str(checkpoint))

    assert checkpoint_path() == checkpoint


def test_rank_orderings_needs_at_least_one_candidate(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "backbone.model"
    checkpoint.write_bytes(b"stub")
    monkeypatch.setenv(CHECKPOINT_ENV, str(checkpoint))

    with pytest.raises(ValueError, match="at least one candidate"):
        rank_orderings([])


def test_rank_orderings_checks_the_checkpoint_before_anything_else(monkeypatch) -> None:
    """The checkpoint is resolved before any candidate is touched, per the
    docstring's "always before relaxing any candidate, never partway
    through the list" promise -- this failure must not depend on whether
    goldilocks-ml happens to be installed."""
    monkeypatch.delenv(CHECKPOINT_ENV, raising=False)

    with pytest.raises(MagneticOrderingMlUnavailable, match=CHECKPOINT_ENV):
        rank_orderings([("fm", _IRON)])


def _stub_relax_seam(monkeypatch: pytest.MonkeyPatch, fake_relax) -> None:
    """Replace goldilocks-ml's own relax/domain_limits/safe_load, so
    rank_orderings()'s winner-selection logic can be exercised without the
    real mace/e3nn/sphericart stack -- these modules import fine without it
    (only calling into them needs it, see their own docstrings), so
    monkeypatching the real functions directly, not injecting fake modules,
    is enough here."""
    monkeypatch.setattr(
        "goldilocks_ml.models.magnetism._mace_backbone.safe_load",
        lambda checkpoint, device: object(),
    )
    monkeypatch.setattr(
        "goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax."
        "relax.domain_limits",
        lambda raw_model: {},
    )
    monkeypatch.setattr(
        "goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax.relax.relax",
        fake_relax,
    )


def test_rank_orderings_never_recommends_a_candidate_relax_flagged_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Confirmed empirically (2026-09-21) against real Fe bcc candidates: a
    diverged relaxation relax() itself flags INVALID can land at an
    enormously negative energy (observed: -1.27e9 eV/atom) that would win a
    naive ``min()`` over every physically reasonable candidate. An invalid
    result must never become the winner, even though it is still reported
    in ``candidates`` for a caller to see."""
    from goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax.relax import (
        INVALID,
        OK,
        MomentRelaxationResult,
    )

    checkpoint = tmp_path / "backbone.model"
    checkpoint.write_bytes(b"stub")
    monkeypatch.setenv(CHECKPOINT_ENV, str(checkpoint))

    def fake_relax(structure, initial_moments, *, checkpoint, device):
        if len(structure) == 1:
            return MomentRelaxationResult(
                moments=np.zeros((1, 3)), energy_ev=-1_000_000_000.0, status=INVALID
            )
        return MomentRelaxationResult(
            moments=np.zeros((len(structure), 3)),
            energy_ev=-48.0,
            status=OK,
        )

    _stub_relax_seam(monkeypatch, fake_relax)

    result = rank_orderings([("fm", _IRON), ("afm-1", _IRON_PAIR)])

    assert result.winner.label == "afm-1"
    assert result.winner.status == OK
    assert {candidate.label for candidate in result.candidates} == {"fm", "afm-1"}


def test_rank_orderings_degrades_when_every_candidate_is_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax.relax import (
        INVALID,
        MomentRelaxationResult,
    )

    checkpoint = tmp_path / "backbone.model"
    checkpoint.write_bytes(b"stub")
    monkeypatch.setenv(CHECKPOINT_ENV, str(checkpoint))

    def fake_relax(structure, initial_moments, *, checkpoint, device):
        return MomentRelaxationResult(
            moments=np.zeros((len(structure), 3)),
            energy_ev=-1_000_000_000.0,
            status=INVALID,
        )

    _stub_relax_seam(monkeypatch, fake_relax)

    with pytest.raises(MagneticOrderingMlUnavailable, match="invalid"):
        rank_orderings([("fm", _IRON)])


def test_rank_orderings_names_goldilocks_ml_when_it_is_missing(
    monkeypatch, tmp_path
) -> None:
    if importlib.util.find_spec("goldilocks_ml") is not None:
        pytest.skip("goldilocks-ml is installed in this environment")
    checkpoint = tmp_path / "backbone.model"
    checkpoint.write_bytes(b"stub")
    monkeypatch.setenv(CHECKPOINT_ENV, str(checkpoint))

    with pytest.raises(MagneticOrderingMlUnavailable, match="goldilocks-ml"):
        rank_orderings([("fm", _IRON)])


@pytest.mark.skipif(
    not os.environ.get("GOLDILOCKS_MACE_BACKBONE")
    or importlib.util.find_spec("goldilocks_ml") is None
    or importlib.util.find_spec("mace") is None,
    reason=(
        "needs a real mMACE backbone checkpoint (GOLDILOCKS_MACE_BACKBONE) and "
        "goldilocks-ml plus mace/e3nn/sphericart/ase manually installed"
    ),
)
def test_rank_orderings_ranks_antiferromagnetic_nio_below_ferromagnetic() -> None:
    """A coarse sanity check against the real backbone, when it is present.

    Never runs in CI: mirrors goldilocks-ml's own
    ``test_the_real_backbone_end_to_end`` for the same reason -- the
    checkpoint and the mace-torch fork are not installable from PyPI alone
    yet. Verified manually (2026-09-19, re-verified 2026-09-21): NiO's
    antiferromagnetic candidates rank below its ferromagnetic guess,
    matching NiO's real, well-established antiferromagnetic ground state.

    Was briefly broken end to end (not by this codebase): goldilocks-ml
    0.2.0's ``fm_fim_relax.relax`` passes ``use_collinear``/
    ``constrain_magnitude`` to ``MagneticSCFMACE``, but the mace-torch fork
    commit its README documented at the time
    (``ac8ff4764122ced0d57198fe2f9ba170c9fcd16d``) implemented neither
    keyword -- confirmed empirically 2026-09-21, reported as
    stfc/goldilocks-ml#92, fixed by re-pinning the documented commit to
    ``19cdf6692c48e068a24e06cfe1ffc670e8aea3dd`` (stfc/goldilocks-ml#93,
    merged). Re-ran this exact test against the fixed commit the same day:
    genuinely passes again (was an ``xfail(strict=True)`` in between, which
    is what turned this from a silent regression into a required check).
    """
    nio_fm = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(4.17), ["Ni", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    # A hand-built compensated AFM candidate: alternating +/-z spin on the
    # rocksalt Ni sublattice, oxygen unsigned -- the same shape
    # ``magnetic_config._label_by_spin`` produces from a real enumerator
    # candidate, built here directly since this test does not need enumlib.
    ni_up, ni_down = Species("Ni", spin=1), Species("Ni", spin=-1)
    nio_afm = Structure(
        nio_fm.lattice,
        [
            ni_up if site.specie.symbol == "Ni" and index % 2 == 0 else site.specie
            for index, site in enumerate(nio_fm)
        ],
        nio_fm.frac_coords,
    )
    for index, site in enumerate(nio_fm):
        if site.specie.symbol == "Ni" and index % 2 == 1:
            nio_afm.replace(index, ni_down)

    result = rank_orderings([("fm", nio_fm), ("afm-1", nio_afm)])

    assert result.winner.label == "afm-1"
    assert result.winner.energy_per_atom_ev < next(
        c.energy_per_atom_ev for c in result.candidates if c.label == "fm"
    )
