"""magnetic_ordering_ml: pick a magnetic ordering by mMACE-relaxed energy.

New (v2, #87). ``magnetic_config.py``'s ``enumerate_magnetic_orderings``
lists candidate magnetic orderings (a plain ferromagnetic guess, plus
whatever compensated antiferromagnetic candidates enumlib finds) but never
picks a winner by energy -- finding the true ground state needs comparing
total energies, which that module's own docstring assigns to "an
agent-level job" rather than its stateless heuristic tier. This module is
that job, made real: relax every candidate's magnetic moments on a frozen
mMACE potential energy surface
(``goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax.relax``,
the same SCF-like moment relaxation a real DFT study would use total
energies from) and rank them by energy per atom.

Verified against a real published checkpoint (2026-09-19): for NiO, this
correctly ranks antiferromagnetic candidates below the ferromagnetic guess,
matching NiO's real, well-established antiferromagnetic ground state.

Needs goldilocks-ml's ``magnetism`` extra (``mace``, ``e3nn``,
``sphericart``, ``ase``) and an mMACE backbone checkpoint, configured via
``GOLDILOCKS_MACE_BACKBONE`` rather than an ``AssetStore``-managed
download: the ``mace-torch`` fork this depends on has no PyPI release
either (see goldilocks-ml's own
``deposits/magnetism/is_magnetic/mace_mlp/VENDORING_TODO.md``), so
automating only the checkpoint's distribution would not make this feature
installable end-to-end on its own. Neither is required at import time of
*this* file -- only calling :func:`rank_orderings` touches them, so a
missing one surfaces as :class:`MagneticOrderingMlUnavailable` naming what
is missing, not a bare ``ImportError`` -- the same pattern
``goldilocks_ml.models.magnetism._mace_backbone`` itself uses.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymatgen.core import Structure

CHECKPOINT_ENV = "GOLDILOCKS_MACE_BACKBONE"


class MagneticOrderingMlUnavailable(Exception):
    """The mMACE-based ordering comparison could not run.

    Never a reason to fail a caller's own request: a caller comparing
    candidates by energy should catch this and fall back to reporting the
    candidates unranked, the same way ``magnetic_config.py``'s own heuristic
    tier degrades on a missing dependency rather than raising."""


@dataclass(frozen=True, slots=True)
class OrderingCandidate:
    """One candidate's relaxed energy, for reporting alongside the winner."""

    label: str
    structure: Structure
    energy_per_atom_ev: float
    status: str


@dataclass(frozen=True, slots=True)
class OrderingMlResult:
    winner: OrderingCandidate
    candidates: tuple[OrderingCandidate, ...]
    model_id: str


def checkpoint_path() -> Path:
    """Resolve the configured mMACE backbone checkpoint, or say why it
    is not usable."""
    raw = os.environ.get(CHECKPOINT_ENV)
    if not raw:
        raise MagneticOrderingMlUnavailable(
            f"set {CHECKPOINT_ENV} to the mMACE backbone checkpoint path to "
            "enable energy-based magnetic ordering selection"
        )
    path = Path(raw)
    if not path.is_file():
        raise MagneticOrderingMlUnavailable(f"{CHECKPOINT_ENV}={path} does not exist")
    return path


def _seed_moments(candidate: Structure, limits: dict[str, float]) -> np.ndarray:
    """Signed initial moments for one candidate: magnitude from the same
    oxidation-state heuristic for every candidate (so the energy comparison
    is apples to apples), sign from whatever spin decoration the candidate
    already carries -- all-positive for the plain ferromagnetic guess,
    alternating for an enumerator-produced antiferromagnetic candidate, per
    ``magnetic_config._label_by_spin``'s own convention."""
    from goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax import (
        seed_moments as seed_moments_module,
    )

    magnitudes = seed_moments_module.seed_moments_fm_fim(
        candidate, max_moment_by_symbol=limits
    )
    for index, site in enumerate(candidate):
        spin = getattr(site.specie, "spin", None) or 0.0
        if spin < 0:
            magnitudes[index, 2] *= -1.0
    return magnitudes


def rank_orderings(
    candidates: Sequence[tuple[str, Structure]], *, device: str = "cpu"
) -> OrderingMlResult:
    """Relax every candidate and rank by energy per atom, lowest first.

    ``candidates`` is ``(label, structure)`` pairs, typically
    ``magnetic_config.enumerate_magnetic_orderings``'s own output. Each
    candidate's cell size may differ (a compensated antiferromagnetic
    ordering can need a larger -- or, if the input cell was redundant,
    smaller -- cell than another candidate), so candidates are compared by
    energy *per atom*, never by total energy.

    Raises :class:`MagneticOrderingMlUnavailable` if the ``magnetism``
    extra or the checkpoint is not available -- always before relaxing any
    candidate, never partway through the list.
    """
    if not candidates:
        raise ValueError("rank_orderings needs at least one candidate")
    checkpoint = checkpoint_path()
    try:
        from goldilocks_ml.models.magnetism._mace_backbone import safe_load
        from goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax.relax import (
            domain_limits,
            relax,
        )
    except ImportError as error:
        raise MagneticOrderingMlUnavailable(
            "goldilocks-ml with the magnetism extra is required for "
            f"energy-based magnetic ordering selection: {error}"
        ) from error

    raw_model = safe_load(checkpoint, device)
    limits = domain_limits(raw_model)

    results = []
    for label, candidate in candidates:
        initial_moments = _seed_moments(candidate, limits)
        relaxed = relax(
            candidate, initial_moments, checkpoint=checkpoint, device=device
        )
        results.append(
            OrderingCandidate(
                label=label,
                structure=candidate,
                energy_per_atom_ev=relaxed.energy_ev / len(candidate),
                status=relaxed.status,
            )
        )
    winner = min(results, key=lambda result: result.energy_per_atom_ev)
    return OrderingMlResult(
        winner=winner, candidates=tuple(results), model_id=checkpoint.name
    )
