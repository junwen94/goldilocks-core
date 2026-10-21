from __future__ import annotations

from dataclasses import replace

import pytest

from goldilocks_core.advisors.job_resources import JobDecision
from goldilocks_core.advisors.parallelisation import (
    WARNING_CATALOGUE,
    ParallelisationHumanInput,
    diag_grid_side,
    parallelisation,
)

_JOB = JobDecision(
    partition="scarf",
    nodes=2,
    ntasks=128,
    ntasks_per_node=64,
    walltime_h=24.0,
    max_seconds=int(24 * 3600 * 0.95),
    account=None,
)


def test_npool_prefers_a_divisor_of_both_ntasks_and_n_irr_k() -> None:
    state = parallelisation(_JOB, n_irr_k=32)

    assert state.ok
    assert 128 % state.value.npool == 0
    assert 32 % state.value.npool == 0
    assert state.value.npool == 32


def test_npool_defaults_to_one_without_n_irr_k() -> None:
    state = parallelisation(_JOB)

    assert state.value.npool == 1
    assert state.value.warnings == ()


def test_npool_warns_when_it_cannot_evenly_divide_n_irr_k() -> None:
    # ntasks=128's divisors <= 5 are 1, 2, 4; the largest, 4, does not
    # divide n_irr_k=5 evenly, so some pools get one extra k-point.
    state = parallelisation(_JOB, n_irr_k=5)

    assert state.value.npool == 4
    assert any(
        "one more k-point" in warning.message for warning in state.value.warnings
    )


def test_npool_never_exceeds_n_irr_k_even_if_that_means_less_parallelism() -> None:
    # n_irr_k=1 (Gamma only): the only valid divisor of ntasks that is
    # <= 1 is 1 itself, so npool=1 with no imbalance warning.
    state = parallelisation(_JOB, n_irr_k=1)

    assert state.value.npool == 1
    assert state.value.warnings == ()


@pytest.mark.parametrize(
    ("ntasks", "n_irr_k"),
    [(64, 1), (64, 6), (128, 32), (64, None)],
)
def test_ndiag_is_left_to_pw_x_by_default(ntasks: int, n_irr_k: int | None) -> None:
    """Regression for #133: with ScaLAPACK the default was the largest
    square <= ntasks/npool (-ndiag 64 for a Gamma-only run on one node,
    -ndiag 16 with npool 4), whatever nbnd was, which aborted or hung
    pw.x for small band counts. Unset, pw.x picks it from nbnd."""
    job = replace(_JOB, nodes=ntasks // 64, ntasks=ntasks)

    state = parallelisation(job, n_irr_k=n_irr_k)

    assert state.value.ndiag is None
    assert state.source == "heuristic"
    assert not any("ndiag" in warning.code for warning in state.value.warnings)


def test_nimage_is_always_none() -> None:
    state = parallelisation(_JOB, n_irr_k=32)

    assert state.value.nimage is None


def test_human_npool_override_is_honored() -> None:
    state = parallelisation(_JOB, human=ParallelisationHumanInput(npool=4))

    assert state.value.npool == 4
    assert state.value.warnings == ()
    assert state.source == "human"


def test_human_npool_that_does_not_divide_ntasks_warns() -> None:
    state = parallelisation(_JOB, human=ParallelisationHumanInput(npool=5))

    assert state.value.npool == 5
    assert any("refuse" in warning.message for warning in state.value.warnings)


def test_human_npool_exceeding_n_irr_k_warns_about_real_idle_pools() -> None:
    state = parallelisation(
        _JOB,
        n_irr_k=5,
        human=ParallelisationHumanInput(npool=8),
    )

    assert state.value.npool == 8
    assert any(
        "no k-point to work on" in warning.message for warning in state.value.warnings
    )


def test_human_square_ndiag_is_kept_without_notes() -> None:
    state = parallelisation(_JOB, n_irr_k=32, human=ParallelisationHumanInput(ndiag=16))

    assert state.value.ndiag == 16
    assert state.value.npool == 32
    assert state.value.warnings == ()
    assert state.source == "human"


def test_human_ndiag_that_is_not_a_square_integer_gets_an_info_note() -> None:
    """pw.x never rejects a non-square ndiag; it silently runs the
    largest square grid that fits (#133 corrected the old "may be
    rejected" wording)."""
    state = parallelisation(_JOB, n_irr_k=1, human=ParallelisationHumanInput(ndiag=10))

    assert state.value.ndiag == 10
    (note,) = state.value.warnings
    assert note.code == "parallelisation.ndiag_not_a_square_integer"
    assert note.level == "info"
    assert "ndiag=10" in note.message
    assert "3x3" in note.message


def test_non_square_note_names_the_grid_capped_by_processes_per_pool() -> None:
    # 128 tasks in 32 pools leave 4 processes per pool: a 2x2 grid.
    state = parallelisation(_JOB, n_irr_k=32, human=ParallelisationHumanInput(ndiag=10))

    (note,) = state.value.warnings
    assert "2x2" in note.message


@pytest.mark.parametrize(
    ("ndiag", "procs_per_pool", "side"),
    [
        (10, 128, 3),  # non-square ndiag: the largest square below it
        (64, 4, 2),  # capped by the processes in the pool
        (64, 64, 8),
        (16, 16, 4),
        (15, 64, 3),
        (1, 64, 1),
        (16, 3, 1),
        (4, 0, 1),  # npool > ntasks: LAXlib still uses at least one process
    ],
)
def test_diag_grid_side_matches_laxlib(
    ndiag: int, procs_per_pool: int, side: int
) -> None:
    assert diag_grid_side(ndiag, procs_per_pool) == side


def test_warning_catalogue_describes_only_the_non_square_note_for_ndiag() -> None:
    ndiag_entries = [w for w in WARNING_CATALOGUE if "ndiag" in w.code]

    assert [(w.code, w.level) for w in ndiag_entries] == [
        ("parallelisation.ndiag_not_a_square_integer", "info")
    ]
