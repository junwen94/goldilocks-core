from __future__ import annotations

from goldilocks_core.advisors.job_resources import JobDecision
from goldilocks_core.advisors.parallelisation import (
    ParallelisationHumanInput,
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
    state = parallelisation(_JOB, has_scalapack=False, n_irr_k=32)

    assert state.ok
    assert 128 % state.value.npool == 0
    assert 32 % state.value.npool == 0
    assert state.value.npool == 32


def test_npool_defaults_to_one_without_n_irr_k() -> None:
    state = parallelisation(_JOB, has_scalapack=False)

    assert state.value.npool == 1
    assert state.value.warnings == ()


def test_npool_warns_when_it_cannot_evenly_divide_n_irr_k() -> None:
    # ntasks=128's divisors <= 5 are 1, 2, 4; the largest, 4, does not
    # divide n_irr_k=5 evenly, so some pools get one extra k-point.
    state = parallelisation(_JOB, has_scalapack=False, n_irr_k=5)

    assert state.value.npool == 4
    assert any("one more k-point" in warning for warning in state.value.warnings)


def test_npool_never_exceeds_n_irr_k_even_if_that_means_less_parallelism() -> None:
    # n_irr_k=1 (Gamma only): the only valid divisor of ntasks that is
    # <= 1 is 1 itself, so npool=1 with no imbalance warning.
    state = parallelisation(_JOB, has_scalapack=False, n_irr_k=1)

    assert state.value.npool == 1
    assert state.value.warnings == ()


def test_ndiag_is_none_without_scalapack() -> None:
    state = parallelisation(_JOB, has_scalapack=False, n_irr_k=32)

    assert state.value.ndiag is None


def test_ndiag_is_the_largest_perfect_square_not_exceeding_tasks_per_pool() -> None:
    state = parallelisation(_JOB, has_scalapack=True, n_irr_k=32)

    per_pool = _JOB.ntasks // state.value.npool
    assert state.value.ndiag <= per_pool
    root = int(state.value.ndiag**0.5)
    assert root * root == state.value.ndiag


def test_nimage_is_always_none() -> None:
    state = parallelisation(_JOB, has_scalapack=True, n_irr_k=32)

    assert state.value.nimage is None


def test_human_npool_override_is_honored() -> None:
    state = parallelisation(
        _JOB, has_scalapack=False, human=ParallelisationHumanInput(npool=4)
    )

    assert state.value.npool == 4
    assert state.value.warnings == ()
    assert state.source == "human"


def test_human_npool_that_does_not_divide_ntasks_warns() -> None:
    state = parallelisation(
        _JOB, has_scalapack=False, human=ParallelisationHumanInput(npool=5)
    )

    assert state.value.npool == 5
    assert any("refuse" in warning for warning in state.value.warnings)


def test_human_npool_exceeding_n_irr_k_warns_about_real_idle_pools() -> None:
    state = parallelisation(
        _JOB,
        has_scalapack=False,
        n_irr_k=5,
        human=ParallelisationHumanInput(npool=8),
    )

    assert state.value.npool == 8
    assert any("no k-point to work on" in warning for warning in state.value.warnings)


def test_human_ndiag_override_bypasses_scalapack_gate() -> None:
    state = parallelisation(
        _JOB, has_scalapack=False, human=ParallelisationHumanInput(ndiag=16)
    )

    assert state.value.ndiag == 16
    assert state.source == "human"
