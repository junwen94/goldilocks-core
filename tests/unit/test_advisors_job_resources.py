from __future__ import annotations

from goldilocks_core.advisors.job_resources import (
    JobHumanInput,
    JobLlmInput,
    job_resources,
)
from goldilocks_core.advisors.size import ResourceEstimate
from goldilocks_core.inputs.hpc import Hardware, HpcProfile, Partition

_SMALL_ESTIMATE = ResourceEstimate(
    npw=1000, ngm=4000, fft_grid_points=8000, ram_mb=500.0
)
_HUGE_ESTIMATE = ResourceEstimate(
    npw=10**6, ngm=4 * 10**6, fft_grid_points=8 * 10**6, ram_mb=50_000_000.0
)


def _profile(*, with_bigmem: bool = False) -> HpcProfile:
    default_hw = Hardware(
        cores_per_node=64, mem_per_node_gb=250, max_nodes=100, max_walltime_h=168
    )
    partitions = {
        "scarf": Partition(name="scarf", hardware=default_hw, default=True),
    }
    if with_bigmem:
        partitions["bigmem"] = Partition(
            name="bigmem",
            hardware=Hardware(
                cores_per_node=64,
                mem_per_node_gb=1024,
                max_nodes=10,
                max_walltime_h=24,
            ),
        )
    return HpcProfile(
        name="test-cluster",
        scheduler="slurm",
        launcher="srun",
        modules={},
        has_scalapack={},
        partitions=partitions,
    )


def test_small_job_stays_on_the_default_partition() -> None:
    state = job_resources(_SMALL_ESTIMATE, _profile())

    assert state.ok
    assert state.value.partition == "scarf"
    assert state.value.nodes == 1
    assert state.value.ntasks == 64


def test_walltime_defaults_to_the_partitions_ceiling_with_a_warning() -> None:
    state = job_resources(_SMALL_ESTIMATE, _profile())

    assert state.value.walltime_h == 168
    assert any("not specified" in warning.message for warning in state.value.warnings)
    assert state.value.max_seconds == int(168 * 3600 * 0.95)


def test_human_walltime_is_used_without_a_warning() -> None:
    state = job_resources(
        _SMALL_ESTIMATE, _profile(), human=JobHumanInput(walltime_h=2.0)
    )

    assert state.value.walltime_h == 2.0
    assert state.value.warnings == ()
    assert state.source == "human"


def test_huge_job_falls_back_to_a_bigger_memory_partition_when_one_exists() -> None:
    state = job_resources(_HUGE_ESTIMATE, _profile(with_bigmem=True))

    assert state.value.partition == "bigmem"
    assert any("using" in warning.message for warning in state.value.warnings)


def test_huge_job_warns_but_does_not_crash_when_no_bigger_partition_exists() -> None:
    state = job_resources(_HUGE_ESTIMATE, _profile(with_bigmem=False))

    assert state.ok
    assert state.value.partition == "scarf"
    assert any("does not fit" in warning.message for warning in state.value.warnings)


def test_human_can_force_a_specific_partition() -> None:
    state = job_resources(
        _SMALL_ESTIMATE,
        _profile(with_bigmem=True),
        human=JobHumanInput(partition="bigmem"),
    )

    assert state.value.partition == "bigmem"
    assert state.value.ntasks_per_node == 64


def test_unknown_partition_name_is_unavailable_not_a_crash() -> None:
    state = job_resources(
        _SMALL_ESTIMATE, _profile(), human=JobHumanInput(partition="does-not-exist")
    )

    assert not state.ok
    assert state.status == "unavailable"


def test_human_nodes_and_ntasks_override_the_heuristic() -> None:
    state = job_resources(
        _SMALL_ESTIMATE, _profile(), human=JobHumanInput(nodes=4, ntasks=200)
    )

    assert state.value.nodes == 4
    assert state.value.ntasks == 200


def test_account_is_never_guessed() -> None:
    state = job_resources(_SMALL_ESTIMATE, _profile())

    assert state.value.account is None


def test_ntasks_alone_derives_a_self_consistent_nodes_count() -> None:
    """Regression for #35 (v2 epic 9, #9): ntasks given with no nodes
    override used to leave nodes at the memory-estimate heuristic value
    regardless, so ntasks-per-node (fixed to the partition's
    cores_per_node) times nodes could be smaller than the requested
    ntasks -- an internally self-contradictory submit.sh."""
    state = job_resources(_SMALL_ESTIMATE, _profile(), human=JobHumanInput(ntasks=200))

    assert state.value.ntasks == 200
    assert state.value.ntasks <= state.value.nodes * state.value.ntasks_per_node


def test_nodes_exceeding_the_partition_ceiling_warns() -> None:
    """Regression for #35 (v2 epic 9, #9): nodes had no upper-bound
    check against the chosen partition's own max_nodes at all."""
    state = job_resources(
        _SMALL_ESTIMATE,
        _profile(with_bigmem=True),
        human=JobHumanInput(partition="bigmem", nodes=50),
    )

    assert state.value.nodes == 50
    assert any(
        warning.code == "job.nodes_exceeds_partition_ceiling"
        for warning in state.value.warnings
    )


def test_walltime_exceeding_the_partition_ceiling_warns() -> None:
    """Regression for #35 (v2 epic 9, #9): walltime_h had no upper
    -bound check against the chosen partition's own max_walltime_h."""
    state = job_resources(
        _SMALL_ESTIMATE, _profile(), human=JobHumanInput(walltime_h=999.0)
    )

    assert state.value.walltime_h == 999.0
    assert any(
        warning.code == "job.walltime_exceeds_partition_ceiling"
        for warning in state.value.warnings
    )


def test_human_account_passes_through() -> None:
    state = job_resources(
        _SMALL_ESTIMATE, _profile(), human=JobHumanInput(account="scd")
    )

    assert state.value.account == "scd"


def test_llm_walltime_used_when_no_human_override() -> None:
    state = job_resources(_SMALL_ESTIMATE, _profile(), llm=JobLlmInput(walltime_h=4.0))

    assert state.value.walltime_h == 4.0
    assert state.source == "llm"
