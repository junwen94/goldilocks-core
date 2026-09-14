"""job_resources: partition, nodes, ntasks, walltime -- from a
resource_estimate envelope and an HPC profile.

New in v2 (v2 epic 7, #7); no v1 precedent. Part of the false
-circularity fix goldilocks-core-design.md:3176-3195 describes:
``resource_estimate`` (``advisors/size.py``) carries zero parallel
information, so this and ``advisors/parallelisation.py`` can each read
it (plus the shared ``human``/``llm`` input) independently, with no
mutual feedback between them -- ``job_resources`` does not read
``parallelisation``'s output, and ``parallelisation`` only reads this
module's ``ntasks``, never the reverse.

**`partition` is resolved with the machine, upstream of nodes/ntasks,
not a peer of them** (goldilocks-core-design.md:1561-1581): choosing an
HPC profile fixes the default partition; the heuristic only moves off
it when the estimated memory envelope will not fit even at the
profile's node ceiling, at which point it looks for a bigger-memory
partition in the same profile (and only warns, does not silently
guess, if none exists -- SCARF's own profile today has no such
"bigmem"-style escape hatch to fall back to).

**`nodes`/`ntasks` are a rough, heuristic-tier sizing, not a
precision-critical one.** ``resource_estimate``'s ``ram_mb`` is treated
as the calculation's aggregate memory footprint across however many
MPI processes it ends up running on (not a strictly single-process
figure) -- `nodes = ceil(ram_mb / (mem_per_node_gb * 1024)) *
safety_factor`, then `ntasks = nodes * cores_per_node`. The
design doc is explicit that the piece which actually needs ml here is
`walltime`, not this.

**`walltime` is anchored at the partition's own ceiling, not a
middling guess** (goldilocks-core-design.md:1337-1394): `size` gives
*relative* workload, not absolute time, and no machine-specific
throughput constant has ever been calibrated -- guessing short kills
the job (the whole run is wasted); guessing long only costs queue time.
So the heuristic default is `max_walltime_h` itself, with a warning,
never a "conservative-looking" smaller number. `max_seconds` (95% of
the requested walltime, matching aiida-quantumespresso's own
`max_seconds` convention) is meant to give the code a chance to
checkpoint gracefully before SLURM kills it outright.

`npool`/`ndiag`/`nimage` are not decided here -- that is
`advisors/parallelisation.py`'s job, reading this module's `ntasks` as
an input. `account` has no ml/heuristic tier at all: it is personal to
whoever is submitting, not a property of the machine or the
calculation, so it is human-only, and left `None` (not guessed) if
never supplied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from goldilocks_core.advisors.size import ResourceEstimate
from goldilocks_core.inputs.hpc import Hardware, HpcProfile, Partition
from goldilocks_core.inputs.overrides import HumanInput, LlmInput
from goldilocks_core.resolution import FieldState, Provenance, Resolved, Unavailable

_NODE_SAFETY_FACTOR = 1.2
_MAX_SECONDS_FRACTION = 0.95
_BYTES_PER_GB_IN_MB = 1024


@dataclass(frozen=True, slots=True)
class JobDecision:
    partition: str
    nodes: int
    ntasks: int
    ntasks_per_node: int
    walltime_h: float
    max_seconds: int
    account: str | None
    warnings: tuple[str, ...] = ()


class JobHumanInput(HumanInput):
    partition: str | None = None
    nodes: int | None = None
    ntasks: int | None = None
    walltime_h: float | None = None
    account: str | None = None


class JobLlmInput(LlmInput):
    partition: str | None = None
    walltime_h: float | None = None


def job_resources(
    estimate: ResourceEstimate,
    hpc: HpcProfile,
    human: JobHumanInput | None = None,
    llm: JobLlmInput | None = None,
) -> FieldState[JobDecision]:
    human = human or JobHumanInput()
    llm = llm or JobLlmInput()

    requested_partition = human.partition or llm.partition
    if requested_partition is not None and requested_partition not in hpc.partitions:
        available = ", ".join(sorted(hpc.partitions))
        return Unavailable(
            reason=f"hpc profile {hpc.name!r} has no partition "
            f"{requested_partition!r}; available: {available}"
        )

    partition, warnings = _pick_partition(estimate, hpc, human, llm)

    nodes = human.nodes if human.nodes is not None else _nodes_needed(
        estimate, partition.hardware
    )
    ntasks_per_node = partition.hardware.cores_per_node
    ntasks = human.ntasks if human.ntasks is not None else nodes * ntasks_per_node

    walltime_h = _pick_walltime(partition, human, llm, warnings)
    max_seconds = int(walltime_h * 3600 * _MAX_SECONDS_FRACTION)

    decision = JobDecision(
        partition=partition.name,
        nodes=nodes,
        ntasks=ntasks,
        ntasks_per_node=ntasks_per_node,
        walltime_h=walltime_h,
        max_seconds=max_seconds,
        account=human.account,
        warnings=tuple(warnings),
    )
    source = "human" if _any_set(human) else "llm" if _any_set(llm) else "heuristic"
    return Resolved(decision, Provenance(source=source))


def _pick_partition(
    estimate: ResourceEstimate,
    hpc: HpcProfile,
    human: JobHumanInput,
    llm: JobLlmInput,
) -> tuple[Partition, list[str]]:
    name = human.partition or llm.partition
    if name is not None:
        return hpc.partition(name), []

    default = hpc.default_partition()
    needed_gb = estimate.ram_mb / _BYTES_PER_GB_IN_MB
    if needed_gb <= default.hardware.mem_per_node_gb * default.hardware.max_nodes:
        return default, []

    bigger = [
        partition
        for partition in hpc.partitions.values()
        if partition.hardware.mem_per_node_gb > default.hardware.mem_per_node_gb
    ]
    if bigger:
        chosen = max(bigger, key=lambda p: p.hardware.mem_per_node_gb)
        return chosen, [
            f"default partition {default.name!r} cannot fit the estimated "
            f"{needed_gb:.0f} GB even at its node ceiling; using "
            f"{chosen.name!r} instead."
        ]
    return default, [
        f"estimated {needed_gb:.0f} GB does not fit in partition "
        f"{default.name!r} even at its node ceiling, and profile "
        f"{hpc.name!r} has no larger-memory partition to fall back to."
    ]


def _nodes_needed(estimate: ResourceEstimate, hardware: Hardware) -> int:
    mem_per_node_mb = hardware.mem_per_node_gb * _BYTES_PER_GB_IN_MB
    bare = estimate.ram_mb / mem_per_node_mb
    return max(1, math.ceil(bare * _NODE_SAFETY_FACTOR))


def _pick_walltime(
    partition: Partition, human: JobHumanInput, llm: JobLlmInput, warnings: list[str]
) -> float:
    if human.walltime_h is not None:
        return human.walltime_h
    if llm.walltime_h is not None:
        return llm.walltime_h
    warnings.append(
        f"walltime not specified; requesting partition {partition.name!r}'s "
        f"ceiling ({partition.hardware.max_walltime_h}h). This slows queueing "
        "-- schedulers favour short jobs -- but guessing short risks the job "
        "being killed before it finishes, wasting the entire run. If you "
        "know roughly how long this will take, set walltime_h explicitly."
    )
    return partition.hardware.max_walltime_h


def _any_set(overrides: HumanInput) -> bool:
    return bool(overrides.model_dump(exclude_defaults=True))
