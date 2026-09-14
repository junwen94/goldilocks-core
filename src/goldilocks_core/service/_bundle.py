"""Submission-script rendering + bundle assembly, the last step of the
v2 orchestrator (v2 epic 8, #8).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from goldilocks_core.bundle import BundleInput
from goldilocks_core.generation.files import InputArtifact
from goldilocks_core.inputs.hpc import HpcProfile
from goldilocks_core.resolution import FieldState
from goldilocks_core.service._advice import Advice
from goldilocks_core.steps import SharedContext, Step
from goldilocks_core.submission.slurm import render_slurm_script


def render_submission(
    advice: Advice,
    hpc: HpcProfile,
    code: str,
    ctx: SharedContext,
    steps: tuple[Step, ...],
) -> str:
    return render_slurm_script(
        hpc, advice.step.resources.job.value, code, ctx, list(steps)
    )


def to_bundle_input(
    advice: Advice,
    steps: tuple[Step, ...],
    submission_script: str,
    ctx: SharedContext,
) -> BundleInput:
    """Assemble one ``BundleInput``: the rendered input file(s), the
    actual pseudopotential file bytes each step's ``ATOMIC_SPECIES``
    card references (read off disk from ``PseudoMetadata.filepath`` --
    neither epic 5 nor epic 7 did this file-assembly step; see
    ``service/_pseudo.py``'s own docstring), the submission script, and
    every advisor decision as a tri-state record."""
    return assemble_bundle_input(
        advice.system.pseudo.metadata, steps, submission_script, ctx, advice.records()
    )


def assemble_bundle_input(
    pseudopotentials: FieldState[object],
    steps: tuple[Step, ...],
    submission_script: str,
    ctx: SharedContext,
    records: dict[str, FieldState[object]],
) -> BundleInput:
    """Shared by ``to_bundle_input`` above and ``_dos.py``'s
    ``to_bundle_input_dos`` -- the actual file-assembly logic never
    differs between a single-step and a ``dos`` task's bundle, only
    which pseudopotential ``FieldState``/records dict feeds it (v2
    epic 9, #9, #28)."""
    artifacts: list[InputArtifact] = []
    for step in steps:
        for path, content in step.files.items():
            artifacts.append(
                {"path": path, "role": "input", "content": content.encode("utf-8")}
            )
    pseudo_dir = PurePosixPath(ctx.pseudo_dir).as_posix()
    for metadata in pseudopotentials.value if pseudopotentials.ok else ():
        content = Path(metadata.filepath).read_bytes()
        artifacts.append(
            {
                "path": f"{pseudo_dir}/{metadata.filename}",
                "role": "pseudopotential",
                "content": content,
            }
        )
    artifacts.append(
        {
            "path": "submit.sh",
            "role": "submission_script",
            "content": submission_script.encode("utf-8"),
        }
    )
    citations = tuple(
        sorted(
            {
                citation
                for metadata in (pseudopotentials.value if pseudopotentials.ok else ())
                if (citation := metadata.pseudo_info.get("citation"))
            }
        )
    )
    return BundleInput(artifacts=tuple(artifacts), records=records, citations=citations)
