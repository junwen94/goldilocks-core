"""Submission-script rendering + bundle assembly, the last step of the
v2 orchestrator (v2 epic 8, #8).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from goldilocks_core.bundle import BundleInput
from goldilocks_core.generation.files import InputArtifact
from goldilocks_core.inputs.hpc import HpcProfile
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
    artifacts: list[InputArtifact] = []
    for step in steps:
        for path, content in step.files.items():
            artifacts.append(
                {"path": path, "role": "input", "content": content.encode("utf-8")}
            )
    pseudopotentials = advice.system.pseudo.metadata
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
    return BundleInput(
        artifacts=tuple(artifacts), records=advice.records(), citations=citations
    )
