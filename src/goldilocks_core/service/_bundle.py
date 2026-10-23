"""Submission-script rendering + bundle assembly, the last step of the
v2 orchestrator (v2 epic 8, #8).
"""

from __future__ import annotations

from pathlib import PurePosixPath

from goldilocks_core.bundle import BundleInput
from goldilocks_core.generation.files import InputArtifact
from goldilocks_core.inputs.hpc import HpcProfile
from goldilocks_core.resolution import FieldState
from goldilocks_core.service._advice import Advice
from goldilocks_core.service._pseudo import PseudoAdvice, pseudo_artifacts
from goldilocks_core.steps import SharedContext, Step
from goldilocks_core.submission.slurm import render_slurm_script


def render_submission(
    advice: Advice,
    hpc: HpcProfile,
    code: str,
    steps: tuple[Step, ...],
) -> str:
    return render_slurm_script(hpc, advice.step.resources.job.value, code, list(steps))


def to_bundle_input(
    advice: Advice,
    steps: tuple[Step, ...],
    submission_script: str,
    ctx: SharedContext,
) -> BundleInput:
    """Assemble one ``BundleInput``: the rendered input file(s), the
    pseudopotential files each step's ``ATOMIC_SPECIES`` card references
    with their table's licence (``service._pseudo.pseudo_artifacts``), the
    submission script, every advisor decision as a tri-state record, and
    the advice's warnings."""
    return assemble_bundle_input(
        advice.system.pseudo,
        steps,
        submission_script,
        ctx,
        advice.records(),
        advice.warnings(),
    )


def assemble_bundle_input(
    pseudo: PseudoAdvice,
    steps: tuple[Step, ...],
    submission_script: str,
    ctx: SharedContext,
    records: dict[str, FieldState[object]],
    warnings: list[dict[str, object]],
) -> BundleInput:
    """Shared by ``to_bundle_input`` above and the ``dos``/``relax``
    tasks' ``to_bundle_input_dos``/``to_bundle_input_relax`` -- the
    file-assembly logic never differs between tasks, only which
    ``PseudoAdvice``/records dict feeds it (v2 epic 9, #9, #28).
    ``warnings`` is the advice's own ``warnings()``, the list every
    transport returns, which ``goldilocks.json`` carries unchanged (#152).
    The UPFs and their table's licence come from ``pseudo_artifacts``,
    which raises ``AssetCorrupt`` for a file changed since its table was
    verified (#157, #177)."""
    artifacts: list[InputArtifact] = []
    for step in steps:
        for path, content in step.files.items():
            artifacts.append(
                {"path": path, "role": "input", "content": content.encode("utf-8")}
            )
    artifacts.extend(pseudo_artifacts(pseudo, PurePosixPath(ctx.pseudo_dir).as_posix()))
    artifacts.append(
        {
            "path": "submit.sh",
            "role": "submission_script",
            "content": submission_script.encode("utf-8"),
        }
    )
    return BundleInput(
        artifacts=tuple(artifacts), records=records, warnings=tuple(warnings)
    )
