"""Bundled data assets: pseudopotentials and ML model artefacts."""

from __future__ import annotations

from pathlib import Path

_DATA_DIR = Path(__file__).parent


def pseudo_dir(family: str) -> Path:
    """Return the bundled pseudopotential directory for a family label.

    The family label (e.g. ``"PseudoDojo/0.4/PBEsol/FR/standard/upf"``)
    maps directly to a sub-path under the bundled pseudopotentials directory.

    Returns:
        Path to the family directory if it exists and contains UPF files.

    Raises:
        FileNotFoundError: if no bundled UPF files are found for this family.
    """
    path = _DATA_DIR / "pseudopotentials" / family
    if not path.is_dir() or not any(path.glob("*.upf")):
        raise FileNotFoundError(
            f"No bundled pseudopotentials found for family {family!r}. "
            f"Expected UPF files at: {path}"
        )
    return path


def model_dir(task: str, version: str) -> Path:
    """Return the bundled ML model directory for a task and version.

    Args:
        task: Model task name, e.g. ``"kpoints"``.
        version: Version string, e.g. ``"1.0"``.

    Returns:
        Path to the model directory if it exists.

    Raises:
        FileNotFoundError: if no bundled model artefacts are found.
    """
    path = _DATA_DIR / "models" / task / version
    if not path.is_dir() or not (path / "manifest.json").exists():
        raise FileNotFoundError(
            f"No bundled model found for task={task!r} version={version!r}. "
            f"Expected manifest.json at: {path}"
        )
    return path


def available_pseudo_families() -> list[str]:
    """Return family labels for all bundled pseudopotential sets."""
    pseudo_root = _DATA_DIR / "pseudopotentials"
    families = []
    for upf_dir in sorted(pseudo_root.rglob("*.upf")):
        rel = upf_dir.parent.relative_to(pseudo_root)
        label = rel.as_posix()
        if label not in families:
            families.append(label)
    return families
