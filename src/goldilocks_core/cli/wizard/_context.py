"""Shared state threaded through wizard steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pymatgen.core import Structure

from goldilocks_core.analyse.structure import StructureAnalysis


@dataclass
class WizardContext:
    """Carries user choices and analysis results between wizard steps."""

    structure_path: Path
    structure: Structure
    analysis: StructureAnalysis
    task: str = "scf"
    accuracy: str = "balanced"
    code: str = "qe"
    hints: dict[str, Any] = field(default_factory=dict)
