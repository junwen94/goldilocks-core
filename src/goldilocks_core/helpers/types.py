"""Shared type definitions for goldilocks_core."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pymatgen.core import Structure

PathLike = str | Path
StructureInput = Structure | PathLike

CodeName = Literal["quantum_espresso"]
CalcTask = Literal["scf_single_point"]
AccuracyLevel = Literal["low", "standard", "high"]
AdvisorKind = Literal["heuristic", "ml", "external"]
ModelSource = Literal["huggingface", "local"]
ModelType = Literal["random_forest", "cgcnn", "xgboost"]


@dataclass(slots=True)
class KPointsAdvice:
    """Recommended k-point settings for a calculation."""

    code: CodeName
    task: CalcTask
    mesh_type: str
    grid: tuple[int, int, int]
    shift: tuple[int, int, int]
    accuracy_level: AccuracyLevel
    advisor_kind: AdvisorKind
    advisor_name: str


@dataclass(slots=True)
class StructureAnalysis:
    """Summary of structure features relevant to DFT recommendations."""

    contains_transition_metals: bool
    contains_lanthanides: bool
    contains_heavy_elements: bool


@dataclass(slots=True)
class StructureFeatureVector:
    """Named numerical feature vector extracted from a structure.

    This object stores the feature values together with their column names
    so feature ordering remains explicit during inference.
    """

    values: np.ndarray
    feature_names: list[str]


@dataclass(slots=True)
class ModelSpec:
    """Metadata describing a trained model used by the package.

    Parameters
    ----------
    name
        Human-readable model name used inside the package.
    version
        Version label for the model specification.
    model_type
        Model family or backend, such as random_forest, cgcnn, or xgboost.
    target
        Prediction target produced by the model.
    feature_set
        Feature set expected by the model, such as cslr.
    source
        Where the model artifacts are stored, such as huggingface or local.
    location
        Source-specific model location, such as a Hugging Face repository ID
        or a local file path.
    revision
        Optional source revision, such as a branch, tag, or commit hash.
    """

    name: str
    version: str
    model_type: ModelType
    target: str
    feature_set: str
    source: ModelSource
    location: str
    revision: str | None = None
