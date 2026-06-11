"""Model loading utilities for machine-learning inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Metadata parsed from a goldilocks-models manifest.json."""

    task: str
    version: str
    model_type: str
    model_file: str
    feature_set: str
    manifest_dir: Path
    extra: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """Loaded model object together with its manifest metadata."""

    model: object
    manifest: ModelManifest


def load_model(manifest_dir: Path) -> LoadedModel:
    """Load a trained model from a goldilocks-models artifact directory.

    The directory must contain ``manifest.json`` and the model file named
    by ``manifest["model_file"]`` (typically ``model.joblib``).

    Args:
        manifest_dir: Path to the versioned artifact directory, e.g.
            ``artifacts/kpoints/1.0/``.
    """
    manifest_path = manifest_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {manifest_dir}")

    raw: dict[str, Any] = json.loads(manifest_path.read_text())

    known_keys = {"task", "version", "model_type", "model_file", "feature_set"}
    manifest = ModelManifest(
        task=str(raw["task"]),
        version=str(raw["version"]),
        model_type=str(raw["model_type"]),
        model_file=str(raw.get("model_file", "model.joblib")),
        feature_set=str(raw.get("feature_set", "cslr")),
        manifest_dir=manifest_dir,
        extra={k: v for k, v in raw.items() if k not in known_keys},
    )

    model_path = manifest_dir / manifest.model_file
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if manifest.model_type not in {"random_forest", "xgboost", "gradient_boosting"}:
        raise NotImplementedError(
            f"Model type {manifest.model_type!r} is not yet supported."
        )

    return LoadedModel(model=joblib.load(model_path), manifest=manifest)


def load_model_from_hf(
    repo_id: str,
    revision: str | None = None,
    cache_dir: Path | None = None,
) -> LoadedModel:
    """Download a model artefact directory from HuggingFace Hub and load it.

    The HuggingFace repository must contain ``manifest.json`` at its root,
    matching the same layout as a local artefact directory.

    Args:
        repo_id: HuggingFace repository ID, e.g.
            ``"goldilocks-models/kpoints-v1.0"``.
        revision: Branch, tag, or commit hash.  Defaults to the repo's
            default branch.
        cache_dir: Local directory for the HuggingFace snapshot cache.
            Defaults to the standard HuggingFace cache location.

    Returns:
        LoadedModel with manifest metadata and loaded model object.

    Raises:
        ImportError: if ``huggingface_hub`` is not installed.
        FileNotFoundError: if ``manifest.json`` is missing in the repo.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download models from HuggingFace. "
            "It should be installed with goldilocks-core. "
            "If missing, run: pip install huggingface_hub"
        ) from exc

    local_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return load_model(Path(local_dir))
