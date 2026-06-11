import json

import joblib
import pytest

from goldilocks_core.ml.models import LoadedModel, ModelManifest, load_model


def _write_manifest(tmp_path, model_type: str = "random_forest") -> None:
    manifest = {
        "task": "kpoints",
        "version": "1.0",
        "model_type": model_type,
        "model_file": "model.joblib",
        "feature_set": "cslr",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))


def test_load_model_returns_loaded_model(tmp_path) -> None:
    """Load a random-forest model from a manifest directory."""
    dummy = {"kind": "dummy-rf"}
    joblib.dump(dummy, tmp_path / "model.joblib")
    _write_manifest(tmp_path)

    result = load_model(tmp_path)

    assert isinstance(result, LoadedModel)
    assert result.model == dummy
    assert isinstance(result.manifest, ModelManifest)
    assert result.manifest.task == "kpoints"
    assert result.manifest.version == "1.0"
    assert result.manifest.feature_set == "cslr"


def test_load_model_raises_for_missing_manifest(tmp_path) -> None:
    """Raise FileNotFoundError when manifest.json is absent."""
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        load_model(tmp_path)


def test_load_model_raises_for_missing_model_file(tmp_path) -> None:
    """Raise FileNotFoundError when the model file named in the manifest is absent."""
    _write_manifest(tmp_path)

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_model(tmp_path)


def test_load_model_rejects_unsupported_model_type(tmp_path) -> None:
    """Raise NotImplementedError for an unsupported model_type in the manifest."""
    joblib.dump({}, tmp_path / "model.joblib")
    _write_manifest(tmp_path, model_type="cgcnn")

    with pytest.raises(NotImplementedError, match="cgcnn"):
        load_model(tmp_path)


def test_load_model_reads_extra_manifest_fields(tmp_path) -> None:
    """Extra manifest keys are stored in ModelManifest.extra."""
    manifest = {
        "task": "kpoints",
        "version": "2.0",
        "model_type": "random_forest",
        "model_file": "model.joblib",
        "feature_set": "cslr",
        "training_set": "mp-2024",
        "metrics": {"mae": 0.5},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    joblib.dump({}, tmp_path / "model.joblib")

    result = load_model(tmp_path)

    assert result.manifest.extra["training_set"] == "mp-2024"
    assert result.manifest.extra["metrics"]["mae"] == 0.5
