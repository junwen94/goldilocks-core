"""Inference utilities for machine-learning models."""

from __future__ import annotations

from goldilocks_core.helpers.types import StructureFeatureVector


def predict(model: object, features: StructureFeatureVector) -> float:
    """Run model inference on a structure feature vector.

    Parameters
    ----------
    model
        Loaded model object.
    features
        Structure-derived feature vector used for prediction.

    Returns
    -------
    float
        Predicted scalar output from the model.
    """
    raise NotImplementedError("Model inference is not implemented yet.")
