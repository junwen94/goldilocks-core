"""Model loading utilities for machine-learning inference."""

from __future__ import annotations

from goldilocks_core.helpers.types import ModelSpec


def load_model(spec: ModelSpec) -> object:
    """Load a trained model from a model specification.

    Parameters
    ----------
    spec
        Metadata describing the model source, type, target, and feature set.

    Returns
    -------
    object
        Loaded model object. The concrete type depends on the model backend.
    """
    raise NotImplementedError("Model loading is not implemented yet.")
