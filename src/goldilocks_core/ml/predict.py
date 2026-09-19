"""predict: call a registered, installed ML model for one structure.

v2 epic 11 (#11). ``capabilities.py``'s own ``_approaches`` docstring
already reserved this moment: "epic 11 only has to change this one
function's body, not any caller." This module is the other half of that
promise -- the one place ``analysis/is_metal.py``/``analysis/is_magnetic.py``
reach for a real prediction, never ``goldilocks_ml`` directly, so every
caller degrades to its own heuristic tier identically whenever a model is
not installed, corrupt, or goldilocks-ml itself is not importable. Same
policy ``advisors/magnetic_ordering_ml.py`` already established for
mMACE, generalized to the two standalone classifiers registered in
``ml/registry.toml`` (``ml.models.ML_CLASSIFIER_ROLES``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from goldilocks_core.assets.store import AssetCorrupt, AssetNotInstalled, AssetStore
from goldilocks_core.ml.models import load_ml_classifier

if TYPE_CHECKING:
    from pymatgen.core import Structure


class MlModelUnavailable(Exception):
    """A registered model could not be loaded or run.

    Never a reason to fail a caller's own request: ``is_metal()`` and
    friends catch this and fall back to their own heuristic tier, the
    same degrade-not-raise policy every advisor in this codebase already
    follows for a missing external dependency."""


def predict(role: str, structure: Structure, *, store: AssetStore | None = None) -> Any:
    """Return the registered ``role`` model's ``ModelPrediction`` for
    ``structure`` (``goldilocks_ml.inference.ModelPrediction``, not
    imported at module scope -- see the module docstring on why
    goldilocks-ml is only ever reached for here, lazily).

    Raises :class:`MlModelUnavailable` for every failure mode short of
    the prediction genuinely succeeding: the model asset not installed
    or failing checksum verification, or the ``models``/``magnetism``
    extra not installed.
    """
    store = store or AssetStore()
    config = load_ml_classifier(role)
    try:
        installed = store.resolve_spec(config.asset)
    except (AssetNotInstalled, AssetCorrupt) as error:
        raise MlModelUnavailable(
            f"{role} model asset {config.asset.id}@{config.asset.version} is not "
            f"usable: run 'goldilocks assets install {config.asset.id}'"
        ) from error
    try:
        from goldilocks_ml.inference import load_model
    except ImportError as error:
        raise MlModelUnavailable(
            f"goldilocks-ml is required to run the {role} model: {error}"
        ) from error
    try:
        model = load_model(installed.root)
        return model.predict(structure)
    except Exception as error:
        raise MlModelUnavailable(f"the {role} model failed to run: {error}") from error
