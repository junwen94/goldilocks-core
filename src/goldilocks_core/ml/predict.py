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

``predict_k_distance`` (#92) is the same policy for QRF95, which predates
``ML_CLASSIFIER_ROLES`` and keeps its own ``QrfKpointsConfig`` asset shape
(``ml.models.load_default_qrf_config`` -- see that module's docstring for
why); both functions share ``_load_and_predict`` below rather than each
repeating the resolve/import/call/degrade sequence.

``is_magnetic`` needs one more thing ``_load_and_predict`` alone can't
supply: ``goldilocks_ml.inference.load_model`` requires its mMACE backbone
checkpoint passed explicitly as an ``artifacts={"mace_backbone": path}``
override -- confirmed empirically (2026-09-21) against the real published
record (1g8rw-q8128): its ``model.json`` declares no ``requires_artifacts``
entry for it, so there is no automatic PSDI download, and calling
``load_model`` without the override fails outright with "needs artifact:
mace_backbone". ``registry.toml``'s own comment on ``[defaults.is_magnetic]``
already named the intended source for that path -- the same manually
-configured ``GOLDILOCKS_MACE_BACKBONE`` checkpoint
``advisors/magnetic_ordering_ml.py`` uses for ordering ranking, reused
rather than downloaded a second time -- this just wires it in for real.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from goldilocks_core.assets.store import AssetCorrupt, AssetNotInstalled, AssetStore
from goldilocks_core.ml.models import load_default_qrf_config, load_ml_classifier

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import Structure

    from goldilocks_core.assets.records import AssetSpec


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
    or failing checksum verification, goldilocks-ml itself not
    importable, or (``is_magnetic`` only) its mMACE backbone checkpoint
    not configured.
    """
    store = store or AssetStore()
    config = load_ml_classifier(role)
    artifacts = _extra_artifacts(role)
    return _load_and_predict(role, config.asset, structure, store, artifacts=artifacts)


def predict_k_distance(structure: Structure, *, store: AssetStore | None = None) -> Any:
    """Return QRF95's ``ModelPrediction`` (a raw k-distance, 1/angstrom)
    for ``structure`` -- same contract and degrade policy as
    :func:`predict`, kept separate because QRF95's asset is described by
    ``QrfKpointsConfig``, not ``MlClassifierConfig``."""
    store = store or AssetStore()
    config = load_default_qrf_config()
    if config.model_asset is None:
        raise MlModelUnavailable("no k_distance model asset is registered")
    return _load_and_predict("k_distance", config.model_asset, structure, store)


def _extra_artifacts(role: str) -> dict[str, Path] | None:
    """Artifacts a role's model needs beyond its own installed asset
    directory, or ``None`` for a role that needs nothing extra."""
    if role != "is_magnetic":
        return None
    from goldilocks_core.advisors.magnetic_ordering_ml import (
        MagneticOrderingMlUnavailable,
        checkpoint_path,
    )

    try:
        checkpoint = checkpoint_path()
    except MagneticOrderingMlUnavailable as error:
        raise MlModelUnavailable(str(error)) from error
    return {"mace_backbone": checkpoint}


def _load_and_predict(
    label: str,
    asset: AssetSpec,
    structure: Structure,
    store: AssetStore,
    *,
    artifacts: dict[str, Path] | None = None,
) -> Any:
    try:
        installed = store.resolve_spec(asset)
    except (AssetNotInstalled, AssetCorrupt) as error:
        raise MlModelUnavailable(
            f"{label} model asset {asset.id}@{asset.version} is not "
            f"usable: run 'goldilocks assets install {asset.id}'"
        ) from error
    try:
        from goldilocks_ml.inference import load_model
    except ImportError as error:
        raise MlModelUnavailable(
            f"goldilocks-ml is required to run the {label} model: {error}"
        ) from error
    try:
        model = load_model(installed.root, artifacts=artifacts)
        return model.predict(structure)
    except Exception as error:
        raise MlModelUnavailable(f"the {label} model failed to run: {error}") from error
