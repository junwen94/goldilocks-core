"""predict: run a registered, installed ML model on one structure (#11, #92).

The one place ``analysis/is_metal.py``, ``analysis/is_magnetic.py`` and
``advisors/k_sampling.py`` reach a model, never ``goldilocks_ml`` directly.
:class:`MlModelUnavailable` means the model is not set up, and it is the only
failure those callers fall back to their next tier on. Anything raised while
loading or running a model that is set up propagates and fails the caller's
request: a model broken by another request's torch state used to pass for a
missing one, and the request silently got a heuristic answer (#130).

is_metal (CGCNN) and k_distance (QRF95, described by ``QrfKpointsConfig``
rather than ``ml.models.ML_CLASSIFIER_ROLES``) are loaded once per installed
asset directory and run in this process. is_magnetic embeds the structure
with mMACE, which changes process-wide torch state, so it runs on a worker
process (``ml/mace_process.py``) and this process never imports mace.

``goldilocks_ml.inference.load_model`` needs is_magnetic's mMACE backbone
passed as ``artifacts={"mace_backbone": path}``: the published record
(1g8rw-q8128) declares no download for it. The path is
``GOLDILOCKS_MACE_BACKBONE``, the checkpoint ordering ranking also uses
(``advisors/magnetic_ordering_ml.checkpoint_path``).
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

from goldilocks_core.assets.store import AssetCorrupt, AssetNotInstalled, AssetStore
from goldilocks_core.ml.models import (
    ML_CLASSIFIER_ROLES,
    load_default_qrf_config,
    load_ml_classifier,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import Structure


class MlModelUnavailable(Exception):
    """A registered model is not set up: its asset is not installed or fails
    verification, or (is_magnetic only) the mMACE backbone is not configured
    or the mace stack is not installed.

    Callers fall back to their next tier on this and on nothing else; a model
    that is set up but fails to load or predict raises its own error."""


def predict(role: str, structure: Structure, *, store: AssetStore | None = None) -> Any:
    """Return the registered ``role`` model's ``ModelPrediction`` for
    ``structure`` (``goldilocks_ml.inference.ModelPrediction``).

    Raises :class:`MlModelUnavailable` if the model is not set up; whatever
    loading or running it raises propagates."""
    root = _installed_root(role, store or AssetStore())
    if role != "is_magnetic":
        return _model(root).predict(structure)
    checkpoint = _mmace_checkpoint()
    from goldilocks_core.ml.mace_process import predict_is_magnetic

    return predict_is_magnetic(root, checkpoint, structure)


def predict_k_distance(structure: Structure, *, store: AssetStore | None = None) -> Any:
    """Return QRF95's ``ModelPrediction`` (a raw k-distance, 1/angstrom) for
    ``structure``, with the same contract as :func:`predict`."""
    return _model(_installed_root("k_distance", store or AssetStore())).predict(
        structure
    )


def model_ready(target: str) -> bool:
    """Whether the model registered for ``target`` (``is_metal``,
    ``is_magnetic`` or ``k_distance``) passes the checks :func:`predict` and
    :func:`predict_k_distance` make before running it; ``False`` for any
    other target. Loads and runs nothing."""
    if target not in (*ML_CLASSIFIER_ROLES, "k_distance"):
        return False
    try:
        _installed_root(target, AssetStore())
        if target == "is_magnetic":
            _mmace_checkpoint()
    except MlModelUnavailable:
        return False
    return True


def _installed_root(target: str, store: AssetStore) -> Path:
    """The verified installed directory of ``target``'s registered model
    asset."""
    asset = (
        load_default_qrf_config().model_asset
        if target == "k_distance"
        else load_ml_classifier(target).asset
    )
    if asset is None:
        raise MlModelUnavailable(f"no {target} model asset is registered")
    try:
        return store.resolve_spec(asset).root
    except (AssetNotInstalled, AssetCorrupt) as error:
        raise MlModelUnavailable(
            f"{target} model asset {asset.id}@{asset.version} is not "
            f"usable: run 'goldilocks assets install {asset.id}'"
        ) from error


def _mmace_checkpoint() -> Path:
    """is_magnetic's mMACE backbone checkpoint, once it is configured and
    mace, e3nn and sphericart are installed."""
    from goldilocks_core.advisors.magnetic_ordering_ml import (
        MagneticOrderingMlUnavailable,
        checkpoint_path,
        require_mace_packages,
    )

    try:
        checkpoint = checkpoint_path()
        require_mace_packages()
    except MagneticOrderingMlUnavailable as error:
        raise MlModelUnavailable(str(error)) from error
    return checkpoint


@cache
def _model(root: Path) -> Any:
    """The model installed at ``root``, loaded on first use and kept for the
    life of the process.

    Its dependencies are read from ``root``, where each deposit ships them
    beside its ``model.json``. goldilocks-ml's default artifact directory is
    relative to the working directory, which a running server cannot rely
    on (#136)."""
    from goldilocks_ml.inference import load_model

    return load_model(root, artifact_directory=root)
