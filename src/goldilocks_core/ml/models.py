from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, cast, get_args

from goldilocks_core.assets.records import AssetFile, AssetSpec
from goldilocks_core.serialization import Portable, portable_record, to_portable
from goldilocks_core.types import JsonDict, ModelSource, ModelType, PathLike


@dataclass(slots=True)
class ModelSpec:
    name: str
    version: str
    model_type: ModelType
    target: str
    feature_set: str
    source: ModelSource
    location: Annotated[str, Portable()]
    revision: str | None = None
    licence: str | None = None
    licence_text: Annotated[str | None, Portable()] = None
    citation: str | None = None


@to_portable.register(ModelSpec)
def _model_spec_portable(spec: ModelSpec) -> JsonDict:
    return portable_record(spec, ModelSpec)


MODEL_REGISTRY_ENV = "GOLDILOCKS_MODEL_REGISTRY"
_REGISTRY_RESOURCE = "registry.toml"
_VALID_MODEL_SOURCES = frozenset(get_args(ModelSource))
_VALID_MODEL_TYPES = frozenset(get_args(ModelType))


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    id: str
    role: str
    spec: ModelSpec


@dataclass(frozen=True, slots=True)
class QrfKpointsConfig:
    model: ModelSpec
    model_asset: AssetSpec | None
    model_file: str
    confidence: float
    correction: float
    metallicity_model: ModelSpec
    metallicity_asset: AssetSpec | None
    metallicity_checkpoint_file: str
    metallicity_atom_init_file: str


def load_default_qrf_config(path: PathLike | None = None) -> QrfKpointsConfig:
    registry_path = path or os.environ.get(MODEL_REGISTRY_ENV)
    if registry_path is None:
        registry = resources.files("goldilocks_core.ml").joinpath(_REGISTRY_RESOURCE)
        with registry.open("rb") as registry_file:
            data = tomllib.load(registry_file)
    else:
        with Path(registry_path).open("rb") as registry_file:
            data = tomllib.load(registry_file)

    kpoints = data["defaults"]["kpoints"]
    metallicity = kpoints["metallicity"]
    calibration = kpoints["calibration"]
    model_asset = _asset_spec(kpoints.get("asset"))
    metallicity_asset = _asset_spec(metallicity.get("asset"))
    model_file = (
        _role_path(model_asset, "model") if model_asset else kpoints["location"]
    )
    checkpoint_file = metallicity["checkpoint_file"]
    atom_init_file = metallicity["atom_init_file"]

    return QrfKpointsConfig(
        model=_model_spec(kpoints, model_file),
        model_asset=model_asset,
        model_file=model_file,
        confidence=kpoints["interval_confidence"],
        correction=calibration["correction"],
        metallicity_model=_model_spec(metallicity, checkpoint_file),
        metallicity_asset=metallicity_asset,
        metallicity_checkpoint_file=checkpoint_file,
        metallicity_atom_init_file=atom_init_file,
    )


def registered_models(path: PathLike | None = None) -> tuple[RegisteredModel, ...]:
    config = load_default_qrf_config(path)
    return (
        RegisteredModel(
            id=config.model_asset.id if config.model_asset else config.model.name,
            role="k_point_advisor",
            spec=config.model,
        ),
        RegisteredModel(
            id=(
                config.metallicity_asset.id
                if config.metallicity_asset
                else config.metallicity_model.name
            ),
            role="metallicity_classifier",
            spec=config.metallicity_model,
        ),
    )


def model_asset_specs(path: PathLike | None = None) -> tuple[AssetSpec, ...]:
    config = load_default_qrf_config(path)
    return tuple(
        spec
        for spec in (config.model_asset, config.metallicity_asset)
        if spec is not None
    )


def _asset_spec(data: dict[str, Any] | None) -> AssetSpec | None:
    if data is None:
        return None
    files = tuple(
        AssetFile(
            role=file["role"],
            path=file["path"],
            url=file["url"],
            checksum=file.get("checksum"),
            size=file.get("size"),
        )
        for file in data["files"]
    )
    invalid_checksums = [
        file.role
        for file in files
        if file.checksum is not None and not file.checksum.startswith("sha256:")
    ]
    if invalid_checksums:
        raise ValueError(
            f"model asset {data['id']!r} checksums must use sha256: "
            + ", ".join(invalid_checksums)
        )
    if "licence" not in {file.role for file in files}:
        raise ValueError(f"model asset {data['id']!r} must include licence material")
    return AssetSpec(
        id=data["id"],
        version=str(data["version"]),
        files=files,
        preparation_revision=str(data.get("preparation_revision", "1")),
    )


def _model_spec(data: dict[str, Any], location: str) -> ModelSpec:
    identity = {
        "name": data["name"],
        "version": str(data["version"]),
        "target": data["target"],
        "feature_set": data["feature_set"],
    }
    for field, value in identity.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"model {field} must be a non-empty string; got {value!r}")
    revision = data.get("revision")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError(
            f"model revision must be a non-empty string, or absent; got {revision!r}"
        )

    source_value = data.get("source", "local")
    if source_value not in _VALID_MODEL_SOURCES:
        valid = ", ".join(sorted(_VALID_MODEL_SOURCES))
        raise ValueError(f"model source must be one of {valid}; got {source_value!r}")
    source = cast(ModelSource, source_value)
    model_type_value = data["model_type"]
    if model_type_value not in _VALID_MODEL_TYPES:
        valid = ", ".join(sorted(_VALID_MODEL_TYPES))
        raise ValueError(f"model type must be one of {valid}; got {model_type_value!r}")
    model_type = cast(ModelType, model_type_value)
    optional_material = {
        field: data.get(field) for field in ("licence", "licence_text", "citation")
    }
    for field, value in optional_material.items():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(
                f"model {field} must be a non-empty string, or absent; got {value!r}"
            )

    return ModelSpec(
        name=identity["name"],
        version=identity["version"],
        model_type=model_type,
        target=identity["target"],
        feature_set=identity["feature_set"],
        source=source,
        location=data.get("location", location),
        revision=revision,
        **optional_material,
    )


def _role_path(spec: AssetSpec, role: str) -> str:
    try:
        return next(file.path for file in spec.files if file.role == role)
    except StopIteration as error:
        raise ValueError(
            f"asset {spec.id}@{spec.version} lacks required role {role!r}"
        ) from error
