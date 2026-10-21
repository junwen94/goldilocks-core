"""ml.predict: the registered-model seam (#11, #92, #130).

A model that is not set up raises ``MlModelUnavailable``, the only signal
callers fall back to their next tier on. Not set up means its asset is not
installed or fails verification, or, for is_magnetic, the mMACE backbone is
not configured or the mace packages are not installed. A model that is set
up but fails while loading or predicting fails the caller instead: a model
broken by another request's torch state used to look like a missing one,
and the caller silently answered from its heuristic (#130).

None of these tests needs real assets. A stub store resolves every asset to
a temporary directory and goldilocks-ml's ``load_model`` (or the mace worker)
is stubbed. ``tests/integration/test_mace_isolation.py`` runs the real models.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.advisors import magnetic_ordering_ml
from goldilocks_core.advisors.k_sampling import k_sampling
from goldilocks_core.analysis.is_magnetic import is_magnetic
from goldilocks_core.analysis.is_metal import is_metal
from goldilocks_core.assets.records import AssetReference
from goldilocks_core.assets.store import AssetCorrupt, AssetNotInstalled
from goldilocks_core.ml import mace_process, predict as predict_module
from goldilocks_core.ml.predict import (
    MlModelUnavailable,
    model_ready,
    predict,
    predict_k_distance,
)
from goldilocks_core.resolution import Provenance, Resolved

_SILICON = Structure.from_spacegroup(
    "Fd-3m", Lattice.cubic(5.43), ["Si"], [[0.0, 0.0, 0.0]]
)
_IRON = Structure.from_spacegroup(
    "Im-3m", Lattice.cubic(2.87), ["Fe"], [[0.0, 0.0, 0.0]]
)
_MODEL_FAILURE = "mat1 and mat2 must have the same dtype, but got Float and Double"
_IN_PROCESS_PREDICTIONS = [
    pytest.param(
        lambda store: predict("is_metal", _SILICON, store=store), id="is_metal"
    ),
    pytest.param(
        lambda store: predict_k_distance(_SILICON, store=store), id="k_distance"
    ),
]
"""The two models that run in this process, each on a given store."""


class _StubStore:
    """Resolves every asset to one directory, as if installed and verified."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def resolve_spec(self, spec: Any) -> SimpleNamespace:
        return SimpleNamespace(root=self.root)


class _UnusableStore:
    """Fails every asset the way ``AssetStore`` does for a missing or
    corrupt install."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def resolve_spec(self, spec: Any) -> SimpleNamespace:
        raise self.error


class _Model:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def predict(self, structure: Structure) -> SimpleNamespace:
        if self.error is not None:
            raise self.error
        return SimpleNamespace(value=True)


@pytest.fixture(autouse=True)
def fresh_model_cache() -> Iterator[None]:
    """Models are cached per installed directory for the life of the
    process; no test may see another test's stub."""
    predict_module._model.cache_clear()
    yield
    predict_module._model.cache_clear()


@pytest.fixture
def loads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, dict[str, Any]]]:
    """Every ``load_model`` call, each returning a model that predicts."""
    calls: list[tuple[Path, dict[str, Any]]] = []

    def fake_load_model(directory: Path, **kwargs: Any) -> _Model:
        calls.append((directory, kwargs))
        return _Model()

    monkeypatch.setattr("goldilocks_ml.inference.load_model", fake_load_model)
    return calls


@pytest.fixture
def failing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """``load_model`` returns a model whose prediction raises the dtype
    error a float64 default dtype caused (#130)."""
    monkeypatch.setattr(
        "goldilocks_ml.inference.load_model",
        lambda directory, **kwargs: _Model(RuntimeError(_MODEL_FAILURE)),
    )


@pytest.fixture
def backbone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    checkpoint = tmp_path / "backbone.model"
    checkpoint.write_bytes(b"stub")
    monkeypatch.setenv("GOLDILOCKS_MACE_BACKBONE", str(checkpoint))
    return checkpoint


@pytest.fixture
def mace_packages(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """The packages ``find_spec`` reports installed: mace, e3nn and
    sphericart unless a test removes one (CI installs none of them)."""
    installed = {"mace", "e3nn", "sphericart"}
    monkeypatch.setattr(
        magnetic_ordering_ml,
        "find_spec",
        lambda name: ModuleSpec(name, None) if name in installed else None,
    )
    return installed


@pytest.mark.parametrize("run", _IN_PROCESS_PREDICTIONS)
def test_a_model_that_fails_to_run_fails_the_prediction(
    run, tmp_path: Path, failing_model: None
) -> None:
    with pytest.raises(RuntimeError, match=_MODEL_FAILURE):
        run(_StubStore(tmp_path))


@pytest.mark.parametrize("run", _IN_PROCESS_PREDICTIONS)
def test_a_model_loads_once_from_its_installed_directory(
    run, tmp_path: Path, loads: list[tuple[Path, dict[str, Any]]]
) -> None:
    """The artifacts a deposit ships beside its ``model.json`` are read from
    there, never from goldilocks-ml's default directory, which is relative to
    the working directory and can be deleted under a running server."""
    store = _StubStore(tmp_path)

    run(store)
    run(store)

    assert loads == [(tmp_path, {"artifact_directory": tmp_path})]


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            AssetNotInstalled(
                AssetReference("models/is-metal-classifier", "1"), Path("/assets")
            ),
            id="not-installed",
        ),
        pytest.param(AssetCorrupt("installed file changed"), id="corrupt"),
    ],
)
def test_a_model_whose_asset_is_unusable_is_unavailable(
    error: Exception, loads: list[tuple[Path, dict[str, Any]]]
) -> None:
    with pytest.raises(
        MlModelUnavailable, match="goldilocks assets install models/is-metal-classifier"
    ):
        predict("is_metal", _SILICON, store=_UnusableStore(error))

    assert loads == []


def test_model_ready_makes_the_checks_a_prediction_makes_first(
    tmp_path: Path,
    backbone: Path,
    mace_packages: set[str],
    loads: list[tuple[Path, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``capabilities`` lists the ``ml`` approach from this, so it must agree
    with what a prediction would do, without loading or running a model."""

    def no_worker(model_root: Path, checkpoint: Path, structure: Structure) -> None:
        raise AssertionError("model_ready must not run is_magnetic")

    monkeypatch.setattr(mace_process, "predict_is_magnetic", no_worker)
    monkeypatch.setattr(predict_module, "AssetStore", lambda: _StubStore(tmp_path))
    targets = ("is_metal", "k_distance", "is_magnetic", "k_index")

    assert [model_ready(target) for target in targets] == [True, True, True, False]
    mace_packages.discard("sphericart")
    assert [model_ready(target) for target in targets] == [True, True, False, False]
    monkeypatch.setattr(
        predict_module, "AssetStore", lambda: _UnusableStore(AssetCorrupt("changed"))
    )
    assert [model_ready(target) for target in targets] == [False] * 4
    assert loads == []


def test_is_magnetic_runs_on_the_mace_worker_with_the_installed_model_and_backbone(
    tmp_path: Path,
    backbone: Path,
    mace_packages: set[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_predict_is_magnetic(
        model_root: Path, checkpoint: Path, structure: Structure
    ) -> str:
        calls.append((model_root, checkpoint, structure))
        return "a prediction"

    monkeypatch.setattr(mace_process, "predict_is_magnetic", fake_predict_is_magnetic)

    assert predict("is_magnetic", _IRON, store=_StubStore(tmp_path)) == "a prediction"
    assert calls == [(tmp_path, backbone, _IRON)]


def test_is_magnetic_hands_the_worker_an_absolute_backbone_path(
    tmp_path: Path,
    mace_packages: set[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker starts in the system temporary directory, so a relative
    ``GOLDILOCKS_MACE_BACKBONE`` reaches it resolved against the caller's
    working directory."""
    (tmp_path / "backbone.model").write_bytes(b"stub")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOLDILOCKS_MACE_BACKBONE", "backbone.model")
    checkpoints = []

    def fake_predict_is_magnetic(
        model_root: Path, checkpoint: Path, structure: Structure
    ) -> str:
        checkpoints.append(checkpoint)
        return "a prediction"

    monkeypatch.setattr(mace_process, "predict_is_magnetic", fake_predict_is_magnetic)

    predict("is_magnetic", _IRON, store=_StubStore(tmp_path))

    assert checkpoints == [tmp_path / "backbone.model"]


@pytest.mark.parametrize("missing", ["mace", "e3nn", "sphericart"])
def test_is_magnetic_is_unavailable_without_a_mace_package(
    missing: str,
    tmp_path: Path,
    backbone: Path,
    mace_packages: set[str],
    loads: list[tuple[Path, dict[str, Any]]],
) -> None:
    mace_packages.discard(missing)

    with pytest.raises(MlModelUnavailable, match=f"^{missing} not installed"):
        predict("is_magnetic", _IRON, store=_StubStore(tmp_path))


def test_is_magnetic_is_unavailable_without_a_configured_backbone(
    tmp_path: Path, mace_packages: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GOLDILOCKS_MACE_BACKBONE", raising=False)

    with pytest.raises(MlModelUnavailable, match="GOLDILOCKS_MACE_BACKBONE"):
        predict("is_magnetic", _IRON, store=_StubStore(tmp_path))


def test_is_magnetic_is_unavailable_when_the_configured_backbone_is_missing(
    tmp_path: Path, mace_packages: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOLDILOCKS_MACE_BACKBONE", str(tmp_path / "missing.model"))

    with pytest.raises(MlModelUnavailable, match="does not exist"):
        predict("is_magnetic", _IRON, store=_StubStore(tmp_path))


@pytest.mark.parametrize(
    "modules",
    [
        pytest.param({"mace": None}, id="no-mace"),
        pytest.param(
            {
                name: ModuleType(name)
                for name in (
                    "mace",
                    "mace.calculators",
                    "mace.calculators.mace",
                    "mace.modules",
                )
            },
            id="stock-mace",
        ),
    ],
)
def test_the_mace_worker_reports_a_mace_without_the_fork_as_unavailable(
    modules: dict[str, ModuleType | None],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stock PyPI mace-torch passes ``find_spec`` but lacks the fork's
    magnetic classes goldilocks-ml imports. The worker reports that, like no
    mace at all, as unavailable, so is_magnetic falls back instead of failing
    every request."""
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    with pytest.raises(MlModelUnavailable, match="mace-torch fork"):
        mace_process._predict_is_magnetic(tmp_path, tmp_path / "backbone.model", _IRON)


def test_is_magnetic_falls_back_when_the_worker_cannot_import_mace(
    tmp_path: Path,
    backbone: Path,
    mace_packages: set[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(model_root: Path, checkpoint: Path, structure: Structure) -> None:
        raise MlModelUnavailable("is_magnetic needs the mace-torch fork")

    monkeypatch.setattr(predict_module, "AssetStore", lambda: _StubStore(tmp_path))
    monkeypatch.setattr(mace_process, "predict_is_magnetic", unavailable)

    assert is_magnetic(_IRON).source == "heuristic"


@pytest.mark.parametrize(
    "decide",
    [
        pytest.param(is_metal, id="is_metal"),
        pytest.param(
            lambda structure: k_sampling(
                Resolved("metal", Provenance(source="human")), structure
            ),
            id="k_sampling",
        ),
        pytest.param(is_magnetic, id="is_magnetic"),
    ],
)
def test_a_failing_model_fails_its_caller_instead_of_a_heuristic_answer(
    decide,
    tmp_path: Path,
    failing_model: None,
    backbone: Path,
    mace_packages: set[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_worker(
        model_root: Path, checkpoint: Path, structure: Structure
    ) -> None:
        raise RuntimeError(_MODEL_FAILURE)

    monkeypatch.setattr(predict_module, "AssetStore", lambda: _StubStore(tmp_path))
    monkeypatch.setattr(mace_process, "predict_is_magnetic", failing_worker)

    with pytest.raises(RuntimeError, match=_MODEL_FAILURE):
        decide(_IRON)
