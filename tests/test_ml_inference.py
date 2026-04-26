from pymatgen.core import Lattice, Structure

from goldilocks_core.kpoints.features import extract_cslr_features
from goldilocks_core.ml.inference import predict


class DummyModel:
    """Minimal sklearn-like model for testing inference flow."""

    def predict(self, X):
        return [float(X.shape[1])]


def test_predict_runs_on_cslr_features() -> None:
    """Run end-to-end prediction from structure features."""
    structure = Structure(
        lattice=Lattice.cubic(3.5),
        species=["Si"],
        coords=[[0.0, 0.0, 0.0]],
    )

    features = extract_cslr_features(structure)
    result = predict(DummyModel(), features)

    assert isinstance(result, float)
    assert result == float(len(features.values))
