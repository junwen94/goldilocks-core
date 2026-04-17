import numpy as np
from pymatgen.core import Lattice, Structure

from goldilocks_core.ml.features import extract_l_features


def test_extract_l_features_returns_lattice_features() -> None:
    """Extract the expected lattice-based feature vector."""
    structure = Structure(
        lattice=Lattice.cubic(3.5),
        species=["Si"],
        coords=[[0.0, 0.0, 0.0]],
    )

    features = extract_l_features(structure)

    assert features.feature_names == [
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
        "volume",
    ]
    assert np.allclose(
        features.values,
        np.array([3.5, 3.5, 3.5, 90.0, 90.0, 90.0, 42.875]),
    )
