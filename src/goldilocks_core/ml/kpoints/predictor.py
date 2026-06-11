"""K-point spacing predictor using CGCNN + Quantile Regression Forest.

Predicts optimal k-point spacing (Å⁻¹) for a crystal structure.
Models are downloaded from HuggingFace on first use and cached locally.
The CGCNN metallicity model is bundled in goldilocks_core/data/kpoints/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_MODELS_DIR = Path(__file__).parent.parent.parent / "data" / "models"
_METAL_DIR = _MODELS_DIR / "metallicity" / "1.0"
_ATOM_INIT_PATH = _METAL_DIR / "atom_init.json"
_IS_METAL_CKPT = _METAL_DIR / "is_metal.ckpt"

_CONFIDENCE_CORRECTIONS: dict[str, dict[float, float]] = {
    "RF":     {0.95: -0.0016, 0.90: -0.0023, 0.85: -0.0021},
    "ALIGNN": {0.95:  0.0005, 0.90:  0.0025, 0.85: -0.0018},
}

_RF_HF_REPO = "STFC-SCD/kpoints-goldilocks-QRF"
_RF_FILES = {0.95: "QRF95.pkl", 0.90: "QRF90.pkl", 0.85: "QRF85.pkl"}


def _load_cgcnn_metal_model():
    """Load CGCNN metallicity model from bundled checkpoint."""
    import torch

    from goldilocks_core.ml.kpoints.cgcnn import CGCNN_PyG

    ckpt = torch.load(_IS_METAL_CKPT, map_location="cpu", weights_only=True)
    model = CGCNN_PyG(**ckpt["hyper_parameters"]["model"])
    weights = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(weights)
    model.eval()
    return model


def _metal_features(structure, metal_model) -> np.ndarray:
    """Extract CGCNN crystal representation as metallicity features."""
    import torch

    from goldilocks_core.ml.kpoints.atom_features import atom_features_from_structure
    from goldilocks_core.ml.kpoints.cgcnn_graph import build_radius_cgcnn_graph_from_structure

    atomic_cfg = {
        "atom_feature_strategy": {
            "atom_feature_file": str(_ATOM_INIT_PATH),
            "soap_atomic": False,
        }
    }
    atom_feats = atom_features_from_structure(structure, atomic_cfg)
    data = build_radius_cgcnn_graph_from_structure(structure, atom_feats)
    with torch.no_grad():
        repr_tensor = metal_model.extract_crystal_repr(data)
    return repr_tensor.numpy()


def _structure_features(structure) -> np.ndarray:
    """Extract composition + structure + SOAP + lattice feature vector."""
    from pymatgen.core.composition import Composition

    from goldilocks_core.ml.kpoints.features import (
        lattice_features,
        matminer_composition_features,
        matminer_structure_features,
        soap_features,
    )

    formula = Composition(
        Composition(structure.formula).get_integer_formula_and_factor()[0]
    ).iupac_formula

    df = pd.DataFrame(
        {"id": [0], "structure": [structure], "formula": [formula],
         "composition": [Composition(structure.formula)]}
    )

    comp = matminer_composition_features(
        df, ["ElementProperty", "Stoichiometry", "ValenceOrbital"]
    )
    struct = matminer_structure_features(
        df, ["GlobalSymmetryFeatures", "DensityFeatures"]
    )
    soap = soap_features(
        df, soap_params={"r_cut": 10.0, "n_max": 8, "l_max": 6, "sigma": 1.0}
    )
    lattice = lattice_features(df)

    return np.concatenate([comp, struct, soap, lattice], axis=1)


class KSpacingPredictor:
    """Predict k-point spacing (Å⁻¹) using RF + CGCNN metallicity features.

    On first call the QRF model is downloaded from HuggingFace and cached.
    The CGCNN metallicity model is loaded from bundled data.
    """

    def __init__(self, confidence_level: float = 0.95) -> None:
        if confidence_level not in _RF_FILES:
            raise ValueError(
                f"confidence_level must be one of {list(_RF_FILES)}; got {confidence_level}"
            )
        self._conf = confidence_level
        self._corr = _CONFIDENCE_CORRECTIONS["RF"][confidence_level]
        self._metal_model = _load_cgcnn_metal_model()
        self._rf_model = self._download_rf()

    def _download_rf(self):
        import joblib
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=_RF_HF_REPO,
            filename=_RF_FILES[self._conf],
        )
        return joblib.load(path)

    def predict(
        self, structure
    ) -> tuple[float, float, float]:
        """Return (kdist, kdist_upper, kdist_lower) in Å⁻¹."""
        struct_feats = _structure_features(structure)
        metal_feats = _metal_features(structure, self._metal_model)
        features = np.concatenate([struct_feats, metal_feats], axis=1)

        out = self._rf_model.predict(features)
        kdist: float = float(out[1][0])
        kdist_lower: float = float(out[0][0]) - self._corr
        kdist_upper: float = float(out[2][0]) + self._corr
        return kdist, kdist_upper, kdist_lower
