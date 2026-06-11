"""Magnetic property classifier using mMACE embeddings + MLP head.

Requires the ``goldilocks[mlip]`` optional dependency group (mace with
magnetic support, torch).  All heavy imports are lazy so this module can
always be imported, but instantiation will raise ``ImportError`` when the
optional dependencies are absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

MagneticLabel = Literal["non_magnetic", "collinear", "non_collinear"]

# Per-element initial magnetic moments (z-component) passed to mMACE.
# These are numerical seeds, not ground-truth labels.
_ELEMENT_INIT_MAG: dict[int, float] = {
    24: 4.0,  # Cr
    25: 4.0,  # Mn
    26: 2.5,  # Fe
    27: 1.7,  # Co
    28: 0.6,  # Ni
    44: 1.0,  # Ru
    45: 0.5,  # Rh
    60: 3.0,  # Nd
    64: 7.0,  # Gd
}

_MLP_HIDDEN = (512, 256, 128)


def _build_mlp(d_in: int, hidden: tuple[int, ...] = _MLP_HIDDEN):  # type: ignore[return]
    """Build the MagneticMLP architecture (must match training exactly)."""
    import torch.nn as nn

    dims = [d_in] + list(hidden)
    layers = []
    for i in range(len(dims) - 1):
        layers += [
            nn.Linear(dims[i], dims[i + 1]),
            nn.BatchNorm1d(dims[i + 1]),
            nn.GELU(),
            nn.Dropout(0.3),
        ]
    layers.append(nn.Linear(dims[-1], 1))
    return nn.Sequential(*layers)


class MagneticClassifier:
    """Binary magnetic classifier: non-magnetic vs magnetic.

    Uses a single mMACE forward pass (no SCF) to extract per-atom embeddings
    from the last ``products`` layer, mean-pools to a structure vector, then
    runs a small MLP head trained on 42 K Materials Project structures.

    Args:
        backbone_path: Path to the mMACE backbone model file
            (e.g. ``mace_matpes_pbe_baseline_run-3.model``).  Not bundled
            with the package because of its size (~500 MB).
        mlp_ckpt_path: Path to the MLP + scaler checkpoint (``.pt``).
            Defaults to the bundled checkpoint in ``goldilocks_core/data``.
        device: Torch device string (``"cpu"`` or ``"cuda"``).

    Requires:
        ``goldilocks[mlip]`` optional dependency group (mace with magnetic
        support, torch).
    """

    def __init__(
        self,
        backbone_path: Path | str,
        mlp_ckpt_path: Path | str | None = None,
        device: str = "cpu",
    ) -> None:
        try:
            import torch
            from mace.calculators.mace import MagneticMACECalculator
            from mace.modules import MagneticSCFMACE
        except ImportError as exc:
            raise ImportError(
                "MagneticClassifier requires mace with magnetic support and torch. "
                "Install with: pip install 'goldilocks-core[mlip]' and ensure "
                "the magnetic-mace package is available."
            ) from exc

        if mlp_ckpt_path is None:
            from goldilocks_core.data import model_dir
            mlp_ckpt_path = model_dir("magnetic_classifier", "1.0") / "magnetic_clf.pt"

        self._device = device

        # Load backbone
        backbone_path = Path(backbone_path)
        if not backbone_path.exists():
            raise FileNotFoundError(f"mMACE backbone not found: {backbone_path}")

        # Patch torch.jit.load to respect device mapping (required for mMACE)
        _orig_jit_load = torch.jit.load

        def _patched_jit_load(f, map_location=None, **kw):
            return _orig_jit_load(f, map_location=device, **kw)

        torch.jit.load = _patched_jit_load
        try:
            raw_backbone = torch.load(
                backbone_path, map_location=device, weights_only=False
            )
        finally:
            torch.jit.load = _orig_jit_load

        scf_1step = MagneticSCFMACE(
            raw_backbone, n_scf_step=1, use_scf=False, scf_tol=1e-4, scf_logging=False
        )
        self._calc = MagneticMACECalculator(
            models=[scf_1step], device=device, default_dtype="float64"
        )
        self._raw = raw_backbone
        self._hook_buf: dict[str, object] = {}

        # Load MLP + scaler
        mlp_ckpt_path = Path(mlp_ckpt_path)
        if not mlp_ckpt_path.exists():
            raise FileNotFoundError(f"MLP checkpoint not found: {mlp_ckpt_path}")

        ckpt = torch.load(mlp_ckpt_path, map_location="cpu", weights_only=False)
        d_in: int = int(ckpt["d_in"])

        self._mlp = _build_mlp(d_in)
        state_dict = ckpt["model_state"]
        # Checkpoint may have been saved with a 'net' wrapper; strip prefix if so.
        if all(k.startswith("net.") for k in state_dict):
            state_dict = {k[4:]: v for k, v in state_dict.items()}
        self._mlp.load_state_dict(state_dict)
        self._mlp.eval()
        self._mlp.float()

        self._mean: np.ndarray = ckpt["scaler_mean"]   # (d_in,) float32
        self._scale: np.ndarray = ckpt["scaler_scale"]  # (d_in,) float32

        self.val_auc: float = float(ckpt.get("val_auc", 0.0))
        self.epoch: int = int(ckpt.get("epoch", 0))

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, atoms) -> np.ndarray:  # type: ignore[return, misc]
        """Return mean-pooled mMACE embedding (shape: d_in,) for one structure."""
        import torch

        a = atoms.copy()
        zs = a.get_atomic_numbers()
        mag = np.zeros((len(a), 3))
        for i, z in enumerate(zs):
            mag[i, 2] = _ELEMENT_INIT_MAG.get(int(z), 0.0)
        a.arrays["dft_magmom"] = mag

        buf: dict[str, object] = {}

        def _hook(m, inp, out):
            feats = out[0] if isinstance(out, tuple) else out
            buf["feats"] = feats.detach().cpu()

        handle = self._raw.products[-1].register_forward_hook(_hook)
        try:
            a.calc = self._calc
            a.get_potential_energy()
        finally:
            handle.remove()

        feats = buf.get("feats")
        if feats is None:
            raise RuntimeError("Forward hook did not fire — check mMACE version.")
        assert isinstance(feats, torch.Tensor)
        result: np.ndarray = feats.mean(dim=0).numpy()  # type: ignore[assignment]
        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        structure,
        threshold: float = 0.5,
    ) -> tuple[MagneticLabel, float]:
        """Predict magnetic character of a structure.

        Args:
            structure: pymatgen Structure.
            threshold: Probability threshold for magnetic classification.

        Returns:
            ``(label, probability)`` where label is one of
            ``"non_magnetic"`` or ``"collinear"`` (Phase 1; multi-class in
            Phase 2), and probability is P(magnetic) ∈ (0, 1).
        """
        import torch
        from pymatgen.io.ase import AseAtomsAdaptor

        atoms = AseAtomsAdaptor.get_atoms(structure)
        z = self._embed(atoms)

        z_sc = (z - self._mean) / self._scale
        x = torch.tensor(z_sc, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            prob = float(torch.sigmoid(self._mlp(x)).item())

        label: MagneticLabel = "non_magnetic" if prob < threshold else "collinear"
        return label, prob
