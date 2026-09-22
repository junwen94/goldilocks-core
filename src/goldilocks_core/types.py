from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

JsonDict = dict[str, Any]

PathLike = str | Path

CodeName = str

CalcTask = Literal["scf_single_point", "dos", "relax", "vc-relax"]
"""``dos`` added v2 epic 9, #9. ``relax``/``vc-relax`` added v2 epic 10,
#10 -- both single-step (one ``pw.x`` run with ``calculation`` set
accordingly), unlike ``dos``'s three-step scf/nscf/dos.x sequence."""

SmearingType = Literal["fixed", "gaussian", "mp", "cold"]

ModelType = Literal["random_forest", "cgcnn", "xgboost", "mlp"]

PseudoAccuracy = Literal["efficiency", "precision"]

PseudoType = Literal["NC", "USPP", "PAW"]

RelativisticTreatment = Literal["scalar", "full", "non-relativistic"]

KPointGrid = tuple[int, int, int]

KPointShift = tuple[Literal[0, 1], Literal[0, 1], Literal[0, 1]]


Dimensionality = Literal["3d", "2d", "1d", "molecule", "unknown"]

VdwMethod = Literal["d3", "d3bj", "ts", "mbd"]
"""Code-agnostic labels mapped to code-specific keywords in Generate
(e.g. ``d3bj`` → QE ``vdw_corr='grimme-d3'`` with ``dftd3_version=4``)."""
