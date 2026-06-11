from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pymatgen.analysis.dimensionality import get_dimensionality_larsen
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

if TYPE_CHECKING:
    from goldilocks_core.ml.magnetic import MagneticClassifier


_MAGNETIC_ELEMENTS: frozenset[str] = frozenset({
    # 3d transition metals
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    # 4d transition metals
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
    # 5d transition metals
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    # lanthanides
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    # actinides
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
})

_SOC_Z_THRESHOLD: int = 30  # Z > 30 → Ga and heavier; conservative SOC screening

_POLAR_POINT_GROUPS: frozenset[str] = frozenset({
    "1", "2", "m", "mm2",
    "3", "3m",
    "4", "4mm",
    "6", "6mm",
})


@dataclass(frozen=True, slots=True)
class StructureAnalysis:
    """Factual observations about a structure. No parameter recommendations."""

    # Composition
    elements: list[str]
    n_atoms: int
    n_species: int
    contains_transition_metals: bool
    contains_lanthanides: bool
    contains_actinides: bool
    contains_heavy_elements: bool
    heavy_elements: list[str]

    # Symmetry
    space_group_number: int
    space_group_symbol: str
    crystal_system: str
    point_group: str
    has_inversion_symmetry: bool
    n_symmetry_operations: int

    # Electronic
    metallicity: Literal["metallic", "insulating", "likely_metallic", "likely_insulating", "unknown"]
    metallicity_source: Literal["heuristic", "ml", "both"]
    metallicity_confidence: float | None
    has_d_electrons: bool   # transition metals only
    has_f_electrons: bool   # lanthanides and actinides
    total_electrons: int    # sum of Z × site count; use pseudo valence electrons in Advise for spin check

    # Magnetic (ML prediction deferred to Phase 2)
    magnetic_prediction: Literal["non_magnetic", "collinear", "non_collinear"] | None
    magnetic_confidence: float | None
    magnetic_source: Literal["heuristic", "ml", "both"] | None
    magnetic_elements: list[str]

    # SOC risk flag (whether to use FR pseudo/noncolin decided in Advise layer)
    soc_relevant: bool

    # Geometry / dimensionality
    pbc: tuple[bool, bool, bool]
    dimensionality: Literal["3d", "2d", "1d", "0d"]
    system_type: Literal["bulk", "slab", "wire", "molecule", "cluster"]
    has_vacuum: bool
    is_slab: bool
    is_primitive: bool | None   # heuristic: compares site count with pymatgen primitive standard structure

    # Polarity and centrosymmetry
    is_noncentrosymmetric: bool  # not has_inversion; necessary but not sufficient for polarity
    is_polar: bool               # based on point group membership in the 10 polar point groups

    # Disorder
    has_partial_occupancy: bool
    disordered_sites: list[int]

    # Warnings
    warnings: list[str]


def analyze_structure(
    structure: Structure,
    symprec: float = 0.1,
    magnetic_classifier: MagneticClassifier | None = None,
) -> StructureAnalysis:
    """Return factual observations about a structure.

    Args:
        structure: Input crystal structure.
        symprec: Symmetry tolerance for SpacegroupAnalyzer.
        magnetic_classifier: Optional MagneticClassifier instance.  When
            provided, runs an mMACE-based binary classification to populate
            ``magnetic_prediction`` and ``magnetic_confidence`` (Phase 2).
            Requires ``goldilocks[mlip]``.  When None (default), these fields
            are left as None and the Advise layer falls back to the heuristic
            element-lookup.

    Returns:
        StructureAnalysis with observations only. No DFT parameters.
    """
    element_objs = structure.elements
    elements = [str(el) for el in element_objs]

    # Composition
    contains_tm = any(el.is_transition_metal for el in element_objs)
    contains_lanthanides = any(el.is_lanthanoid for el in element_objs)
    contains_actinides = any(el.is_actinoid for el in element_objs)
    contains_heavy = any(el.Z > _SOC_Z_THRESHOLD for el in element_objs)
    heavy_elements = [str(el) for el in element_objs if el.Z > _SOC_Z_THRESHOLD]
    has_d = contains_tm
    has_f = contains_lanthanides or contains_actinides
    total_electrons = int(round(sum(
        el.Z * structure.composition[el] for el in element_objs
    )))

    # Symmetry
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    sg_number = sga.get_space_group_number()
    sg_symbol = sga.get_space_group_symbol()
    crystal_system = sga.get_crystal_system()
    point_group = sga.get_point_group_symbol()
    sym_ops = sga.get_symmetry_operations()
    n_sym_ops = len(sym_ops)
    has_inversion = sga.is_laue()

    # Metallicity heuristic (conservative; ML integration in Phase 2)
    # Unknown → smearing in Advise layer (safe default).
    if all(el.is_metal for el in element_objs):
        metallicity: Literal[
            "metallic", "insulating", "likely_metallic", "likely_insulating", "unknown"
        ] = "likely_metallic"
    else:
        metallicity = "unknown"
    metallicity_source: Literal["heuristic", "ml", "both"] = "heuristic"
    metallicity_confidence: float | None = None

    # Magnetic
    mag_elements = [e for e in elements if e in _MAGNETIC_ELEMENTS]

    _classifier_error: str | None = None
    if magnetic_classifier is not None:
        try:
            ml_label, ml_conf = magnetic_classifier.predict(structure)
            magnetic_prediction: Literal["non_magnetic", "collinear", "non_collinear"] | None = ml_label
            magnetic_confidence: float | None = ml_conf
            magnetic_source: Literal["heuristic", "ml", "both"] | None = "ml"
        except Exception as exc:
            _classifier_error = f"MagneticClassifier failed: {exc}"
            magnetic_prediction = None
            magnetic_confidence = None
            magnetic_source = None
    else:
        magnetic_prediction = None
        magnetic_confidence = None
        magnetic_source = None

    # SOC
    soc_relevant = contains_heavy

    # Geometry / dimensionality
    pbc_raw = structure.pbc
    pbc: tuple[bool, bool, bool] = (bool(pbc_raw[0]), bool(pbc_raw[1]), bool(pbc_raw[2]))
    try:
        dim_value = get_dimensionality_larsen(structure)
    except Exception:
        dim_value = 3 if all(pbc) else sum(pbc)

    if dim_value == 3:
        dimensionality: Literal["3d", "2d", "1d", "0d"] = "3d"
        system_type: Literal["bulk", "slab", "wire", "molecule", "cluster"] = "bulk"
    elif dim_value == 2:
        dimensionality = "2d"
        system_type = "slab"
    elif dim_value == 1:
        dimensionality = "1d"
        system_type = "wire"
    else:
        dimensionality = "0d"
        system_type = "molecule"

    has_vacuum = dim_value < 3
    is_slab = dim_value == 2

    try:
        prim = sga.get_primitive_standard_structure()
        is_primitive: bool | None = len(prim) == len(structure)
    except Exception:
        is_primitive = None

    # Polarity
    is_noncentrosymmetric = not has_inversion
    is_polar = point_group in _POLAR_POINT_GROUPS

    # Disorder
    disordered_sites = [i for i, site in enumerate(structure) if not site.is_ordered]
    has_partial_occupancy = bool(disordered_sites)

    # Warnings (risk flags only, no parameter recommendations)
    warnings: list[str] = []
    if _classifier_error is not None:
        warnings.append(_classifier_error)
    if has_partial_occupancy:
        warnings.append(f"Partial occupancy at sites: {disordered_sites}")
    if has_f:
        warnings.append("f-electron elements detected: check whether DFT+U is needed")
    elif has_d:
        warnings.append("transition-metal d-electrons detected: consider DFT+U")
    if soc_relevant and is_noncentrosymmetric:
        warnings.append(
            "Heavy elements without inversion symmetry: check Rashba/Dresselhaus effects"
        )
    if structure.num_sites > 100:
        warnings.append(f"Large unit cell ({structure.num_sites} atoms): consider cost")

    return StructureAnalysis(
        elements=elements,
        n_atoms=structure.num_sites,
        n_species=len(element_objs),
        contains_transition_metals=contains_tm,
        contains_lanthanides=contains_lanthanides,
        contains_actinides=contains_actinides,
        contains_heavy_elements=contains_heavy,
        heavy_elements=heavy_elements,
        space_group_number=sg_number,
        space_group_symbol=sg_symbol,
        crystal_system=crystal_system,
        point_group=point_group,
        has_inversion_symmetry=has_inversion,
        n_symmetry_operations=n_sym_ops,
        metallicity=metallicity,
        metallicity_source=metallicity_source,
        metallicity_confidence=metallicity_confidence,
        has_d_electrons=has_d,
        has_f_electrons=has_f,
        total_electrons=total_electrons,
        magnetic_prediction=magnetic_prediction,
        magnetic_confidence=magnetic_confidence,
        magnetic_source=magnetic_source,
        magnetic_elements=mag_elements,
        soc_relevant=soc_relevant,
        pbc=pbc,
        dimensionality=dimensionality,
        system_type=system_type,
        has_vacuum=has_vacuum,
        is_slab=is_slab,
        is_primitive=is_primitive,
        is_noncentrosymmetric=is_noncentrosymmetric,
        is_polar=is_polar,
        has_partial_occupancy=has_partial_occupancy,
        disordered_sites=disordered_sites,
        warnings=warnings,
    )
