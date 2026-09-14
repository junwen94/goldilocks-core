"""write_qe_relax: render one ``pw.x`` relax/vc-relax ``Step`` from
resolved settings.

New in v2 (v2 epic 10, #10). ``write_qe_scf`` (v2 epic 7, #7) never
rendered a relax/vc-relax input at all -- its own docstring explicitly
earmarked "whichever future ``generation/quantum_espresso/relax.py``"
for that job. This module is that writer, sharing every
purpose-agnostic card/namelist helper with ``scf.py`` (un-privatized
there for this reuse: ``control_keywords``/``system_keywords``/
``electrons_keywords``/``atomic_species``/``cell_parameters``/
``atomic_positions``/``k_points``/``validated_pseudo_by_element``) --
the &CONTROL/&SYSTEM/&ELECTRONS namelists and every card are identical
between an scf step and a relax step; only &IONS/&CELL (this module's
own ``ions_keywords``/``cell_keywords``) and the ``&CONTROL``
relaxation-stopping keywords (``forc_conv_thr``/``etot_conv_thr``/
``nstep``, added to ``scf.py``'s own ``control_keywords`` since they
are real ``&CONTROL`` keys per ``INPUT_PW.txt``/ASE's own
``ALL_KEYS['pw']['control']`` registry, confirmed 2026-09-14 -- not an
``&IONS`` key as the design doc's own directory sketch might suggest)
are new.

``render_namelist`` sections a flat keyword dict automatically (ASE's
``Namelist.to_nested``, keyed against ``ALL_KEYS['pw']``) -- this writer
does not need to know or assert which QE namelist a keyword belongs to,
only which *keywords* apply, matching every other writer in this
package.

**``if_pos`` slab-layer fixing (#44).** When ``relax.fix_bottom_layers``
is set, this module (not ``analysis/geometry.py``, not
``advisors/relax.py``) detects which atoms sit in the bottom N layers
and passes their indices to ``atomic_positions``. Done here, at
generation time, against ``system.magnetic.relabeled_structure``
directly -- the exact structure ``atomic_positions`` iterates -- rather
than reusing a fact computed earlier in the pipeline against a
structure that AFM species-splitting could in principle have changed
the site count/order of (``advisors/relax.py``'s own docstring).
``checks.py``'s ``_fix_bottom_layers_requires_2d_geometry`` already
guarantees the structure classifies as 2D by the time this runs; the
detection below re-derives the actual stacking direction rather than
assuming the c-axis, since that assumption only holds for slabs built
by ``pymatgen``'s own ``SlabGenerator``/ASE's ``surface()`` (confirmed
2026-09-14: ``SlabGenerator`` always reorients its output cell so the
surface normal is c, regardless of the *original* bulk Miller index
requested), not for an arbitrary slab structure reaching this codebase
by another route.
"""

from __future__ import annotations

from typing import Literal

from ase.geometry import get_layers
from pymatgen.analysis.dimensionality import get_structure_components
from pymatgen.analysis.local_env import JmolNN
from pymatgen.core import Structure
from pymatgen.core.graphs import StructureGraph
from pymatgen.io.ase import AseAtomsAdaptor

from goldilocks_core.advisors.job_resources import JobDecision
from goldilocks_core.advisors.relax import RelaxOptions, VcRelaxOptions
from goldilocks_core.generation.errors import GenerationError
from goldilocks_core.generation.quantum_espresso.namelists import render_namelist
from goldilocks_core.generation.quantum_espresso.scf import (
    atomic_positions,
    atomic_species,
    cell_parameters,
    control_keywords,
    electrons_keywords,
    k_points,
    system_keywords,
    validated_pseudo_by_element,
)
from goldilocks_core.step_settings import PwSettings
from goldilocks_core.steps import SharedContext, Step
from goldilocks_core.system_settings import SystemSettings

_LAYER_TOLERANCE_ANGSTROM = 0.5
"""Maximum out-of-plane distance (Angstrom) for two atoms to count as
the same layer (``ase.geometry.get_layers``' own ``tolerance`` param,
default ``0.001`` -- far too tight for anything but a perfectly flat,
freshly-generated slab). This is a chosen heuristic, not sourced from
an official QE/ASE/pymatgen recommendation for this exact problem --
loosely anchored to pymatgen's own ``Slab.get_tasker2_slabs`` same
-plane tolerance (``tol=0.01`` *fractional*, i.e. roughly this order of
magnitude in Angstrom for a typical slab+vacuum cell) and to typical
DFT-relaxation-induced surface rumpling being well under 1 Angstrom for
most systems. Worth a domain-expert sanity check against real slab
systems rather than treated as authoritative."""


def write_qe_relax(
    system: SystemSettings,
    step: PwSettings,
    job: JobDecision,
    ctx: SharedContext,
    purpose: Literal["relax", "vc-relax"],
) -> list[Step]:
    """``purpose`` selects both QE's own ``&CONTROL calculation`` value
    and which of ``RelaxOptions``/``VcRelaxOptions`` ``step.relax`` must
    be -- the two must agree, checked below, since ``PwSettings.relax``'s
    own type (``RelaxOptions | VcRelaxOptions | None``) cannot enforce
    that by itself.
    """
    structure = system.magnetic.relabeled_structure
    if not structure.is_ordered:
        raise GenerationError(
            "cannot generate Quantum ESPRESSO input for a disordered structure"
        )
    relax = step.relax
    if relax is None:
        raise GenerationError(
            f"PwSettings.relax is required to generate a {purpose}.in input"
        )
    is_vc_relax = isinstance(relax, VcRelaxOptions)
    if purpose == "vc-relax" and not is_vc_relax:
        raise GenerationError(
            "write_qe_relax needs a VcRelaxOptions for purpose='vc-relax', "
            f"got {type(relax).__name__}"
        )
    if purpose == "relax" and is_vc_relax:
        raise GenerationError(
            "write_qe_relax needs a plain RelaxOptions for purpose='relax'; "
            "got a VcRelaxOptions -- use purpose='vc-relax' instead"
        )
    for name, value in (
        ("occupations", step.occupations),
        ("k_sampling", step.k_sampling),
        ("nbnd", step.nbnd),
        ("convergence", step.convergence),
        ("parallel", step.parallel),
    ):
        if value is None:
            raise GenerationError(
                f"PwSettings.{name} is required to generate {purpose}.in"
            )
    if system.hubbard.plan != "not_needed":
        raise GenerationError(
            "a Hubbard +U correction was resolved "
            f"(plan={system.hubbard.plan!r}), but rendering the HUBBARD card "
            "is not implemented by this writer yet -- see advisors/hubbard_u.py's "
            "own note on the atom-index-keyed QE >= 7.1 card format"
        )

    elements = sorted({site.specie.symbol for site in structure})
    pseudo_by_element = validated_pseudo_by_element(elements, system)
    species_labels = sorted({site.label for site in structure})
    label_to_element = {site.label: site.specie.symbol for site in structure}
    species_index = {label: index + 1 for index, label in enumerate(species_labels)}

    keywords: dict[str, object] = {}
    keywords.update(control_keywords(ctx, step, job, purpose))
    keywords.update(
        system_keywords(structure, system, step, len(species_labels), species_index)
    )
    keywords.update(electrons_keywords(step))
    keywords.update(ions_keywords(relax))
    if isinstance(relax, VcRelaxOptions):
        keywords.update(cell_keywords(relax))

    lines = [render_namelist(keywords)]
    lines.append(atomic_species(species_labels, label_to_element, pseudo_by_element))
    lines.append(cell_parameters(structure))
    lines.append(atomic_positions(structure, _fixed_site_indices(structure, relax)))
    lines.append(k_points(step))
    content = "\n".join(lines)

    args = ["-npool", str(step.parallel.npool)]
    if step.parallel.ndiag is not None:
        args += ["-ndiag", str(step.parallel.ndiag)]
    args += ["-in", f"{purpose}.in"]

    return [
        Step(
            name=purpose,
            executable="pw.x",
            args=tuple(args),
            files={f"{purpose}.in": content},
            stdout=f"{purpose}.out",
        )
    ]


def ions_keywords(relax: RelaxOptions) -> dict[str, object]:
    """``trust_radius_max/min/ini`` are only meaningful for
    ``ion_dynamics='bfgs'`` (INPUT_PW.txt marks all three "(bfgs
    only)") -- omitted rather than writing a value QE would just ignore
    for ``damp``/``fire``, matching this codebase's existing convention
    of only writing a keyword when it applies given an already-resolved
    fact (e.g. ``scf.py``'s own ``_occupations_keywords``)."""
    keywords: dict[str, object] = {
        "ion_dynamics": relax.ion_dynamics,
        "remove_rigid_rot": relax.remove_rigid_rot,
    }
    if relax.ion_dynamics == "bfgs":
        keywords["trust_radius_max"] = relax.trust_radius_max
        keywords["trust_radius_min"] = relax.trust_radius_min
        keywords["trust_radius_ini"] = relax.trust_radius_ini
    return keywords


def cell_keywords(relax: VcRelaxOptions) -> dict[str, object]:
    return {
        "cell_dynamics": relax.cell_dynamics,
        "cell_dofree": relax.cell_dofree,
        "press": relax.press,
        "press_conv_thr": relax.press_conv_thr,
        "cell_factor": relax.cell_factor,
    }


def _fixed_site_indices(
    structure: Structure, relax: RelaxOptions
) -> frozenset[int] | None:
    if relax.fix_bottom_layers is None:
        return None
    return _bottom_layer_site_indices(structure, relax.fix_bottom_layers)


def _bottom_layer_site_indices(structure: Structure, count: int) -> frozenset[int]:
    """0-based indices (``structure``'s own site order) of every atom in
    the bottom ``count`` atomic layers along the structure's own
    stacking direction.

    Bonding method (``JmolNN``) matches ``analysis/geometry.py``'s own
    2D classification, so this reaches the same "is it 2D" answer
    ``checks.py`` already validated against, on a real structure rather
    than by assumption. A slab can decompose into more than one bonded
    component along the stacking direction (e.g. van-der-Waals-bonded
    multilayers, where ``JmolNN`` does not bond across the gap) -- every
    2D component's own detected orientation is required to agree, since
    disagreement means genuinely ambiguous stacking this function cannot
    safely guess at.
    """
    bonded = StructureGraph.from_local_env_strategy(structure, JmolNN())
    components = get_structure_components(bonded, inc_orientation=True)
    orientations = {
        component["orientation"]
        for component in components
        if component["dimensionality"] == 2
    }
    if not orientations:
        raise GenerationError(
            "relax.fix_bottom_layers: could not detect a 2D bonded component "
            "to determine the slab's stacking direction"
        )
    if len(orientations) > 1:
        raise GenerationError(
            "relax.fix_bottom_layers: detected more than one stacking "
            f"orientation across bonded components ({sorted(orientations)}); "
            "cannot determine an unambiguous set of atomic layers"
        )
    miller = tuple(int(component) for component in next(iter(orientations)))

    atoms = AseAtomsAdaptor.get_atoms(structure)
    layer_by_site, _ = get_layers(atoms, miller, tolerance=_LAYER_TOLERANCE_ANGSTROM)
    distinct_layers = sorted({int(layer) for layer in layer_by_site})
    if count > len(distinct_layers):
        raise GenerationError(
            f"relax.fix_bottom_layers={count} exceeds the {len(distinct_layers)} "
            "distinct atomic layers detected in this structure"
        )
    bottom_layers = set(distinct_layers[:count])
    return frozenset(
        index
        for index, layer in enumerate(layer_by_site)
        if int(layer) in bottom_layers
    )
