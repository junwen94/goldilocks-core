"""n_irr_k: symmetry-irreducible k-point count, for one already-chosen mesh.

New in v2 (v2 epic 6, #6). Third of five per-step advisors
(``occupations -> k_sampling -> n_irr_k -> nbnd -> convergence``,
goldilocks-core-design.md:3239-3248, :3385). The one place core reduces
a k-mesh by symmetry: one spglib reduction (``SpacegroupAnalyzer.
get_ir_reciprocal_mesh``, or the magnetic count below) of the mesh
``advisors/k_sampling.py`` picked, on the simulated cell
(``MagneticCell``). The ladder
(``kmesh.build_gamma_kmesh_entries``) carries no reduced counts;
reducing every rung of it cost seconds per ``advise()`` (#155).

**Counted on the cell's species.** pw.x keeps only the operations that
map every atom onto one of its own species, and a spin-polarized cell
holds one species per element and starting-magnetization class
(``advisors/species_split.py``), each carried as ``Species.spin`` so
spglib sees the same types (#170). A cell whose sublattices start with
opposite moments therefore reduces less than its bare elements would.

**Noncollinear moments (#172).** A spin-polarized run with spin-orbit
coupling is the one case where pw.x reduces by magnetic symmetry
(``magnetic_sym = noncolin .AND. domag``, QE 7.3
``PW/src/setup.f90:484-485``): it keeps the operations that map every
moment onto its image's moment, alone or combined with time reversal
(``symm_base.f90``'s ``sgam_at_mag``), lets a time-reversed operation act
on k as -R, and drops the plain k -> -k pairing. Its moments are each
site's starting-magnetization class (``Species.spin``) times the cell's
moment direction (``MagneticCell.moment_axis``), on the species labels pw.x
types its atoms by; only their types, signs and direction matter, so the
count needs no pseudopotential. spglib's ``get_magnetic_symmetry`` finds
those operations, and ``get_stabilized_reciprocal_mesh`` reduces the mesh
by {R, or -R when time-reversed}, without time reversal. B2 FeCo with
moments along [001] keeps 16 of 48 operations, 126 k-points of a 10^3
mesh against the structural 56, as pw.x 7.5 ran it. Every other run
counts on the structure alone.

**Not guarded** against a structure ``SpacegroupAnalyzer`` cannot
handle, and pymatgen is a required dependency, so there is no import
fallback either. The irreducible count and the full mesh size are both
ordinary integers, so a caller cannot tell a fallback from a real answer
(a cubic cell reduces by up to 48x), and core uses this number to size
memory and choose ``npool`` -- a silently wrong value there is far more
dangerous than a raised error.

**``nosym``**: QE's own flag to disable symmetry reduction entirely
(every k-point in the full mesh is kept). When set, this returns the
full mesh size directly without calling into spglib at all. The other
three symmetry-related flags in the design doc's dependency table for
this step -- ``noinv``, ``no_t_rev``, ``force_symmorphic`` -- are not
exposed here: pymatgen's own ``get_ir_reciprocal_mesh`` never forwards
``noinv`` to spglib at all (a confirmed no-op), and ``no_t_rev``/
``force_symmorphic`` would change the magnetic count above, but no
setting writes them to the input, so pw.x runs with its defaults, which
this count follows. Add them once something writes them.

No ``llm`` override: ``nosym`` is an operator's technical choice about
how a specific calculation should be run (e.g. preserving site-resolved
information around a defect), not a scientific judgement call an agent
is asked to make on the structure's behalf -- the same category
``analysis/symmetry.py`` put itself in when it declined an override
entirely, though here a human operator's own preference is still
plausible enough to keep.
"""

from __future__ import annotations

import numpy as np
import spglib
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from goldilocks_core.advisors.k_sampling import KSamplingDecision
from goldilocks_core.advisors.magnetic_config import MagneticCell
from goldilocks_core.analysis.symmetry import SYMPREC
from goldilocks_core.inputs.overrides import HumanInput
from goldilocks_core.resolution import (
    Blocked,
    FieldState,
    Provenance,
    Resolved,
    blocked_by,
)


class NIrrKHumanInput(HumanInput):
    nosym: bool = False


def n_irr_k(
    cell: MagneticCell,
    k_sampling: FieldState[KSamplingDecision],
    human: NIrrKHumanInput | None = None,
) -> FieldState[int]:
    """The irreducible k-points of ``k_sampling``'s mesh on the simulated
    ``cell``: by its magnetic symmetry when it has a moment direction
    (a spin-polarized spin-orbit run), otherwise by its structural
    symmetry (module docstring)."""
    human = human or NIrrKHumanInput()

    if isinstance(k_sampling, Blocked):
        return Blocked(by=k_sampling)
    if not k_sampling.ok:
        return Blocked(by=blocked_by(k_sampling))

    mesh = k_sampling.value.mesh
    if human.nosym:
        return Resolved(mesh[0] * mesh[1] * mesh[2], Provenance(source="human"))

    shift = k_sampling.value.shift
    axis = cell.moment_axis
    if axis is None:
        analyzer = SpacegroupAnalyzer(cell.structure, symprec=SYMPREC)
        count = len(analyzer.get_ir_reciprocal_mesh(mesh=mesh, is_shift=shift))
    else:
        count = _magnetic_count(cell.structure, axis, mesh, shift)
    return Resolved(count, Provenance(source="heuristic"))


def _magnetic_count(
    structure: Structure,
    axis: np.ndarray,
    mesh: tuple[int, int, int],
    shift: tuple[int, int, int],
) -> int:
    """The irreducible points of ``mesh`` under the operations that keep
    the moments ``Species.spin`` x ``axis`` of ``structure``'s sites,
    typed by species label (module docstring)."""
    types = {label: index for index, label in enumerate(sorted(set(structure.labels)))}
    dataset = spglib.get_magnetic_symmetry(
        (
            structure.lattice.matrix,
            structure.frac_coords,
            [types[site.label] for site in structure],
            [site.specie.spin * axis for site in structure],
        ),
        symprec=SYMPREC,
        is_axial=True,
    )
    k_rotations = np.unique(
        [
            -rotation if reversed_ else rotation
            for rotation, reversed_ in zip(
                dataset["rotations"], dataset["time_reversals"], strict=True
            )
        ],
        axis=0,
    )
    mapping, _grid = spglib.get_stabilized_reciprocal_mesh(
        mesh, k_rotations, is_shift=shift, is_time_reversal=False
    )
    return len(np.unique(mapping))
