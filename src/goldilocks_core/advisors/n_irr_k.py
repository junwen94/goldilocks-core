"""n_irr_k: symmetry-irreducible k-point count, for one already-chosen mesh.

New in v2 (v2 epic 6, #6). Third of five per-step advisors
(``occupations -> k_sampling -> n_irr_k -> nbnd -> convergence``,
goldilocks-core-design.md:3239-3248, :3385). The one place core reduces
a k-mesh by symmetry: one ``SpacegroupAnalyzer.get_ir_reciprocal_mesh``
call on the mesh ``advisors/k_sampling.py`` picked. The ladder
(``kmesh.build_gamma_kmesh_entries``) carries no reduced counts;
reducing every rung of it cost seconds per ``advise()`` (#155).

**Counted on the cell's species.** pw.x keeps only the operations that
map every atom onto one of its own species, and a spin-polarized cell
holds one species per element and starting-magnetization class
(``advisors/species_split.py``), each carried as ``Species.spin`` so
spglib sees the same types (#170). A cell whose sublattices start with
opposite moments therefore reduces less than its bare elements would.

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
``force_symmorphic`` only matter once the simulated cell carries
real (not FM-identity) magnetic moments, which it does not yet
(``advisors/magnetic_config.py``). Adding fields for flags that
currently do nothing would be speculative; add them once the first
thing that actually needs them exists.

No ``llm`` override: ``nosym`` is an operator's technical choice about
how a specific calculation should be run (e.g. preserving site-resolved
information around a defect), not a scientific judgement call an agent
is asked to make on the structure's behalf -- the same category
``analysis/symmetry.py`` put itself in when it declined an override
entirely, though here a human operator's own preference is still
plausible enough to keep.
"""

from __future__ import annotations

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from goldilocks_core.advisors.k_sampling import KSamplingDecision
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
    structure: Structure,
    k_sampling: FieldState[KSamplingDecision],
    human: NIrrKHumanInput | None = None,
) -> FieldState[int]:
    human = human or NIrrKHumanInput()

    if isinstance(k_sampling, Blocked):
        return Blocked(by=k_sampling)
    if not k_sampling.ok:
        return Blocked(by=blocked_by(k_sampling))

    mesh = k_sampling.value.mesh
    if human.nosym:
        return Resolved(mesh[0] * mesh[1] * mesh[2], Provenance(source="human"))

    analyzer = SpacegroupAnalyzer(structure)
    count = len(
        analyzer.get_ir_reciprocal_mesh(mesh=mesh, is_shift=k_sampling.value.shift)
    )
    return Resolved(count, Provenance(source="heuristic"))
