"""n_irr_k: symmetry-irreducible k-point count, for one already-chosen mesh.

New in v2 (v2 epic 6, #6). Third of five per-step advisors
(``occupations -> k_sampling -> n_irr_k -> nbnd -> convergence``,
goldilocks-core-design.md:3239-3248, :3385). Uses the same
``SpacegroupAnalyzer.get_ir_reciprocal_mesh`` call ``kmesh.py``'s own
``build_gamma_kmesh_entries`` uses internally to fill in each
``KMeshEntry.n_reduced_kpoints`` while enumerating a whole ladder -- this
module makes the same computation available for one already-chosen mesh
(the one ``advisors/k_sampling.py`` actually picked), without
re-building the ladder from scratch.

**Not guarded** against a structure ``SpacegroupAnalyzer`` cannot
handle, for the same reason ``kmesh.py``'s own ladder builder is not
guarded: the irreducible count and the full mesh size are both ordinary
integers, so a caller cannot tell a fallback from a real answer, and
core uses this number to size memory and choose ``npool`` -- a silently
wrong value there is far more dangerous than a raised error.

**``nosym``**: QE's own flag to disable symmetry reduction entirely
(every k-point in the full mesh is kept). When set, this returns the
full mesh size directly without calling into spglib at all. The other
three symmetry-related flags in the design doc's dependency table for
this step -- ``noinv``, ``no_t_rev``, ``force_symmorphic`` -- are not
exposed here: pymatgen's own ``get_ir_reciprocal_mesh`` never forwards
``noinv`` to spglib at all (a confirmed no-op), and ``no_t_rev``/
``force_symmorphic`` only matter once ``relabeled_structure`` carries
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
