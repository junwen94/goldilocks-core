# Check your recommendations

Goldilocks suggests starting inputs for a Quantum ESPRESSO calculation --
self-consistent-field (SCF), density of states (DOS), relaxation (`relax`), or
variable-cell relaxation (`vc-relax`). It does not establish that your energy,
forces, or other properties are converged. Review the choices below before
running a calculation, then test the settings against the accuracy your work
needs.

For a first calculation, use the [quickstart](quickstart.md). For exact units,
defaults, and override precedence, see [Scientific conventions](conventions.md).

## Check electronic occupations

Smearing smooths electron occupations near the Fermi energy to help metals
converge. Goldilocks chooses the occupations scheme from the `is_metal`
analysis fact:

| `is_metal` result                                     | Default occupations                              |
| ------------------------------------------------------ | ------------------------------------------------- |
| `metal`, or unavailable (composition couldn't confirm) | Smearing (`smearing_type='cold'`, `degauss=0.01` Ry) |
| `non_metal`                                            | Fixed occupations, without smearing               |

Ahead of the heuristic, a published CGCNN classifier (installed by default as
`models/is-metal-classifier`) can resolve `is_metal` directly from structure;
`is_metal` only falls back to the heuristic below when that model asset is
not installed or `goldilocks-ml` is not importable. The heuristic itself
excludes common anion-forming elements (O, N, S, Se, Te, F, Cl, Br, I, H)
from the composition, then checks whether every remaining element is a metal
according to pymatgen. It returns `metal` if so; otherwise it reports that
composition alone can't confirm metallic character, which -- like an actual
`metal` result -- still defaults to smearing, not fixed occupations, because
assuming "fixed" on an undetected metal risks silent non-convergence. A
confident `non_metal` result is reachable via a human override, the ML
classifier, or an LLM override. Neither the heuristic nor the ML model
determines a band structure.

Check the `is_metal` record's `status`, `source`, and `reason` (via
`goldilocks explain --json` or the published `goldilocks.json`'s
`records.is_metal`), and the `occupations` record's own `warnings` array. A
composition-only "metal" result can still give the wrong occupation choice
for your system. Override the scheme itself with `occupations` (`fixed`,
`smearing`, or `tetrahedra_opt`); when using smearing, refine it with
`smearing_type` (default `cold`) and `degauss` (Ry, default 0.01). Test the
width together with the k-point mesh.

## Check k-point sampling

K-points sample the Brillouin zone for reciprocal-space integration. Too sparse
a mesh can leave energies, forces, or electronic properties unconverged.

Without a grid or distance hint, Goldilocks first asks a published
machine-learning k-point model (`qrf-kpoints`, installed by default) to
predict a raw `k_distance` for the structure; only when that model asset is
not installed or `goldilocks-ml` is not importable does it fall back to
picking a target k-point spacing heuristically: 0.15 Å⁻¹ if `is_metal`
resolves to `metal`, otherwise 0.30 Å⁻¹ (this also covers the unavailable
case, since a mesh that is too coarse on an undetected metal risks silent
under-convergence, while a mesh that is too fine on an actual insulator only
costs more compute). There is no "confidence" or prediction-interval field
in `explain`/`run` output today.

Compare results on denser meshes. For slabs, wires, or molecules in periodic
cells, check sampling along vacuum directions explicitly; dimensionality advice
does not by itself set those mesh components to one. Use `k_grid` or
`k_distance` to override the default. Setting either one only affects
k-point selection; it does not change the separate `is_metal` heuristic used
elsewhere (e.g. by occupations).

## Check pseudopotentials and cutoffs

A pseudopotential replaces the core-electron potential with an effective
potential for the valence electrons. Check that its functional, valence
configuration, and relativistic treatment suit your system.

Goldilocks selects files and cutoff metadata, not system-specific convergence
limits. Quantum ESPRESSO receives the largest wavefunction cutoff and largest
charge-density cutoff among the selected elements. An `efficiency` or
`precision` table is a library choice, not proof that your target property is
converged. Test both cutoffs. See [Pseudopotential tables](pseudopotentials.md)
for selection rules, installation, and custom files.

## Check magnetism and spin-orbit coupling

Spin polarization allows different spin populations, decided by `is_magnetic`
(`human > ml > llm > heuristic`, see [CLI reference](cli.md#scientific-controls)).
Once its ML asset is installed, `mace`/`e3nn`/`sphericart` are on hand (a
manual install -- see the CLI reference), and `GOLDILOCKS_MACE_BACKBONE`
points at a downloaded mMACE checkpoint, a published mMACE-embedding
classifier answers directly; the heuristic tier otherwise enables it by
default for lanthanides or actinides whenever present, and for transition
metals only when a guessed oxidation state implies an open d-shell (a
d-electron count other than 0 or 10 -- a closed or empty d-shell has no
unpaired d electrons to align). Oxidation states are guessed from composition
alone (pymatgen's `Composition.oxi_state_guesses`); if that guess is empty,
ambiguous, or chemically implausible, the heuristic can't resolve
`is_magnetic` either, and Goldilocks defaults to non-magnetic with a warning
rather than guessing either way. Neither tier predicts magnetic *order* (see
`magnetic_ordering` below for that). The generated input does not assign
starting magnetic moments or magnetic sublattices; review and complete the
magnetic setup for your calculation. Override either tier with
`spin_polarized`.

Spin-orbit coupling (SOC) couples electron spin to orbital motion. Goldilocks
flags elements at or above Rb (Z=37) whose valence character is p, d, or f for
consideration, but does not enable SOC unless you request it -- heavy s-block
elements (Rb, Sr, Cs, Ba) are excluded because s orbitals carry no
first-order spin-orbit splitting. Whether SOC matters depends on the property,
not only the elements. Enabling it changes the required pseudopotentials and
the QE spin settings. See [relativistic modes](conventions.md#relativistic-modes)
and the [table restrictions](pseudopotentials.md#choose-a-table).

Set `magnetic_ordering` to `afm` to opt in to a compensated antiferromagnetic
search instead of the ferromagnetic default. Goldilocks cannot tell you which
magnetic ordering is the true ground state -- that needs comparing total
energies from several actual calculations -- so this only gives one
reasonable, deterministic starting point (the smallest compensated ordering
found), for you to run and compare against the ferromagnetic guess yourself.
This needs the external `enumlib` executables (`enum.x` or `multienum.x`,
plus `makeStr.py`) on `PATH`; without them, above a site-count ceiling, or if
no compensated ordering exists for your structure, it degrades to the
ferromagnetic default with a warning rather than failing.

`uv run goldilocks magnetic-orderings` lists every candidate this same
enumeration finds -- the ferromagnetic guess plus any compensated
antiferromagnetic ones -- for you to inspect or generate inputs from
directly, optionally ranked by relaxing each on a frozen mMACE potential
energy surface (`--rank-with-mmace`) rather than picking the ferromagnetic
guess by default. See [Magnetic orderings](cli.md#magnetic-orderings) for
the command and its setup requirements.

## Check dispersion and dimensionality

Dispersion accounts for long-range interactions that common semilocal
functionals can miss. The default D3BJ correction is the D3 method with
Becke–Johnson damping. Goldilocks enables it (`use_vdw=True`) for structures
classified as `molecule`, `1d`, or `2d`, and disables it (`use_vdw=False`) for
`3d`.

The classification uses pymatgen's JmolNN bonding method plus Larsen's
connectivity-rank algorithm to assign dimensionality. It is a heuristic, not a
test of whether a dispersion correction is physically appropriate. Check the
structure and choose `use_vdw` and `method` accordingly, including for
molecular crystals classified as `3d`. Disordered structures, and a bonding or
classification failure on an ordered structure, both report that
dimensionality couldn't be determined (with a reason) rather than raising an
error or silently guessing -- and the vdW decision itself then becomes
unresolved (blocked), not silently set to `use_vdw=False`.

## Check SCF convergence

The SCF energy-convergence threshold (`conv_thr`) and total-energy threshold
(`etot_conv_thr`) scale with atom count rather than being flat defaults;
density-mixing strength (`mixing_beta`) and the iteration limit
(`electron_maxstep`) are flat defaults carried over unchanged. None of these
are model predictions. Inspect the actual QE convergence history and adjust
these settings if needed. Reaching the SCF threshold does not demonstrate
convergence with respect to k-points or cutoffs.

## Check relaxation settings

For `relax` and `vc-relax` tasks, Goldilocks also picks ionic and cell
relaxation parameters -- `ion_dynamics`, `cell_dofree`, `cell_factor`,
`forc_conv_thr`, `nstep`, `press`, `press_conv_thr`,
`trust_radius_ini`/`trust_radius_max`/`trust_radius_min`, `remove_rigid_rot`,
and `fix_bottom_layers` (fixing the bottom N atomic layers of a 2D slab
during relaxation; only valid when the structure classifies as 2D). Every
default is copied verbatim from Quantum ESPRESSO's own documentation, not
derived from a model. See `goldilocks settings` for each key's group, type,
and default.

## Read the reasons and warnings

Every analysis fact and every advisor decision -- occupations, k-point
sampling, pseudopotential selection, and every other record in
`goldilocks explain`/`goldilocks.json` -- resolves to one of three states:
`resolved`, `unavailable`, or `blocked`. A resolved record carries a `source`
of `human`, `ml`, `llm`, or `heuristic`, in that priority order (highest
first); an unavailable record carries a `reason` instead of a value; a
blocked record carries `blocked_by`, naming the upstream fact that could not
be resolved. There is no separate "provenance block" distinct from structure
analysis -- both use this same shape.

This describes a decision as a whole, not necessarily each field inside it
separately: overriding one convergence setting (e.g. `mixing_beta`) does not
by itself mark the whole convergence decision `human`-sourced. Advisors precise
enough to track that instead populate an opt-in `field_sources` map naming
which individual scalar was overridden, alongside the decision's own overall
`source`. Per-decision warnings live inside that decision's own value (e.g.
`job.warnings`, `cutoffs.warnings`), and every run also carries one aggregated
top-level `warnings` list. There is no `confidence`, `details`, or "asset
identity" field in this shape today. Read analysis and selection warnings as
well as advice; do not treat a `resolved` status as scientific validation.
