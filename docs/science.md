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
not installed or fails verification. If the installed model fails to run,
the request fails rather than silently using the heuristic. The heuristic
itself excludes common anion-forming elements (O, N, S, Se, Te, F, Cl, Br, I,
H) from the composition, then checks whether every remaining element is a
metal according to pymatgen. It returns `metal` if so; otherwise it reports that
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
predict a raw `k_distance` for the structure (if the installed model fails to
run, the request fails); only when that model asset is not installed or fails
verification does it fall back to picking a target k-point spacing
heuristically: 0.15 Å⁻¹ if `is_metal`
resolves to `metal`, otherwise 0.30 Å⁻¹ (this also covers the unavailable
case, since a mesh that is too coarse on an undetected metal risks silent
under-convergence, while a mesh that is too fine on an actual insulator only
costs more compute). There is no "confidence" or prediction-interval field
in `explain`/`run` output today.

Each lattice vector in the `geometry` record's `vacuum_axes` (see
[boundary conditions and vacuum](#check-boundary-conditions-and-vacuum)) gets
exactly one k-point, whatever the `k_distance` and its source: electrons do not
disperse across vacuum, so more k-points there only cost compute. A slab gets
one k-point across its vacuum, and a molecule in a box only the Γ-point,
written as a 1×1×1 `K_POINTS automatic` mesh (pw.x rejects `K_POINTS gamma`
with noncollinear spin or k-point pools). The rule reads the vacuum of the cell
that is written, not its dimensionality, so a bulk layered crystal such as
graphite is still sampled along its stacking axis. `k_index` rungs step the
periodic axes only.

Compare results on denser meshes. Use `k_grid` or `k_distance` to override
the default; an explicit `k_grid` is written as given, vacuum axes included.
Setting either one only affects k-point selection; it does not change the
separate `is_metal` heuristic used elsewhere (e.g. by occupations).

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
classifier answers directly (if it then fails to run, the request fails);
the heuristic tier otherwise enables it by
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
plus `makeStr.py`, `makestr.x` or `makeStr.x`) on `PATH`. Without them, above a
site-count ceiling, when pymatgen cannot enumerate your structure, when enumlib
runs out of time, or if no compensated ordering exists for your structure, it
degrades to the ferromagnetic default with a warning; any other enumeration
error fails the request. Enumeration runs in a separate worker process, inside
a private scratch directory, so it writes nothing to the current directory and
concurrent requests cannot disturb each other.

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

D3BJ's damping parameters are fitted per functional, and Quantum ESPRESSO's
DFT-D3 has them only for PBE and PBEsol. With any other functional, LDA
included, a low-dimensional structure gets no correction and the warning
`vdw.d3bj_unavailable_for_functional`, and Goldilocks refuses an explicit
`use_vdw=true` setting before generating the input. pw.x would otherwise stop
at startup with "functional name unknown".

The classification builds a bond graph and assigns dimensionality with
Larsen's connectivity-rank algorithm. Two atoms are bonded when they are
within pymatgen's JmolNN covalent cutoffs, or when one lies in the other's
nearest-neighbour shell: within 1.1 times that atom's shortest contact,
searched up to 6 Å. The shell bonds the ionic and metallic contacts that
covalent cutoffs miss, so rock-salt NaCl and bcc Na classify as `3d`. In
most molecular crystals and van der Waals layered crystals each atom's
nearest shell lies inside its own molecule or layer, so they classify as
`molecule` and `2d`. It is a heuristic, not a test of whether a dispersion
correction is physically appropriate: a solid held together only by
dispersion, such as a noble-gas crystal, classifies as `3d`, and so does a
molecular crystal whose intermolecular contacts fall inside those bonding
ranges. Check the structure and choose `use_vdw` and `method` accordingly.
Disordered structures, and a bonding or
classification failure on an ordered structure, both report that
dimensionality couldn't be determined (with a reason) rather than raising an
error or silently guessing -- and the vdW decision itself then becomes
unresolved (blocked), not silently set to `use_vdw=False`.

## Check boundary conditions and vacuum

A molecule, cluster or slab in a periodic cell interacts with its own
periodic images. The `isolation` setting says how the electrostatics treat
them: `periodic` (no correction), `slab` (the Coulomb interaction truncated
along the slab normal) or `isolated` (truncated along all three lattice
vectors). For Quantum ESPRESSO, `slab` writes `assume_isolated='2D'` and
`isolated` writes `assume_isolated='martyna-tuckerman'`; `periodic` writes
neither.

Goldilocks decides from vacuum, not from dimensionality. For each lattice
vector, the `geometry` record reports the widest atom-free slab parallel to
the other two vectors, measured between atom centres
(`vacuum_widths_angstrom`), and lists the vectors with more than 5 Å as
`vacuum_axes`. Only a cell with vacuum along all three lattice vectors is
`isolated` by default. Every other cell is `periodic`, including molecular
crystals such as CO2 dry ice (classified `molecule`), slabs and wires: pw.x has
no 1D truncation for a wire. A layer that is not parallel to two lattice
vectors shows no vacuum, so its cell is treated as periodic.

Martyna-Tuckerman treats the system as a molecule or cluster in a box, so
Goldilocks refuses `isolation='isolated'` on a cell without vacuum along all
three lattice vectors, whoever set it. The correction is exact only when the
electron density lies within half the cell along every axis, which 5 Å of
vacuum does not guarantee, so check the box size. pw.x computes no stress under
Martyna-Tuckerman: the input then omits `tstress`, and Goldilocks refuses a
`vc-relax`. Use `relax` for an isolated system, or set `isolation='periodic'`
for a periodic one.

`isolation='slab'` is Quantum ESPRESSO's 2D Coulomb truncation (Sohier,
Calandra and Mauri, Phys. Rev. B 96, 075448 (2017)), which computes energies,
forces and stresses in a two-dimensional framework. It matters for charged,
doped or polar slabs; for a neutral, symmetric monolayer with about 20 Å of
vacuum the periodic default is accurate, and Goldilocks never picks `slab` by
itself. Set it explicitly. Goldilocks refuses it unless the cell to be written
meets pw.x's preconditions: vacuum along c only, a and b in the Cartesian xy
plane, c along z (the only orientation in which pw.x's truncation formula is
exact), a cell at least 21 bohr (11.11 Å) high (pw.x stops when half the height
is below the pseudopotentials' 10 bohr radial cutoff), and vacuum at least 20
bohr (10.58 Å) wider than the slab. The last condition is Quantum ESPRESSO's
safe estimate of a cell more than twice as high as the slab including its
electrons; a monolayer with about 20 Å of vacuum meets it. Under
`slab` only the in-plane stress is meaningful, so a `vc-relax` must use
`cell_dofree='2Dxy'` (the default for such a slab) or `'2Dshape'`.

A slab can carry a dipole along its normal only if no symmetry operation
reverses that normal: an inversion centre, a mirror in the sheet, a two-fold
axis in the sheet or a rotoinversion axis along the normal cancels it. The
`symmetry` record lists the lattice vectors whose normal some operation
reverses as `reversible_axes`. For a polar slab with `periodic` boundaries,
such as a Janus MoSSe sheet, the images' dipoles add a spurious field across
the vacuum, and Goldilocks adds the `boundary.dipole_correction_suggested`
notice. It names Quantum ESPRESSO's dipole correction (`tefield` and
`dipfield` with `eamp = 0`) along the vacuum axis (`edir`), with the
sawtooth's drop (`emaxpos`, `eopreg`) in the middle of the vacuum, where QE
requires it; the `geometry` record's `slab` field locates the vacuum
(`vacuum_center`, `vacuum_fraction`). The notice offers `isolation='slab'`
only when the slab meets every precondition of that setting (the `slab`
field's `fits_2d_truncation`). It does not set the
correction itself, because pw.x then uses no symmetry and computes no stress.
The notice and the `geometry` record describe the structure as given. With
`magnetic_ordering=afm` the cell written can be a supercell that moves the
vacuum to another lattice vector or doubles the cell across it; the notice's
`edir`, `emaxpos` and `slab` offer then refer to the input cell, and
Goldilocks still refuses `slab` on the cell written.
Symmetric monolayers and bulk crystals get no notice: a bulk layered crystal
such as graphite classifies as `2d` but has no vacuum.

## Check Hubbard +U

A Hubbard +U correction counters the self-interaction error that semilocal
functionals make for localized d and f electrons. Goldilocks applies it when
the `needs_correlation` fact is true -- by default, a compound of a
transition metal, lanthanide, or actinide with O or F, the Materials Project
screening rule -- and never with a hybrid functional. Each element in the
built-in table below then gets its Dudarev U_eff (U - J, in eV) on its 3d,
4d, 5d, 4f, or 5f shell, written as one `U <species>-<manifold> <value>`
line per species under `HUBBARD (ortho-atomic)`:

| Source                              | U_eff (eV)                                                                         |
| ----------------------------------- | ---------------------------------------------------------------------------------- |
| Materials Project                   | V 3.25, Cr 3.7, Mn 3.9, Fe 5.3, Co 3.32, Ni 6.2, Mo 4.38, W 6.2                     |
| Wang et al. 2006 / Jain et al. 2011 | Cu 4.0                                                                             |
| Loschen et al. 2007                 | Ce 5.0                                                                             |
| Topsakal & Wentzcovitch 2014        | Pr 4.0, Nd 3.1, Pm 3.4, Sm 3.3, Eu 3.0, Gd 4.6, Tb 5.0, Dy 5.0, Ho 4.9, Er 4.2, Tm 4.8, Yb 3.0 |
| OQMD qmpy configuration (hubbards.yml) | Pa 4.0                                                                          |
| OQMD (Kirklin et al. 2015)          | U 4.0, Np 4.0, Pu 4.0                                                              |
| Suzuki et al. 2013                  | Am 4.0, Cm 4.0                                                                     |
| Chen & Kaltsoyannis 2025            | Bk 3.54                                                                            |

The other d-block elements, La, Lu, and Th get no U and no warning: the
Materials Project applies none to those d elements, and La3+, Lu3+, and Th4+
have empty or full shells. Ac, Cf, Es, Fm, Md, No, and Lr have no published
U; they also get plain DFT, with a `hubbard.u_value_missing` warning that
names the element and the setting to use (for Ac, why no setting can apply
one). When no element gets a U, the input has no HUBBARD card.

These values come from different codes and fits, and none was derived with
the ortho-atomic projectors the generated input uses: every built-in U
carries a `hubbard.projector_mismatch` warning that lists the VASP-derived
values. Every applied U, built-in or from `u_by_element`, carries a
`hubbard.starting_ns_eigenvalue_missing` warning, since +U calculations can
converge to a metastable occupation matrix and Goldilocks supplies no
initial guess; a `u_by_element` value gets no projector warning, because
Goldilocks does not know how it was derived. U also depends on oxidation
state and environment, so treat one value per element as a starting point:
test your property against U, or calibrate it (for example by linear
response with `hp.x`, which Goldilocks does not set up).

`u_by_element` replaces the table: only the listed elements present in the
structure get +U, e.g. `--set 'u_by_element={"Ti": 4.0}'`, and listed
elements the structure does not contain are ignored with a
`hubbard.u_by_element_unused` warning. Keys must be element symbols and
values finite and positive. La and Ac are rejected: no installed La
pseudopotential has an occupied 4f channel and the only Ac one has no 5f
channel, so pw.x stops on any U for them. `hubbard_needs_correlation=false`
runs without +U. The `hubbard` record in `goldilocks explain --json` and
`goldilocks.json` lists every applied U with its `value_source` (`table` or
`human`), citation, and DOI.

## Check SCF convergence

The SCF energy-convergence threshold (`conv_thr`) and total-energy threshold
(`etot_conv_thr`) scale with atom count rather than being flat defaults;
density-mixing strength (`mixing_beta`) and the iteration limit
(`electron_maxstep`) are flat defaults carried over unchanged. None of these
are model predictions. Inspect the actual QE convergence history and adjust
these settings if needed. Reaching the SCF threshold does not demonstrate
convergence with respect to k-points or cutoffs.

The density-mixing mode (`mixing_mode`) is `local-TF`, Quantum ESPRESSO's
screening for highly inhomogeneous densities, for a cell with vacuum along any
lattice vector (a slab, a wire or a molecule in a box). Every other cell gets
QE's default `plain`, including bulk layered crystals such as graphite, which
classify as `2d` but have no vacuum.

Goldilocks writes no `mixing_fixed_ns` for +U inputs unless you set it, so
pw.x uses its own default of 0 and converges the Hubbard occupation matrix
together with the density. For the first `mixing_fixed_ns` iterations pw.x
holds that matrix at its starting value, and it can report convergence before
the window ends. The energy, forces and stress are then those of the starting
occupations, not a self-consistent DFT+U result, and the stress calculation
can abort with "non-symmetric stress contribution". A value you set is
written as given. On an input with a HUBBARD card, a nonzero value carries a
`convergence.mixing_fixed_ns_freezes_hubbard_occupations` warning; without a
card, pw.x ignores the keyword. Set a nonzero value only for a preparatory
run whose output seeds a separate run with the occupations free. If a +U SCF
needs more than `electron_maxstep` iterations, adjust `mixing_beta` or
`electron_maxstep` rather than freezing the occupations.

## Check relaxation settings

For `relax` and `vc-relax` tasks, Goldilocks also picks ionic and cell
relaxation parameters -- `ion_dynamics`, `cell_dofree`, `cell_factor`,
`forc_conv_thr`, `nstep`, `press`, `press_conv_thr`,
`trust_radius_ini`/`trust_radius_max`/`trust_radius_min`, `remove_rigid_rot`,
and `fix_bottom_layers` (fixing the bottom N atomic layers of a slab during
relaxation). Every default is copied verbatim from Quantum ESPRESSO's own
documentation, not derived from a model. See `goldilocks settings` for each
key's group, type, and default.

`fix_bottom_layers` counts atomic layers along the slab's vacuum axis, up from
the upper edge of the vacuum, so a slab that wraps across the cell boundary
keeps its true bottom; atoms within 0.5 Å of each other along the normal share
a layer. Goldilocks refuses the setting unless the cell to be written has
one vacuum gap, along exactly one lattice vector, and at least N layers. A
bulk crystal has no bottom layer to fix, even a layered one that classifies
as `2d`, and nor has a cell holding two slabs, such as a monolayer doubled
along its normal.

The `vc-relax` default for `cell_dofree` comes from the vacuum, not from
dimensionality. A cell without vacuum gets `'all'`, so the stacking axis of a
bulk layered crystal such as graphite relaxes too. `'2Dxy'` moves only the x
and y components of a and b and holds c fixed, so a slab gets it only when its
vacuum lies along c and a and b lie in the Cartesian xy plane. The `geometry`
record's `slab` field shows the vacuum axis and whether the slab is in that
frame (`xy_frame`). A wire, or a slab in any other orientation, has no
default: Goldilocks refuses the `vc-relax` until you set `cell_dofree` or
reorient the cell. Whatever its source, `'2Dxy'` or `'2Dshape'` is refused
unless a and b lie in the xy plane and any vacuum lies along c. Otherwise
pw.x would distort the cell, or move a vacuum vector while holding a vector
of the sheet fixed.

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
