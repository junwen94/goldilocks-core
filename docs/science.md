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
| `metal`, or unavailable (composition couldn't decide)  | Smearing (`smearing_type='cold'`, `degauss=0.01` Ry) |
| `non_metal`, where fixed occupations can run (below)  | Fixed occupations, without smearing               |
| `non_metal`, otherwise                                 | Smearing, as for a metal                          |

Ahead of the heuristic, a published CGCNN classifier (installed by default as
`models/is-metal-classifier`) can resolve `is_metal` directly from structure;
`is_metal` only falls back to the heuristic below when that model asset is
not installed or fails verification. If the installed model fails to run,
the request fails rather than silently using the heuristic. The heuristic
itself returns `metal` only when every element in the composition is a metal
according to pymatgen (elemental metals and intermetallics). Any other
element (O, N, the halogens, H, metalloids such as Si) leaves `is_metal`
unavailable, naming those elements: whether such a compound conducts depends
on its band filling, which composition cannot settle, so insulating NaCl and
MgO look the same to it as metallic RuO2 and TiN. Unavailable -- like an
actual `metal` result -- still defaults to smearing, not fixed occupations,
because assuming "fixed" on an undetected metal risks silent
non-convergence. A confident `non_metal` result is reachable via a human
override (`--set is_metal=non_metal`), the ML classifier, or an LLM override.
Neither the heuristic nor the ML model determines a band structure.

Fixed occupations fill whole bands in each spin channel: pw.x puts
(nelec + `tot_magnetization`)/2 electrons in the spin-up channel and
(nelec - `tot_magnetization`)/2 in the spin-down one. A non-metal therefore
gets fixed occupations only where both are whole numbers: a run without spin
polarization and an even electron count, with or without spin-orbit coupling
(without moments, time reversal pairs every spin-orbit band with a degenerate
partner, so an odd count half-fills one); a spin-polarized spin-orbit run,
which has no `tot_magnetization` and fills single-electron bands; or a collinear
spin-polarized run with an integer `tot_magnetization` of the electron
count's parity, either one you set or 0 when the applied `magnetic_ordering`
is compensated by symmetry (see below). Any other non-metal gets smearing.
LiFePO4, or NiO classified as a non-metal, therefore runs spin-polarized on
smearing instead of being refused. The occupations record's
`tot_magnetization` is the moment the input pins: the one you set, with any
occupations, or the derived 0 under fixed occupations. If you set
`occupations` to `fixed` on a collinear spin-polarized run with no such
moment, it gets the warning
`occupations.fixed_spin_polarized_needs_integer_tot_magnetization` and is
refused before generation: set `tot_magnetization` too, or use smearing.

Check the `is_metal` record's `status`, `source`, and `reason` (via
`goldilocks explain --json` or the published `goldilocks.json`'s
`records.is_metal`), and the `occupations` record's own `warnings` array.
When the classifier decided (`source` `ml`), `model.details` holds its
`score` and the `threshold` it was compared with; a score close to the
threshold is a call the model nearly made the other way. A
composition-only `metal` result can still be wrong -- intermetallic
semiconductors such as CsAu or Mg2Sn contain only metals -- and give the
wrong occupation choice for your system. Correct the classification with
`is_metal` (`metal` or `non_metal`, the record's own values; `true`/`false`
are refused), or override the scheme itself with
`occupations` (`fixed`, `smearing`, or `tetrahedra_opt`); when using
smearing, refine it with `smearing_type` (`gaussian`, `mp` for
Methfessel-Paxton, `cold` for Marzari-Vanderbilt, or `fd` for Fermi-Dirac;
default `cold`) and `degauss` (Ry, default 0.01). Test the width together
with the k-point mesh.

## Check k-point sampling

K-points sample the Brillouin zone for reciprocal-space integration. Too sparse
a mesh can leave energies, forces, or electronic properties unconverged.

Without a grid or distance hint, Goldilocks first asks a published
machine-learning k-point model (`qrf-kpoints`, installed by default) to
predict a raw `k_distance` for the structure (if the installed model fails to
run, the request fails); only when that model asset is not installed or fails
verification does it fall back to picking a target k-point spacing
heuristically: 0.15 Å⁻¹ if `is_metal`
resolves to `metal` or is unavailable, 0.30 Å⁻¹ for a `non_metal` (a mesh
that is too coarse on an undetected metal risks silent under-convergence,
while a mesh that is too fine on an actual insulator only costs more
compute). When the model picked the distance, the `k_sampling` record's
`model` keeps QRF95's own prediction interval in `details.interval`. The
published model records no conformal calibration, so a
`k_sampling.model_warning` says that interval carries no coverage guarantee.
`k_distance_interval` is not that interval: it is the range of k-distances
that give the chosen ladder rung.

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

The mesh is one the crystal's point group maps onto itself: lattice vectors
that a symmetry operation mixes get the same number of k-points, the larger
one, so no direction is sampled more coarsely than `k_distance` asks and
pw.x can reduce the mesh by every operation it keeps. Meshing each vector
from its own length broke this on cells whose mixed vectors differ in length.
Alpha-U as the Materials Project serves it (the reduced primitive cell of its
base-centred lattice) got 11×10×6, which pw.x ran with 224 irreducible
k-points where the denser 11×11×6 has 144. CIF noise in the seventh
significant figure between the three equal vectors of UN's primitive cell made
`k_index` 33 a 32×33×32 mesh. Equal counts can cost irreducible k-points at
some spacings and save them at most others: on La2CuO4's body-centred
primitive cell, 8×8×4 (48 irreducible k-points) becomes 8×8×8 (59) at
0.24 Å⁻¹, and 18×18×10 (800) becomes 18×18×18 (480) at 0.10 Å⁻¹. `k_index`
numbers the distinct meshes of this kind from the Γ-point down to 0.03 Å⁻¹,
so its rungs are numbered differently from goldilocks-data's k-index ladder
wherever mixed vectors differ in length.

Compare results on denser meshes. Use `k_grid` or `k_distance` to override
the default; an explicit `k_grid` is written as given, vacuum axes included,
even where the point group does not map it onto itself.
Setting either one only affects k-point selection; it does not change the
separate `is_metal` heuristic used elsewhere (e.g. by occupations).

## Check the written cell

Quantum ESPRESSO does not search a cell for its symmetry. pw.x tests 32 fixed
Cartesian rotations, the 24 of a cube with its axes along x, y and z and 8 for
a hexagonal axis along z, and keeps those that map the cell as written onto
itself within 1e-5 in crystal coordinates, with a fractional translation of 0
or ±1/n (n = 2, 3, 4 or 6) only, and none at all in a cell that is a supercell
of a smaller one. A cell in pymatgen's frame (c along z, a in the xz plane)
shows it only part of its point group: diamond Si's primitive cell shows 4 of
its 48 operations, so pw.x ran 282 k-points of a 10×10×10 mesh where the full
group gives 47, and `npool` and the memory estimate were sized for 47.

Goldilocks therefore writes the cell in the frame where pw.x keeps the most of
its point group, which is all of it except possibly for the sheets, the cells
that keep their orientation and the supercells described below, and counts
`n_irr_k` with pw.x's rules on the cell it writes. The cell is the uploaded one
(or the cell of an antiferromagnetic ordering), idealized to the space group
spglib finds within 1e-3 Å, which moves atoms of a CIF by about 1e-5 Å, and:

- a cell without vacuum is turned rigidly into its standard orientation (cubic
  axes along x, y and z; a hexagonal a along x and c along z) when pw.x would
  miss operations otherwise, as it does for fcc, bcc and rhombohedral
  primitive cells;
- a slab with its vacuum along c is turned rigidly so that a lies along +x,
  a and b lie in the xy plane and c points to positive z, the frame Quantum
  ESPRESSO's 2D cell constraint and 2D Coulomb truncation assume. There pw.x
  keeps an in-plane 2-fold axis or mirror only where it lies at a multiple of
  30° or 45° from x, so a sheet whose in-plane axes lie elsewhere relative to a
  loses them: a centred-rectangular sheet given in its primitive basis (a = b,
  γ not 60°, 90° or 120°) has them along a ± b and keeps half its operations,
  and `n_irr_k` counts the extra k-points pw.x then runs. A slab with its
  vacuum along a or b, a wire and a molecule in a box keep their orientation;
- the origin moves to a symmetry centre when an off-centre mirror or screw
  axis would give pw.x a fractional translation it rejects, as on the CeTe3
  monolayer, preferring the shift that keeps a slab in one piece inside the
  cell.

The lattice vectors are the uploaded ones, turned, so fractional coordinates
(up to the origin shift), the k-point grid, `k_index`, species labels,
`starting_magnetization`, Hubbard labels, `fix_bottom_layers` and
`moment_direction` keep their meaning, and the energy does not change. The
input keeps `ibrav = 0` with an explicit `CELL_PARAMETERS` card.
`records.magnetic.frame` records the rotation (`rotation`, R with written
lattice vectors = uploaded ones × Rᵀ), the origin shift, the largest
displacement and the number of operations pw.x keeps. A turn that moves no
lattice vector by more than 1e-3 Å only undoes a tilt the idealization gave
the cell; it is recorded as no rotation and counted in the displacement. The
info notice `magnetic.cell_framed` names the rotation, shift and
displacement when the cell was turned, its origin moved, or an atom moved by
1e-4 Å or more.
The `geometry` record describes the uploaded cell. The dipole notice and the
`vc-relax` default for `cell_dofree` read the vacuum and frame of the written
cell, so a slab uploaded tilted shows `xy_frame` false in that record and is
still written, and relaxed with `'2Dxy'`, in the xy frame.
A `cell_dofree` that holds or frees Cartesian components (`a`, `b`, `c`, `x`,
`y`, `z`, `xy`, `xz`, `yz`, `xyz`) is refused on a turned cell, whose x, y and
z are not the uploaded cell's; `all`, `shape` and `fixa`, `fixb` or `fixc`
keep their meaning.

A supercell, such as the 8-atom conventional cell of diamond, gets no
fractional translations in pw.x, so it keeps only the operations that need
none (24 of diamond's 48), and is counted that way.

The cards write lattice components and fractional coordinates to 10
significant digits, and any smaller than 1e-10 in magnitude as `0`: the
1e-16 Å left where a lattice built from its angles, or turned into its
standard orientation, has a zero, and the 1e-36 coordinates some CIFs list.

## Check pseudopotentials and cutoffs

A pseudopotential replaces the core-electron potential with an effective
potential for the valence electrons. Check that its functional, valence
configuration, and relativistic treatment suit your system.

Goldilocks selects files and cutoff metadata, not system-specific convergence
limits. Quantum ESPRESSO receives the largest wavefunction cutoff and largest
charge-density cutoff among the selected elements. PseudoDojo publishes no
charge-density cutoff, so for its tables `ecutrho` is 4 times `ecutwfc` (the
table's `charge_density_dual`: Quantum ESPRESSO's default, which its
documentation advises keeping for norm-conserving files), flagged by the
`cutoffs.ecutrho_derived_from_dual` notice. An `efficiency` or `precision`
table is a library choice, not proof that your target property is converged.
Test both cutoffs. See [Pseudopotential tables](pseudopotentials.md) for
selection rules, installation, and custom files.

## Check magnetism and spin-orbit coupling

Spin polarization allows different spin populations, decided by `is_magnetic`
(`human > ml > llm > heuristic`, see [CLI reference](cli.md#scientific-controls)).
Once its ML asset is installed, `mace`/`e3nn`/`sphericart` are on hand (a
manual install -- see the CLI reference), and `GOLDILOCKS_MACE_BACKBONE`
points at a downloaded mMACE checkpoint, a published mMACE-embedding
classifier answers directly (if it then fails to run, the request fails).
Otherwise the heuristic tier answers, and it calls a structure magnetic when
an element carries a local d or f moment or the structure holds an open-shell
radical:

- **Ionic compounds** (an O, S, Se, Te, F, Cl, Br, or I anion is present): the
  first oxidation-state guess from the composition (pymatgen's
  `Composition.oxi_state_guesses`) gives each transition metal its d count
  (group minus oxidation state) and each lanthanide or actinide its f count. A
  shell strictly between empty and full carries a moment: Ni2+ (d8), Fe3+
  (d5), and Gd3+ (f7) do; Zn2+ (d10), Ti4+ (d0), La3+, Ce4+, and Th4+ (f0),
  and Lu3+ (f14) do not. When pymatgen finds no guess (GdI2, Fe3GeTe2, NpO2),
  an element carries a moment if every one of its common oxidation states
  leaves the shell open. Otherwise the heuristic cannot decide: a composition
  those states cannot balance holds the element in another state.
- **Metals and other compounds** (elements, intermetallics, nitrides,
  carbides, borides, hydrides): oxidation states do not describe these, so Cr,
  Mn, Fe, Co, and Ni carry moments, as do Ce to Tm (not La, Yb, or Lu) and, in
  a compound such as UN, the actinides Pa to Md. Cu, the 4d and 5d metals,
  metallic TiN and WC, and the elemental actinide metals (alpha-U) do not.
- **Radicals**, in any structure: an open-shell main-group atom with no
  neighbour within 1.5 times the covalent-radius sum (atoms in a box), or an
  O2 or S2 molecule bonded only to its partner (solid O2).

When the heuristic cannot decide, Goldilocks defaults to spin-polarized with
the warning `magnetic.magnetism_unknown_defaulted_spin_polarized`. An SCF can
quench a moment it was seeded with, but a non-spin-polarized run can never
find one, so a wrong default costs compute time, not physics. The rules still
over-seed where a formally open shell keeps no moment (low-spin Co3+ in
LiCoO2, Mo4+ in MoS2), and a misguessed oxidation state can hide one (pymatgen
reads CrTe3 as d0 Cr6+): set `spin_polarized` when you know better. A
structure with fractional site occupancies is undecidable too, and since a
spin-polarized run needs one species per site to seed, its magnetic setup is
blocked until you set `spin_polarized` to false or upload an ordered cell.
Spin-polarized inputs get a `starting_magnetization` for every species;
review it for your calculation. Neither tier predicts magnetic *order* (see
`magnetic_ordering` below for that). Override either tier with
`spin_polarized`, or correct the classification with `is_magnetic`
(`magnetic` or `non_magnetic`).

Bands are counted per spin channel, and the default follows pw.x's own rule
(QE setup.f90), sized so the majority channel holds (nelec + |M|)/2 electrons.
M is `tot_magnetization` when the moment is pinned, as in pw.x. On a collinear
smearing run without one, where pw.x would split the electrons evenly, M is
the net moment the starting magnetization seeds (each species' fraction times
its pseudopotential's valence electrons, `records.magnetic.seeded_moment`),
so the default can exceed pw.x's own count. pw.x holds a smearing run's
moment at 2 × `nbnd` − nelec Bohr magnetons without an error: with pw.x's
default 22 bands, hcp Gd converged to 8 instead of 15.5 Bohr magnetons per
cell; Goldilocks gives it 28. An oversized seed only adds
bands. An `nbnd` or `extra_bands` that leaves fewer bands than pw.x accepts
-- the electrons of each spin channel, or one band per electron with
spin-orbit coupling -- is refused before generation, and one that holds the
moment below its seed gets the warning `nbnd.moment_capped`.

A `starting_magnetization` you set is keyed by site label and replaces the
heuristic seeds; labels you leave out start at 0. The input has one species
per element and starting value, whatever labels your file gave its sites
(`Si1`..`Si40` are one `Si`), and sites of one element with different values
become separate species (`Fe0` at 0.5 and `Fe1` at -0.5 are written as `Fe1`
and `Fe2`). Symmetry and the irreducible k-points are counted on those
species, the way pw.x counts them: that bcc Fe pair is simple cubic (Pm-3m),
not body-centred. `records.magnetic.starting_magnetization` is keyed by the
written species, not by your file's site labels.

Spin-orbit coupling (SOC) couples electron spin to orbital motion. Goldilocks
flags elements at or above Rb (Z=37) whose valence character is p, d, or f for
consideration, but does not enable SOC unless you request it -- heavy s-block
elements (Rb, Sr, Cs, Ba) are excluded because s orbitals carry no
first-order spin-orbit splitting. Whether SOC matters depends on the property,
not only the elements. Enabling it changes the required pseudopotentials and
the QE spin settings. See [relativistic modes](conventions.md#relativistic-modes)
and the [table restrictions](pseudopotentials.md#choose-a-table). The info
notice follows the table selection SOC would make. `magnetic.soc_suggested`
names the fully relativistic table automatic selection would use, for
example `pseudodojo-pbesol-efficiency-fr` for fcc Pb under PBEsol. Where no
registered table can serve SOC, the notice is `magnetic.soc_unavailable`
instead and carries the selector's reason. That covers every lanthanide and
actinide, which must use SSSP, and any element under LDA, which has no fully
relativistic table: enabling `spin_orbit_coupling` there would leave
`pseudo_table` unavailable.

A spin-polarized run with SOC is noncollinear: every moment starts along one
direction, and that direction is a physical choice, since magnetic
anisotropy, orbital moments and band splittings depend on it. Set
`moment_direction` to a lattice direction [u, v, w] of the uploaded cell
(`--set 'moment_direction=[1,1,0]'`). Unset, the moments start along
[0, 0, 1], the uploaded cell's third lattice vector, with the info notice
`magnetic.soc_moment_direction_defaulted`; for a CIF that is the c axis. The
direction is a crystal direction, so it keeps its meaning whatever cell is
written, an antiferromagnetic one or one turned into pw.x's frame (see
[the written cell](#check-the-written-cell)) included. It is QE's own
default, Cartesian z, only where c is written along z, which a cell turned
into pw.x's frame (an fcc, bcc or rhombohedral primitive cell) is not, so
`angle1` is not 0 there. `records.magnetic.moment_direction` shows it
and whether you set it, and the input carries it as `angle1`/`angle2`, the
same for every species (the sign stays in `starting_magnetization`). Any
other run ignores `moment_direction` with the warning
`magnetic.moment_direction_ignored`. The irreducible k-points of such a run
are counted the way pw.x counts them: with only the symmetry operations
that keep every moment, alone or combined with time reversal, and no
k -> -k pairing. Cubic B2 FeCo with moments along [001] keeps 16 of its 48
operations, so a 10x10x10 mesh has 126 irreducible k-points rather than
56, and `npool` and the memory estimate follow that count.

Set `magnetic_ordering` to `afm` to opt in to an antiferromagnetic
ordering instead of the ferromagnetic default, or to the label of one
candidate `goldilocks magnetic-orderings` lists (`afm-3`). Like
`tot_magnetization` and
`starting_magnetization`, any `magnetic_ordering`, `fm` included, asks for a
spin-polarized run, so it turns spin
polarization on even for a structure the classifier calls non-magnetic; an
explicit `spin_polarized` false wins and ignores it with the warning
`magnetic.ordering_ignored_spin_polarized_false`. Goldilocks cannot tell you
which magnetic ordering is the true ground state -- that needs comparing total
energies from several actual calculations -- so `afm` only gives one
reasonable, deterministic starting point (`afm-1`, the smallest and most
symmetric ordering found), for you to run and compare against
the ferromagnetic guess yourself. Only the sites the heuristic above says
carry a local d or f moment are ordered, up against down:
the Cu2+ ions of La2CuO4 but not its La3+ (4f0), the Fe of FePt but not its
Pt, whose moment Fe induces. Those sites are seeded up and down with the
calibrated value (5 Bohr magnetons over the pseudopotential's valence for
Ni, 5/18 with an 18-electron pseudopotential); every other species starts
at 0, as in QE's own antiferromagnetic examples, so the start carries no
net moment. `records.magnetic` records the label applied, and whether the
ordering is compensated: whether some symmetry operation of its cell maps
every spin onto an opposite one, so its net moment is zero by construction.
NiO's AFM-II cell is compensated. Spinel Fe3O4's R-3m candidate is not: it
has 3 Fe up and 3 down, but both tetrahedral Fe point down, so it is a
ferrimagnet. Nor is B2 FeCo's `afm-1`, Fe up and Co down in the uploaded
cell: its starting moments sum to zero, but no symmetry makes the two
elements' moments cancel. A compensated non-metal gets fixed occupations with
`tot_magnetization = 0`. Starting magnetizations you set on the applied
cell decide this in place of the heuristic signs.
A structure with no such site (closed-shell Si, or solid O2, whose moment
belongs to the O2 molecule) keeps the ferromagnetic default with the warning
`magnetic.afm_no_open_shell_sites`; to try an antiferromagnetic pattern
there, upload a cell whose site labels separate the sublattices and set
`starting_magnetization` per label. When the heuristic cannot decide which
sites carry a moment, the warning `magnetic.afm_ordering_unavailable` gives
its reason.
That ordering can need a different cell from the one you uploaded: NiO's
2-atom primitive cell becomes the 4-atom AFM-II cell, and its 8-atom
conventional cell is reduced to the same 4-atom one. The input is then
written for that cell, and everything counted for it (electrons, bands, the
convergence threshold, k-points and the job's memory) is counted on it;
`records.magnetic` names it by formula and atom count. A `nelec` you set
counts the electrons of the cell you uploaded, so it is refused together
with an ordering that changes the cell's size.
pymatgen keeps at most 16 of enumlib's orderings, smallest cells first;
goldilocks has it read enumlib's output in enumlib's own numbering, so the
same 16 are kept wherever the structure is enumerated again with the same
pymatgen-core, spglib and enumlib versions. They are numbered in one order
that depends on the cells alone -- smallest first, then most symmetric --
so a label means the same cell there too; a label
past the listed candidates is refused.
`records.magnetic.ordering` says why the cell was written (`criterion`:
`smallest_cell` for `afm`, `requested` for a label, `identity` when the
uploaded cell is kept) and lists the other candidates with as many sites
(`alternatives`); the `magnetic.afm_ordering_applied` notice names them too.
Smallest-and-most-symmetric is a convention, not a prediction: for Fe2O3 it
picks the Cr2O3-type order (`afm-1`, R3c), not hematite's own (`afm-3`, R-3),
so list the candidates and choose among them.
A slab, a wire or a molecule in a box only gets orderings that keep its
vacuum: a candidate is kept when its cell can be written with the uploaded
vacuum vectors (the `geometry` record's `vacuum_axes`) as cell vectors, and
it is written that way, each vacuum vector where it was and the other
vectors in the plane of the rest. enumlib also offers cells that repeat the structure across the
vacuum -- the GdI2 monolayer's first candidate doubled c, every sheet
ferromagnetic and its only antiparallel Gd 19.94 Å away across the vacuum
-- and those are dropped, so the monolayer gets the in-plane stripe. The
written cell then has the uploaded vacuum, and the slab advice and gates
(one k-point along the vacuum, `isolation`, `fix_bottom_layers`, the
dipole notice) describe one slab. A bulk van der Waals magnet has no vacuum
and keeps its interlayer orderings. When every candidate repeats across
the vacuum, `afm` keeps the ferromagnetic default with a warning saying so.
This needs the external `enumlib` executables (`enum.x` or `multienum.x`,
plus `makeStr.py`, `makestr.x` or `makeStr.x`) on `PATH`. Its cost grows with
the number of moment-bearing sites in your structure's primitive cell, which
is what enumlib orders, not with the atom count: 28-site LiFePO4 has 4 Fe.
Without enumlib, with more than 8 such sites, when enumlib finds no ordering
or a call runs past its 30 s limit, or if no antiferromagnetic ordering
exists for your structure, `afm` degrades to the ferromagnetic default with a warning,
and an `afm-<n>` label is refused with the same reason; any other enumeration
error fails the request. Magnetic moments your file carries
(a magCIF's) are not read; only the seeds above are. Enumeration runs in a
separate worker process, inside a private scratch directory, so it writes
nothing to the current directory and concurrent requests cannot disturb each
other.

`uv run goldilocks magnetic-orderings` lists every candidate this same
enumeration finds -- the ferromagnetic guess plus any antiferromagnetic
ones, by label -- for you to inspect or generate inputs
from with `magnetic_ordering=<label>` on the same structure, optionally
ranked by relaxing each on a frozen mMACE potential
energy surface (`--rank-with-mmace`) rather than picking the ferromagnetic
guess by default. It warns the same way when a structure with a moment gets
no antiferromagnetic candidate, and lists a closed-shell structure's
ferromagnetic guess alone without a warning. See
[Magnetic orderings](cli.md#magnetic-orderings) for the command and its
setup requirements.

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
requires it. The notice offers `isolation='slab'` only when the slab meets
every precondition of that setting. It does not set the
correction itself, because pw.x then uses no symmetry and computes no stress.
The notice describes the cell that is written, which an antiferromagnetic
`magnetic_ordering` writes with the uploaded vacuum (see
[magnetism](#check-magnetism-and-spin-orbit-coupling)), turned and with its
origin moved as [the written cell](#check-the-written-cell) describes. The
`geometry` record's `slab` field (`vacuum_center`, `vacuum_fraction`,
`fits_2d_truncation`) describes the uploaded cell, so it can differ from the
written one when the cell was turned or its origin moved.
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
vacuum lies along c and a and b lie in the Cartesian xy plane, the frame every
slab with its vacuum along c is written in (see
[the written cell](#check-the-written-cell)). The `geometry` record's `slab`
field shows the vacuum axis and whether the uploaded cell is in that frame
(`xy_frame`); the default is decided on the written cell. A wire, or a slab
with its vacuum along a or b, has no default:
Goldilocks refuses the `vc-relax` until you set `cell_dofree` or reorient the
cell. Whatever its source, `'2Dxy'` or `'2Dshape'` is refused
unless a and b lie in the xy plane and any vacuum lies along c. Otherwise
pw.x would distort the cell, or move a vacuum vector while holding a vector
of the sheet fixed.

`cell_dofree` accepts only the values pw.x honours under the `ibrav = 0` every
written input has. `'volume'` is refused, because pw.x stops on it unless
`ibrav = 1`. `'ibrav'` and the `'ibrav+...'` forms are refused too: under
`ibrav = 0` pw.x ignores the `ibrav` part and only warns that it has no
effect.

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

A decision that applies several settings (the magnetic decision,
occupations, k-point sampling, Hubbard +U, convergence, relaxation, the
`dos.x` settings) also carries `field_sources`, one source per setting it
applied, keyed by the setting's override name: `human` (or `llm`) for a value
that tier set, otherwise the tier that chose it. The record's `source` is the
highest of them, so `mixing_beta=0.3` alone makes `convergence` `human` while
`field_sources` keeps `conv_thr` `heuristic`, and `spin_orbit_coupling=true`
makes `magnetic` `human` while spin polarization still reads `heuristic`. A
setting the decision did not apply has no entry and cannot make the record
`human`: a `degauss` under fixed occupations, a `smearing_type` on a `dos`
task's `tetrahedra_opt` nscf step, or a `magnetic_ordering` that an explicit
`spin_polarized=false` ignores. Hubbard's switch is keyed `needs_correlation`,
its override field, though `--set` calls it `hubbard_needs_correlation`. A
pinned `pseudo_table_id` makes `pseudo_table` `human`. The other records that
apply several settings (`cutoffs`, `job`, `nbnd`, `parallelisation`, `vdw`)
carry one `source` and no `field_sources`: their `human` means the request set
at least one of their settings.

A record whose own rule followed another record's answer keeps the tier that
applied the rule, `heuristic`, and names that record in `derived_from`:
`magnetic` names `is_magnetic` when that fact decided spin polarization,
`occupations` and a heuristic `k_sampling` name `is_metal`, `hubbard` names
`needs_correlation` (or `functional` for a hybrid functional), an
automatically selected `pseudo_table` names `functional` and `magnetic`,
`relativistic` names `magnetic`, and `pseudopotentials` names
`pseudo_table`. A non-metal's `occupations` also names the records that
decided whether fixed occupations can run: `magnetic`, and `electron_count`
where its parity was tested. An `occupations` that pins a `tot_magnetization`
names `magnetic` too, whichever tier chose the occupations (a requested
`occupations` reads `human` and still names it): the moment is the request
that record admitted, or the 0 of its compensated ordering, which also names
`electron_count`, since that 0 needs an even count. The upstream record's
`source` is not copied: `occupations`
reads `heuristic` even when the classifier decided `is_metal` (`ml`), and you
follow `derived_from` to find that. `human` only ever means a value set in the
request: a `dos` task's nscf protocol values (`nscf_k_sampling`'s k_distance,
`nscf_occupations`) read `heuristic`, with `source_note` naming the protocol.
Per-decision warnings live inside that decision's own value (e.g.
`job.warnings`, `cutoffs.warnings`), and every run, whatever its task, also
carries one aggregated top-level `warnings` list: every record's warnings, in
record-name order, each naming the record it came from in `record`, then the
request's routing notices (a setting given without a step that reached several
steps), under the `record` `overrides`. A `dos` task's nscf step keeps its
own, under `nscf_`-prefixed records: a notice on `nscf_parallelisation` is
about the nscf step's npool, not the scf step's.

A resolved record also carries the rest of its provenance: `source_note`,
the advisor's rationale where it gives one (such as why a molecule gets a
Coulomb cutoff), and `citations`, what to cite for it. A record that a
machine-learning model decided, whole or in one setting (`source` or
`field_sources` `ml`: `is_metal`, `is_magnetic`, `k_sampling`'s
`k_distance`), names the model in `model`: `model_id` (goldilocks-ml's id and
feature contract, such as `metallicity.is_metal.cgcnn@crystal_graph.v1`),
`asset` (the installed release, such as `models/is-metal-classifier@1`),
`target_contract`, the model's own `value`, its `details` as returned (a
classifier's `score` and `threshold`, QRF95's `interval`) and its `warnings`.
Each model warning, such as the is_magnetic classifier's "has not been
externally validated", is also in the top-level `warnings` list as
`<record>.model_warning`. A bundle's `CITATIONS.md` and the manifest's
`citations` list every record's citations: the selected pseudopotential table
and every model that decided a record. `CITATIONS.md` also names the table's
licence file under `licences/`, which lists SSSP's per-family references. Read analysis and selection warnings
as well as advice; do not treat a `resolved` status as scientific validation.
