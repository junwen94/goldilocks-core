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
override, the ML classifier, or an LLM override. Neither the heuristic nor
the ML model determines a band structure.

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
`records.is_metal`), and the `occupations` record's own `warnings` array. A
composition-only `metal` result can still be wrong -- intermetallic
semiconductors such as CsAu or Mg2Sn contain only metals -- and give the
wrong occupation choice for your system. Override the scheme itself with
`occupations` (`fixed`, `smearing`, or `tetrahedra_opt`); when using
smearing, refine it with `smearing_type` (default `cold`) and `degauss` (Ry,
default 0.01). Test the width together with the k-point mesh.

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
compute). There is no "confidence" or prediction-interval field
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
`spin_polarized`.

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
and the [table restrictions](pseudopotentials.md#choose-a-table).

A spin-polarized run with SOC is noncollinear: every moment starts along one
direction, and that direction is a physical choice, since magnetic
anisotropy, orbital moments and band splittings depend on it. Set
`moment_direction` to a lattice direction [u, v, w] of the uploaded cell
(`--set 'moment_direction=[1,1,0]'`). Unset, the moments start along
[0, 0, 1], the uploaded cell's third lattice vector, with the info notice
`magnetic.soc_moment_direction_defaulted`; for a CIF that is the c axis,
which pymatgen puts along Cartesian z, QE's own default. The direction is a
crystal direction, so it keeps its meaning whatever cell is written, an
antiferromagnetic one included. `records.magnetic.moment_direction` shows it
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
requires it; the `geometry` record's `slab` field locates the vacuum
(`vacuum_center`, `vacuum_fraction`). The notice offers `isolation='slab'`
only when the slab meets every precondition of that setting (the `slab`
field's `fits_2d_truncation`). It does not set the
correction itself, because pw.x then uses no symmetry and computes no stress.
The notice describes the cell that is written, which an antiferromagnetic
`magnetic_ordering` writes with the uploaded vacuum (see
[magnetism](#check-magnetism-and-spin-orbit-coupling)).
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
