# Quantum ESPRESSO SCF Template

Use this only after Goldilocks has already selected the scientific values. The generator should not invent missing pseudos, cutoffs, smearing, spin, SOC, or convergence settings.

```text
&CONTROL
  calculation = 'scf'
  prefix = '<prefix>'
  pseudo_dir = './pseudo'
  outdir = './out'
  tstress = .true.
  tprnfor = .true.
/

&SYSTEM
  ibrav = 0
  nat = <site_count>
  ntyp = <species_count>
  ecutwfc = <max_selected_ecutwfc_ry>
  ecutrho = <max_selected_ecutrho_ry>
  occupations = '<fixed_or_smearing>'
  smearing = '<smearing_type>'        ! omit if fixed occupations
  degauss = <smearing_width_ry>       ! omit if fixed occupations
  nspin = 2                           ! only for collinear spin-polarized non-SOC
  noncolin = .true.                   ! only when SOC/noncollinear is enabled
  lspinorb = .true.                   ! only when SOC is enabled
/

&ELECTRONS
  conv_thr = <conv_thr>
  mixing_beta = <mixing_beta>
  electron_maxstep = <electron_maxstep>
/

ATOMIC_SPECIES
  <species_label>  <atomic_mass>  <selected_pseudo_filename>

CELL_PARAMETERS angstrom
  <a_x>  <a_y>  <a_z>
  <b_x>  <b_y>  <b_z>
  <c_x>  <c_y>  <c_z>

ATOMIC_POSITIONS crystal
  <species_label>  <f_x>  <f_y>  <f_z>

K_POINTS automatic
  <nk1>  <nk2>  <nk3>  <s1>  <s2>  <s3>
```

`<species_label>` is the QE species name (e.g. `Fe1`/`Fe2` for an
AFM-relabeled structure), which is the real element (`Fe`) except when
`magnetic_ordering="afm"` split one element into several oppositely-spinned
species. `write_qe_scf` (`generation/quantum_espresso/scf.py`) already
handles this distinction -- this template documents what it produces, it is
not something to reimplement by hand.

## Mapping from Goldilocks records

```text
site_count          -> len(structure)
species_count       -> len({site.label for site in structure})
k-grid              -> advice.step.kpoints.k_sampling.value.mesh
k-shift             -> advice.step.kpoints.k_sampling.value.shift
pseudos             -> advice.system.pseudo.metadata.value  (one entry per element)
ecutwfc / ecutrho   -> advice.system.cutoffs.value (max across elements)
occupations         -> advice.step.kpoints.occupations.value
spin / SOC          -> advice.system.magnetic.value, advice.system.pseudo.relativistic
convergence         -> advice.step.kpoints.convergence.value
warnings            -> advice.warnings()
```

## Minimal Python extraction pattern

```python
from pymatgen.core.periodic_table import Element

from goldilocks_core.examples.structures import structure
from goldilocks_core.inputs.hpc import load_hpc_profile
from goldilocks_core.inputs.structure import PathStructureSource, normalize_structure
from goldilocks_core.resolution import Resolved
from goldilocks_core.service import advise

source = PathStructureSource(structure("Si.cif"))
silicon = normalize_structure(source).structure
hpc = load_hpc_profile("scarf")

advice = advise(silicon, hpc=hpc)

pseudo_metadata = advice.system.pseudo.metadata
assert isinstance(pseudo_metadata, Resolved)
pseudo_by_element = {pseudo.element: pseudo for pseudo in pseudo_metadata.value}
elements = tuple(sorted(pseudo_by_element))

cutoffs = advice.system.cutoffs
assert isinstance(cutoffs, Resolved)
ecutwfc = cutoffs.value.ecutwfc_ry
ecutrho = cutoffs.value.ecutrho_ry

k_sampling = advice.step.kpoints.k_sampling
assert isinstance(k_sampling, Resolved)
grid = k_sampling.value.mesh
shift = k_sampling.value.shift

for element in elements:
    pseudo = pseudo_by_element[element]
    print(element, float(Element(element).atomic_mass), pseudo.filename)
```

`advice.system.pseudo.metadata` is `Unavailable`/`Blocked`, not `Resolved`,
if pseudopotential selection failed for any element -- check `.ok` (or use
`check(advice)`/`report.blocking`, see [workflows.md](workflows.md)) before
assuming a runnable input can be generated. Do not proceed to a runnable
input if any selected pseudopotential's `filename`, `ecutwfc_ry`, or
`ecutrho_ry` is `None`.
