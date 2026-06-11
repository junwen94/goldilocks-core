# CLI Reference

## Installation

```bash
uv sync --group dev
```

After that, `gl` is available via `uv run gl ...`.

---

## `gl` — interactive wizard

Running `gl` with no arguments opens the interactive menu:

```
╔══════════════════════════════════════════╗
║          goldilocks  ·  wizard           ║
╠══════════════════════════════════════════╣
║  Pre-Analysis                            ║
║    1) Structure analysis & DB lookup     ║
║    2) Input Kit — AI parameter guide     ║
║                                          ║
║  HPC Playground                          ║
║    3) HPC scripts & QE install guide     ║
║    4) AiiDA workflow        coming soon  ║
║                                          ║
║  Post-Analysis Lab                       ║
║    5) Parse & Validate QE output         ║
║    6) Visualise DOS / bands              ║
║                                          ║
║    0) Quit                               ║
╚══════════════════════════════════════════╝
```

### Option 1 — Pre-Analysis

First asks how to source the structure:

- **Load local file** — load and analyse a CIF / POSCAR / XSF file directly.
- **Search database** — query Materials Cloud (MC3D) and NOMAD in parallel via their public OPTIMADE endpoints. Displays a results table with entry IDs and clickable URLs. No API key required.

Displays a full `StructureAnalysis`: formula, space group, crystal system, metallicity, magnetic elements, SOC relevance, dimensionality, disorder warnings.

### Option 2 — Input Kit

Optionally preceded by Pre-Analysis if no structure has been loaded yet. Prompts for:

- Calculation task (default: `scf`)
- Accuracy tier (default: `balanced`)
- Optional parameter hints (`key=value`)

Runs the advise pipeline and displays the full QE parameter recommendation table, then optionally generates input files.

### Option 3 — HPC Playground

- Auto-detects the HPC scheduler (SLURM / PBS) and available partitions / queues.
- Auto-detects `pw.x` installation (module system or PATH).
- Generates a ready-to-submit SLURM or PBS script with all detected defaults pre-filled.
- If `pw.x` is not found, offers to generate a tailored **QE installation guide** (Markdown). The guide detects the system (ARCHER2, Cirrus, SCARF, JADE2, …) and provides EasyBuild / Spack / conda / container instructions in priority order.

### Option 4 — AiiDA workflow

Coming soon. Will require `goldilocks[aiida]`.

### Option 5 — Parse & Validate

- First asks which DFT code produced the output (currently QE only).
- Scans a directory for `pw.x` output files.
- Extracts and displays: convergence status, total energy (Ry and eV), Fermi energy, total magnetisation, iteration count, max force.
- If a `goldilocks_manifest.json` is present in the directory (written by Input Kit), also checks for metallicity / magnetisation consistency between the recommendation and the DFT result.

### Option 6 — Visualise

- First asks which DFT code produced the output (currently QE only).
- Scans a directory for DOS (`.dos` / `pwscf.dos`) or bands (`.dat.gnu`) files.
- Plots DOS (shifted to Fermi level) or band structure and saves PNG files.

---

## `gl input` — non-interactive command

Runs the full pipeline from a single command. Useful for scripting and reproducible workflows.

```
gl input [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--structure` | `-s` | **required** | Structure file (CIF, POSCAR, XSF, …) |
| `--task` | `-t` | `scf` | Calculation task: `scf` \| `relax` \| `vc-relax` \| `nscf` \| `bands` \| `md` \| `vc-md` \| `ph` |
| `--accuracy` | `-a` | `balanced` | Accuracy tier: `fast` \| `balanced` \| `accurate` |
| `--code` | `-c` | `qe` | DFT code (currently only `qe`) |
| `--xc` | | `pbesol` | XC functional (currently only `pbesol`) |
| `--pseudo` / `--pp` | | PseudoDojo SR | Pseudo family. Defaults to SR; auto-upgrades to FR for SOC structures when not set explicitly. |
| `--hints` | `-H` | | Parameter overrides as `key=value`, repeatable |
| `--explain` | `-e` | off | Print full rationale for every decision |
| `--output` | `-o` | | Directory to write QE input files; creates `run_NNN/` subdirectory |

### Examples

Minimal — just the structure file:

```bash
gl input -s Fe.cif
```

Relaxation with accurate settings and full rationale:

```bash
gl input -s Fe.cif -t relax -a accurate -e
```

Force fully-relativistic pseudopotentials:

```bash
gl input -s FeBi.cif --pseudo PseudoDojo/0.4/PBEsol/FR/standard/upf
```

Override spin treatment and wavefunction cutoff:

```bash
gl input -s Fe.cif -H spin_treatment=collinear -H ecutwfc_ev=680
```

Generate QE input files:

```bash
gl input -s Fe.cif --output ./run
```

Phonon calculation (writes `gl-pw-scf.in` + `gl-ph.in`):

```bash
gl input -s Si.cif -t ph --output ./ph_run
```

---

## Pseudo families

| Family string | Relativistic | Notes |
|--------------|-------------|-------|
| `PseudoDojo/0.4/PBEsol/SR/standard/upf` | Scalar | Default |
| `PseudoDojo/0.4/PBEsol/FR/standard/upf` | Fully relativistic | Auto-selected when SOC is relevant |

When `--pseudo` is not set, the advise layer picks SR by default and upgrades to FR automatically if the structure contains heavy elements (period 5+).

---

## Hints reference

Hints override specific parameters in the advise pipeline. Pass with `-H key=value`.

| Key | Values / format | Effect |
|-----|----------------|--------|
| `spin_treatment` | `non_magnetic` \| `collinear` \| `non_collinear` \| `non_collinear_soc` | Override spin / SOC treatment |
| `ecutwfc_ev` | float (eV) | Override wavefunction cutoff |
| `ecutrho_ev` | float (eV) | Override charge density cutoff |
| `initial_magnetization` | `Fe:3.0` or `Fe:3.0,Ni:1.5` | Starting magnetic moments (μB) |
| `use_vdw` | `true` \| `false` | Enable / disable vdW correction |
| `vdw_method` | `d3` \| `d3bj` \| `ts` \| `mbd` | vdW correction method |

Examples:

```bash
gl input -s Fe.cif -H spin_treatment=collinear -H ecutwfc_ev=680
gl input -s FeNi.cif -H initial_magnetization="Fe:3.0,Ni:1.5"
gl input -s MoS2.cif -H use_vdw=true -H vdw_method=d3bj
```
