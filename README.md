# goldilocks-core

`goldilocks-core` is a research-grade Python package that turns a crystal structure and a calculation intent into structured DFT parameter recommendations. It does not write DFT input files itself — that is done downstream by the `gl` CLI.

## What It Does

The package runs a four-stage pipeline:

```
Load → Analyse → Advise → Generate
```

| Stage | What happens |
|-------|-------------|
| **Load** | Parse CIF / POSCAR / XSF into a `pymatgen.Structure` |
| **Analyse** | Extract material facts: metallicity, magnetic elements, SOC relevance, dimensionality, disorder |
| **Advise** | Combine analysis + intent + hints → parameter decisions with provenance tags (`heuristic` / `ML` / `user_hint`) |
| **Generate** | Write QE `pw.x` / `ph.x` input files using ASE |

Currently supports Quantum ESPRESSO only. The pipeline architecture is code-agnostic from the Advise stage down.

## Quick Start

### Interactive wizard

```bash
uv run gl
```

The wizard walks through Pre-Analysis → Input Kit → file generation → HPC script → output parsing.

### Non-interactive command

```bash
uv run gl input -s Fe.cif
uv run gl input -s Fe.cif -t relax -a accurate -e
uv run gl input -s Fe.cif -H spin_treatment=collinear -H ecutwfc_ev=680
```

### Generate files

```bash
uv run gl input -s Fe.cif --output ./run
```

Writes `run/run_001/gl-pw-scf.in` and copies pseudopotentials.

### Python API

```python
from goldilocks_core.io.structures import load_structure
from goldilocks_core.analyse.structure import analyze_structure
from goldilocks_core.intent import CalculationIntent
from goldilocks_core.advise.pipeline import advise
from goldilocks_core.select.qe import build_qe_parameter_set

structure = load_structure("Fe.cif")
analysis = analyze_structure(structure)
intent = CalculationIntent(
    structure=structure,
    code="qe",
    task="scf",
    xc="pbesol",
    pseudo_family="PseudoDojo/0.4/PBEsol/SR/standard/upf",
    accuracy="balanced",
)
bundle = advise(analysis, intent)
params = build_qe_parameter_set(bundle)
print(params.kpoints_grid, params.ecutwfc, params.nspin)
```

## Installation

This project uses `uv` for dependency management.

```bash
git clone https://github.com/stfc/goldilocks-core
cd goldilocks-core
uv sync --group dev
```

Optional extras:

```bash
uv sync --extra kpoints-ml   # CGCNN/QRF k-points ML backend
uv sync --extra mlip         # MACE MLIP pre-relaxation
uv sync --extra aiida        # AiiDA workflow submission
```

## Package Layout

```text
src/goldilocks_core/
├── intent.py          # CalculationIntent — shared input to all pipeline stages
├── kmesh.py           # k-spacing ↔ mesh maths (🔒 stable API, used by goldilocks-models)
├── io/                # Stage 1: Load — structure file parsing
│   ├── structures.py  #   load_structure(), StructureInput type
│   └── db_search.py   #   OPTIMADE database search (Materials Cloud, NOMAD)
├── analyse/           # Stage 2: Analyse — material facts (no parameter recommendations)
│   └── structure.py   #   analyze_structure() → StructureAnalysis
├── advise/            # Stage 3: Advise — code-agnostic parameter decisions → AdviceBundle
│   ├── pipeline.py    #   main orchestrator → AdviceBundle
│   ├── smearing.py    #   metallicity → smearing method / width
│   ├── kpoints.py     #   dimensionality + accuracy → k-grid
│   ├── spin.py        #   magnetic elements → nspin / noncolin / initial moments
│   ├── pseudo.py      #   SOC relevance → SR/FR pseudo family + cutoffs
│   ├── basis.py       #   accuracy → ecutwfc / ecutrho
│   ├── vdw.py         #   dimensionality → vdW correction
│   ├── phonon.py      #   q-grid / k-grid / convergence for ph.x
│   ├── protocol.py    #   accuracy tier → sampling protocol (smearing width + k-spacing)
│   └── types.py       #   AdviceBundle, SmearingDecision, KPointsDecision, … QEParameterSet
├── select/            # Stage 4: Select — translate AdviceBundle → code-specific values
│   └── qe.py          #   build_qe_parameter_set(bundle) → QEParameterSet
├── generate/          # Stage 5: Generate — write QE input files
│   └── qe.py          #   write_qe_inputs(params, structure, intent, output_dir)
├── data/              # Bundled assets (pseudopotentials + ML model artefacts)
│   ├── pseudopotentials/  #   PseudoDojo 0.4 UPF files
│   └── models/            #   kpoints / metallicity / magnetic_classifier manifests
├── pseudo/            # UPF parsing and local pseudo registry / policy
│   ├── parse_upf.py   #   parse_upf_metadata(), parse_upf_folders()
│   ├── registry.py    #   load_pseudo_metadata(), filter_by_*()
│   ├── policy.py      #   PseudoPolicy, apply_pseudo_policy()
│   └── metadata.py    #   PseudoMetadata dataclass
├── ml/                # ML backends (k-index predictor, magnetic classifier)
│   ├── types.py       #   StructureFeatureVector
│   ├── features.py    #   infer_features() — 🔒 stable API, used by goldilocks-models
│   ├── models.py      #   load_model() from manifest directory
│   ├── inference.py   #   predict() → float k_index
│   ├── loader.py      #   try_load_kpoints_predictor(), try_load_magnetic_classifier()
│   ├── magnetic.py    #   MagneticClassifier (requires goldilocks[mlip])
│   └── kpoints/       #   Advanced ML backends (CGCNN + QRF, ALIGNN)
├── results/           # Results Lab — parse, validate, and plot DFT output
│   ├── types.py       #   SCFResult, RelaxResult, BandResult, ValidationReport
│   ├── check.py       #   validate(result, manifest_path) → ValidationReport
│   ├── plot.py        #   plot_bands(), plot_dos() → PNG
│   └── local/
│       └── qe.py      #   parse_scf(), parse_relax() from pw.x output files
├── aiida/             # AiiDA workflow integration (goldilocks[aiida])
│   ├── convert.py     #   QEParameterSet → AiiDA input dicts (pure Python)
│   ├── qe.py          #   build_pw_inputs(), build_relax_inputs()
│   ├── pseudo.py      #   aiida-pseudo family lookup
│   ├── submit.py      #   submit_pw() → (pk, uuid)
│   ├── status.py      #   get_status(), is_finished()
│   └── results.py     #   load_scf_result(), load_relax_result()
├── mlip/              # MLIP pre-analysis (goldilocks[mlip], MACE-based)
│   ├── types.py       #   MLIPPreview dataclass (importable without mace-torch)
│   ├── preview.py     #   run_mlip_prep() → MLIPPreview
│   ├── relax.py       #   relax_structure() using MACE + ASE BFGS
│   └── phonon.py      #   check_phonon_stability() using phonopy
└── cli/               # Thin Typer / Rich adapters
    ├── main.py        #   app entry point (gl)
    ├── commands/      #   gl input (non-interactive)
    └── wizard/        #   gl (interactive wizard)
```

Every parameter decision in `advise/` carries a `provenance` field (`heuristic` / `ML` / `user_hint`) and a `rationale` string. Pass `--explain` to `gl input` to see all rationales.

## CLI at a Glance

```
gl
 1) Pre-Analysis      — load structure, database lookup, material analysis
 2) Input Kit         — AI-guided QE parameter recommendation + file generation
 3) HPC Playground    — detect scheduler/QE, generate PBS/SLURM scripts, QE install guide
 5) Parse & Validate  — parse QE output files, show convergence / energy / magnetisation
 6) Visualise         — plot DOS and band structure from QE output files
 0) Quit
```

See [docs/cli.md](docs/cli.md) for the full reference.

## Hints Reference

Hints override specific advise-stage decisions. Pass with `-H key=value` (non-interactive) or enter in the wizard.

| Key | Example value | Effect |
|-----|--------------|--------|
| `spin_treatment` | `collinear` | Override spin/SOC treatment |
| `ecutwfc_ev` | `680` | Wavefunction cutoff (eV) |
| `ecutrho_ev` | `5440` | Charge density cutoff (eV) |
| `initial_magnetization` | `Fe:3.0,Ni:1.5` | Starting magnetic moments (μB) |
| `use_vdw` | `true` | Enable vdW correction |
| `vdw_method` | `d3bj` | vdW method: `d3` / `d3bj` / `ts` / `mbd` |

## Development

```bash
uv run pytest
uv run pre-commit run --all-files
uv run mkdocs serve        # local docs preview
```

The test suite runs without any private local data. Tests that require local pseudo libraries are marked `skip` with a clear message.

## Ecosystem

`goldilocks-core` sits in a four-repo pipeline:

```
goldilocks-data → goldilocks-models → goldilocks-core → goldilocks-webapp
```

`goldilocks-models` imports two stable APIs from this package:

- `goldilocks_core.kmesh.build_kmesh_entries` — k-index schedule
- `goldilocks_core.infer_features` — structure feature extraction

These must not change without retraining the models.
