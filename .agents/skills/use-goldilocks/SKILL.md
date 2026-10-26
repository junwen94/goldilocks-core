---
name: use-goldilocks
description: Use goldilocks-core quickly for DFT input recommendation workflows. Trigger when running or scripting recommendations, extracting parameters from structures, generating or publishing Quantum ESPRESSO inputs (SCF, DOS, band structure, relaxation), or manually writing inputs from Goldilocks-selected Records.
---

# Use Goldilocks

Use the canonical capabilities, structure inspection, explain and run
operations. Core prepares Quantum ESPRESSO inputs for five tasks:
`scf_single_point` (scf), `dos` (scf, nscf, `dos.x`), `band` (scf, `pw.x`
along a high-symmetry path, `bands.x`), `relax` and `vc-relax`.

Fresh installations must install the runtime assets first: run
`goldilocks assets install default` once, or pass `--fetch-missing`, before any
command that needs the default k-point model or pseudopotential tables.

## Progressive disclosure

- `references/workflows.md` — Python, CLI, HTTP, and MCP workflows.
- `references/qe-scf-template.md` — mechanical QE SCF template for manually
  writing an input from trusted Records.

Pair with `use-uv` for execution and `dft-basics` for physics-bearing choices.

## Mental model

```text
Load -> Analyze -> Advise
Load -> Kmesh
Load + Advice -> Select
Load + Advice + Select + Kmesh -> Generate
Analysis + Advice + Kmesh + Select + Generate -> DFT Input Data
```

Each task has five service functions, `advise`, `check`, `generate`,
`render_submission` and `to_bundle_input`, suffixed `_dos`, `_band` or
`_relax` for those tasks; `goldilocks_core.service.TASK_PHASES[task]` holds
them by task name.

## Inputs to identify

1. Structure Source: path, inline content, or `pymatgen.Structure`.
2. Calculation Intent: target code, Calculation Task, functional, and
   pseudopotential accuracy.
3. Calculation Hints: k-grid or spacing, smearing, spin/SOC, pseudopotential
   type, convergence, and dispersion.
4. Pseudopotential source: registered Set ID, local root, or explicit metadata.
5. Computation Selection and output target.

## Canonical entry points

```python
from goldilocks_core.inputs.hpc import load_hpc_profile
from goldilocks_core.inputs.structure import PathStructureSource, normalize_structure
from goldilocks_core.service import advise, check, generate

structure = normalize_structure(PathStructureSource("structure.cif")).structure
hpc = load_hpc_profile("scarf")

advice = advise(structure, hpc=hpc)
report = check(advice)
if report.ok:
    steps = generate(advice, report)
```

```bash
uv run goldilocks capabilities --json
uv run goldilocks inspect STRUCTURE --json
uv run goldilocks explain STRUCTURE --task band --json
uv run goldilocks run STRUCTURE --task band --out RUN_DIR --json
```

HTTP exposes `/capabilities`, `/inspect`, `/explain` and `/run`; local stdio
MCP exposes `capabilities`, `inspect_structure`, `explain` and `run`, each
request naming its `task`. Send inline structure content
and, when needed, a registered Pseudopotential Set ID. HTTP and MCP do not
accept structure paths, local pseudopotential sources, model locations, or
publication paths.

## Completion check

For prepared input data, report the Structure Source, functional, target
code/Task, k-grid and shift, pseudopotentials and cutoffs, smearing, spin/SOC,
convergence controls, for a `band` task the path (`bands_k_path`: its
`lattice_type`, `n_kpoints` and `primitive_cells`, above 1 for folded bands),
factual warnings, generated paths, and publication path.
Core prepares inputs; it does not run DFT.
