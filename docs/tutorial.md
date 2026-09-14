# Python API

Use Goldilocks to generate inputs from a script. First complete the
[installation and asset setup](quickstart.md#1-install). Run the examples in
order in the same Python session, or save them in a script and run
`uv run your_script.py`.

## Generate inputs

`advise()` diagnoses a structure against an HPC profile: it always returns,
degrading field-by-field rather than raising, so you can inspect what is
known before deciding whether to generate anything.

```python
from goldilocks_core.examples.structures import structure
from goldilocks_core.inputs.hpc import load_hpc_profile
from goldilocks_core.inputs.structure import PathStructureSource, normalize_structure
from goldilocks_core.service import advise

source = PathStructureSource(structure("Si.cif"))
silicon = normalize_structure(source).structure
hpc = load_hpc_profile("scarf")

advice = advise(silicon, hpc=hpc)
```

Replace `structure("Si.cif")` with a path to your own CIF or POSCAR.
`load_hpc_profile` loads a named profile from `inputs/profiles/`, the same
profiles `--hpc` names on the CLI (see the [CLI reference](cli.md)).
`advise()` runs every analysis/advisor stage (structure facts, functional
and pseudopotential selection, k-point sampling, resource sizing, ...) and
returns one `Advice` holding a `FieldState` per decision.

## Read a recommendation

Each field on `Advice` is a tri-state `Resolved`/`Unavailable`/`Blocked`
value, never a bare number:

```python
from goldilocks_core.resolution import Resolved

functional = advice.system.functional
if isinstance(functional, Resolved):
    print(functional.value, functional.source)

k_sampling = advice.step.kpoints.k_sampling
if isinstance(k_sampling, Resolved):
    print(k_sampling.value.mesh)
```

This prints the selected functional, the source tier that picked it
(`human`/`ml`/`llm`/`heuristic`), and the k-point mesh. Model-dependent
values can change with the installed model; review warnings and check
convergence rather than treating the recommendation as a verified result.
`advice.records()` returns every decision as a plain `{name: FieldState}`
dict, and `advice.warnings()` flattens every advisor's warnings into one
list, if you want to inspect everything at once instead of field by field.

## Choose settings yourself

Pass overrides built by `build_overrides` for the settings you want to
control — the same validation path `--set`/HTTP/MCP use, so an unknown key
or a badly-shaped value fails the same way here as anywhere else:

```python
from goldilocks_core.set_overrides import build_overrides

overrides = build_overrides({"k_grid": (4, 4, 4)})
advice = advise(silicon, hpc=hpc, overrides=overrides)

k_sampling = advice.step.kpoints.k_sampling
if isinstance(k_sampling, Resolved):
    print(k_sampling.value.mesh, k_sampling.source)
```

The mesh is now `(4, 4, 4)` with `source == "human"`. An explicit grid
bypasses the k-point model; other settings are still recommended. See
`goldilocks settings` (or the [CLI reference](cli.md)) for every overridable
key.

## Generate and check before publishing

`advise()` never fails; `check()`/`generate()` do, if something the run
needs is still `unavailable`/`blocked`. Always call `check()` first and
inspect `report.blocking` rather than relying on `generate()`'s exception
alone:

```python
from goldilocks_core.service import check, generate
from goldilocks_core.steps import default_shared_context

report = check(advice)
if not report.ok:
    for reason in dict.fromkeys(report.blocking):
        print("blocked:", reason)
else:
    ctx = default_shared_context()
    steps = generate(advice, report, ctx=ctx)
    print([step.name for step in steps])
```

`generate()` renders the actual Quantum ESPRESSO input file(s) as `Step`
objects, raising `AdviceIncomplete` if called with a non-`ok` report — this
is a hard boundary, not the primary way to learn what is missing.

## Choose an output

Turn generated steps into a submission script and a publishable bundle,
then choose where the result goes:

```python
from goldilocks_core.bundle import ArchiveOutput, DirectoryOutput, bundle_files, publish
from goldilocks_core.service import render_submission, to_bundle_input

script = render_submission(advice, hpc, "quantum_espresso", ctx, steps)
bundle_input = to_bundle_input(advice, steps, script, ctx)
```

| Output                            | Result                             |
| ---------------------------------- | ----------------------------------- |
| `bundle_files(bundle_input)`        | Keep the result in memory           |
| `publish(bundle_input, DirectoryOutput("si-run"))` | Write a new directory  |
| `publish(bundle_input, ArchiveOutput("si-run.zip"))` | Write a ZIP with the same contents |

Choose a destination that does not already exist; `publish` refuses to
overwrite one. `publish` returns a `Publication` with `path`, `kind`, and
the list of published file paths.

## Inspect structures

`normalize_structure` is also the entry point for structure-only
inspection, without running any advisor:

```python
inspection = normalize_structure(source).inspection
print(inspection["structure"]["reduced_formula"])
print(inspection["canonical_cif"])
```

## Generate a multi-step task (DOS)

Tasks with more than one Quantum ESPRESSO step use the `_dos`-suffixed
equivalents, which internally run `advise()`/`generate()` once per step
(scf, then a denser nscf pass) and merge the results:

```python
from goldilocks_core.service import (
    advise_dos,
    check_dos,
    generate_dos,
    render_submission_dos,
    to_bundle_input_dos,
)

dos_advice = advise_dos(silicon, hpc=hpc)
dos_report = check_dos(dos_advice)
if dos_report.ok:
    dos_steps = generate_dos(dos_advice, dos_report, ctx=ctx)
    dos_script = render_submission_dos(dos_advice, hpc, "quantum_espresso", ctx, dos_steps)
    dos_bundle = to_bundle_input_dos(dos_advice, dos_steps, dos_script, ctx)
    print([step.name for step in dos_steps])
```

`dos_advice.records()` keys the shared system-level fields unprefixed, the
nscf step's own fields under an `nscf_`-prefixed name, and the DOS decision
itself under `"dos"`. The submission script uses the scf step's resource
decision, since `pw.x`'s scf pass is normally the dominant cost and `dos.x`
is fast.

See [Recommendations](science.md) to interpret other settings, or
[Architecture](architecture.md) to extend the computation workflow.
