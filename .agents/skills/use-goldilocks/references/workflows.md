# Goldilocks workflows

Use these canonical patterns before reading implementation files.

## Inspect a Structure Source

```python
from goldilocks_core.inputs.structure import PathStructureSource, normalize_structure

inspection = normalize_structure(PathStructureSource("structure.cif")).inspection

print(inspection["structure"]["reduced_formula"])
print(inspection["canonical_cif"])
```

## Diagnose a structure (advise)

```python
from goldilocks_core.inputs.hpc import load_hpc_profile
from goldilocks_core.inputs.structure import PathStructureSource, normalize_structure
from goldilocks_core.resolution import Resolved
from goldilocks_core.service import advise
from goldilocks_core.set_overrides import build_overrides

structure = normalize_structure(PathStructureSource("structure.cif")).structure
hpc = load_hpc_profile("scarf")
overrides = build_overrides({"k_grid": (4, 4, 4)})

advice = advise(structure, hpc=hpc, overrides=overrides)

records = advice.records()
print(records["functional"])
k_sampling = advice.step.kpoints.k_sampling
if isinstance(k_sampling, Resolved):
    print(k_sampling.value.mesh)
print(advice.warnings())
```

`advise()` always returns -- diagnosis degrades field-by-field
(`Resolved`/`Unavailable`/`Blocked`) rather than raising. `advice.records()`
returns every decision keyed by name, already JSON-safe; `advice.warnings()`
flattens every advisor's warnings into one list.

## Generate and publish a runnable input

```python
from goldilocks_core.bundle import DirectoryOutput, publish
from goldilocks_core.service import check, generate, render_submission, to_bundle_input
from goldilocks_core.steps import default_shared_context

report = check(advice)
if not report.ok:
    raise SystemExit(f"blocked: {report.blocking}")

ctx = default_shared_context()
steps = generate(advice, report, ctx=ctx)
script = render_submission(advice, hpc, "quantum_espresso", ctx, steps)
bundle_input = to_bundle_input(advice, steps, script, ctx)

publication = publish(bundle_input, DirectoryOutput("run-dir"))
print(publication["path"])
```

`check()` never raises; always inspect `report.ok`/`report.blocking` before
calling `generate()`, which raises `AdviceIncomplete` on an incomplete
report rather than silently degrading (delivery has preconditions that
diagnosis does not).

The destination must not exist. Use `ArchiveOutput("run.zip")` for a ZIP, or
skip `publish` entirely and call `bundle_files(bundle_input)` for a
memory-only preview. Directory and ZIP outputs contain the same logical
files: inputs, selected UPFs, a submission script, licences, citations, and
`goldilocks.json` (records, warnings, checksums).

CLI equivalents:

```bash
uv run goldilocks run structure.cif --hpc scarf --set k_grid=4,4,4 --out run-dir --json
uv run goldilocks run structure.cif --hpc scarf --set k_grid=4,4,4 --out run.zip --json
```

## Read only the records you need

`Advice`/`DosAdvice` always compute every field; there is no partial-record
selection in v2 -- filter the dict `advice.records()` returns instead:

```python
wanted = {"functional", "k_sampling", "cutoffs"}
subset = {name: state for name, state in advice.records().items() if name in wanted}
print(subset)
```

CLI equivalent (the full record set, since `explain`/`run` do not support
partial selection either):

```bash
uv run goldilocks explain structure.cif --hpc scarf --json
```

## Use a local pseudopotential root

```bash
uv run goldilocks run structure.cif --hpc scarf --set pseudo_table_id=my-local-table --out run-dir
```

The registered table's metadata must identify the real licence file and
citation before Core can publish local UPFs. Never invent missing cutoffs
or redistribution terms. See [Pseudopotentials](../../../../docs/pseudopotentials.md)
for how local tables are registered.

## Generate a multi-step task (DOS)

```python
from goldilocks_core.service import advise_dos, check_dos, generate_dos

dos_advice = advise_dos(structure, hpc=hpc)
dos_report = check_dos(dos_advice)
if dos_report.ok:
    dos_steps = generate_dos(dos_advice, dos_report, ctx=ctx)
    print([step.name for step in dos_steps])
```

`DosAdvice.records()` keys the shared system-level fields unprefixed, the
nscf step's own fields under `nscf_`, and the DOS decision itself under
`"dos"`. CLI/HTTP/MCP equivalent: pass `task="dos"` (CLI: `--task dos`).

## Optional transports

```bash
uv sync --all-extras
uv run goldilocks serve http --host 127.0.0.1 --port 8000
uv run goldilocks serve mcp
```

HTTP and MCP share `server/_handlers.py` with the CLI's `run`/`explain`
commands -- same `RunOverrides`/`CalcTask` validation, same tri-state
records and warnings shape, same `dos` task support. Both accept inline
structure content; neither accepts filesystem paths, since a server process
should not read from the caller's local disk.
