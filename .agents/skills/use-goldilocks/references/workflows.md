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

normalized = normalize_structure(PathStructureSource("structure.cif"))
structure = normalized.structure
hpc = load_hpc_profile("scarf")
assignments = {"k_grid": (4, 4, 4)}
overrides = build_overrides(assignments, task="scf_single_point")

advice = advise(structure, hpc=hpc, overrides=overrides)

records = advice.records()
print(records["functional"])
k_sampling = advice.step.kpoints.k_sampling
if isinstance(k_sampling, Resolved):
    print(k_sampling.value.mesh)
print(advice.warnings())
```

`build_overrides` routes each setting to the steps of `task` it applies to and
raises `InvalidSetting` for one no step of the task reads (`press` on an scf).

`advise()` always returns -- diagnosis degrades field-by-field
(`Resolved`/`Unavailable`/`Blocked`) rather than raising. `advice.records()`
returns every decision keyed by name, as the advisors' typed values;
`goldilocks_core.resolution.records_document(records)` projects them to the
JSON the CLI, HTTP, MCP and `goldilocks.json` publish. `advice.warnings()`
flattens every record's warnings into one list, each naming the record it came
from in `record`.

## Generate and publish a runnable input

```python
from goldilocks_core.bundle import DirectoryOutput, publish
from goldilocks_core.service import check, generate, render_submission, to_bundle_input
from goldilocks_core.steps import default_shared_context

report = check(advice)
if not report.ok:
    raise SystemExit(
        "blocked: " + "; ".join(finding.message for finding in report.blocking)
    )

ctx = default_shared_context()
steps = generate(advice, report, ctx=ctx)
script = render_submission(advice, hpc, "quantum_espresso", steps)
bundle_input = to_bundle_input(
    advice,
    steps,
    script,
    ctx,
    code="quantum_espresso",
    hpc=hpc,
    assignments=assignments,
    source=normalized.source,
)

publication = publish(bundle_input, DirectoryOutput("run-dir"))
print(publication["path"])
```

`to_bundle_input`'s keywords are the request the bundle answers, which
`goldilocks.json` records under `request` with the task and the goldilocks-core
version: pass the assignments given to `build_overrides` (the overrides as
submitted) and `normalized.source` (the structure's name, format, sha256 and
size).

`check()` never raises; always inspect `report.ok`/`report.blocking` before
calling `generate()`, which raises `AdviceIncomplete` on an incomplete
report rather than silently degrading (delivery has preconditions that
diagnosis does not). Each entry of `report.blocking` is a `Finding`: a stable
`code`, the settings it concerns (`fields`), a `message` to relay, and
`remedies`, override patches that each resolve it (a `None` value removes
that override from the assignments). Apply one and call `build_overrides`
again rather than guessing a fix from the message.

The destination must not exist. Use `ArchiveOutput("run.zip")` for a ZIP, or
skip `publish` entirely and call `bundle_files(bundle_input)` for a
memory-only preview. Directory and ZIP outputs contain the same logical
files: inputs, selected UPFs, a submission script, licences, citations, and
`goldilocks.json` (the request, records, warnings, citations, checksums).

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

dos_overrides = build_overrides({"steps.nscf.k_grid": (12, 12, 12)}, task="dos")
dos_advice = advise_dos(structure, hpc=hpc, overrides=dos_overrides)
dos_report = check_dos(dos_advice)
if dos_report.ok:
    dos_steps = generate_dos(dos_advice, dos_report, ctx=ctx)
    print([step.name for step in dos_steps])
```

`DosAdvice.records()` keys the shared system-level fields and the task's one
`job` unprefixed, the nscf step's own fields under `nscf_`, and the DOS
decision itself under `"dos"`; `dos_advice.tiers().layout()` names each
record's tier and step (scf, nscf, dos), which `explain`/`run --json` publish
as `layout`. The job splits 95% of `walltime_h` between the
scf and nscf inputs as `max_seconds` (`job.max_seconds_by_step`). Unless the
request sets them, the nscf step's k-point spacing and occupations come from
the dos protocol (aiida-quantumespresso `pdos.yaml`); those records read
`heuristic` with the protocol in `source_note`, never `human`.
`steps.<step>.<key>` sets one step (`scf`, `nscf`, or `dos` for the `dos.x`
settings `delta_e`, `emin`, `emax`, `broadening`, `ngauss`; `dos.x` reads
`ngauss` only with a broadening width, so it needs `broadening` unless the
nscf step uses smearing); a key without a step reaches every step it applies
to, with an `overrides.applies_to_multiple_steps` notice when that is both
`pw.x` steps.
CLI/HTTP/MCP equivalent: pass `task="dos"` (CLI: `--task dos`).

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
