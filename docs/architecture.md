# Architecture

This page is for contributors changing Core or its transports. For using the
library, start with the [Python tutorial](tutorial.md); for browser development,
use the [Workbench guide](../web/README.md).

## Set up a contribution

From the repository root:

```bash
uv sync --group dev
uv run pre-commit install
uv run poe check
```

`poe check` runs Ruff lint, format, and complexity checks, then pytest. Use
`uv run poe fmt` to apply Python formatting. The commit hooks run the same checks
and pytest with branch coverage. For frontend checks and API-schema refresh, follow the
[Workbench guide](../web/README.md).

## Follow a request

There is no `Service`/`Runtime` object. `goldilocks_core.service` is a package
of plain functions that a transport (or a Python caller) calls in order.

A `run` (or `explain`) request follows this path:

1. `inputs.structure.normalize_structure` turns a CIF/POSCAR source into a
   pymatgen `Structure`, recording a sha256 of the source bytes for
   provenance. This is the same normalization CLI's `inspect` and the HTTP
   `/inspect` route use.
2. `service.advise(structure, *, code, hpc, overrides, store, fetch_missing)`
   runs three phases in a fixed order: `_analysis.analyze` (structure facts),
   `_system.system_advice` (system-wide advisors, including pseudopotential-
   table selection), and `_step.step_advice` (per-step k-point/occupation/
   resource advisors). `advise()` always returns an `Advice`; it degrades a
   field to `Unavailable`/`Blocked` rather than raising, for anything short
   of a malformed structure.
3. `service.check(advice, *, purpose="scf", relax=None)` calls
   `checks.check_all(...)`, the one place in the whole system allowed to
   hard-fail a request. It returns a `CheckReport` whose `.ok`/`.blocking`
   decide whether `generate()` may proceed.
4. `service.generate(advice, report, *, ctx, purpose, relax)` raises
   `AdviceIncomplete` (an `ExpectedFailure`, see `failures.py`) if
   `report.ok` is false; otherwise it renders one or more `Step`s by calling
   `generation.quantum_espresso.scf.write_qe_scf` or
   `generation.quantum_espresso.relax.write_qe_relax` directly. There is no
   dispatch table.
5. `service.render_submission(advice, hpc, code, ctx, steps)` renders the
   SLURM script via `submission/slurm.py`.
6. `service.to_bundle_input(advice, steps, submission_script, ctx)` assembles
   a `bundle.BundleInput`: the rendered input file(s), the pseudopotential
   bytes read straight off disk for each selected element, the submission
   script, every advisor decision as a tri-state record (`advice.records()`),
   and citations pulled from the pseudopotentials' own registry metadata.
7. `bundle.publish(bundle_input, output)` / `bundle.bundle_files(bundle_input)`
   / `bundle.archive_bytes(bundle_input)` write a directory (`DirectoryOutput`)
   or ZIP (`ArchiveOutput`), or return an in-memory preview.

CLI's `explain` is exactly `advise()` (plus `check()`, to print blocking
reasons); CLI's `run` is `advise()` -> `check()` -> `generate()` ->
`render_submission()` -> `to_bundle_input()` -> `bundle.publish()`/preview.
Omitting `-o/--out` gives a memory-only preview (file names and byte sizes
only, nothing written); a directory path publishes a directory; a path
ending in `.zip` publishes an archive. HTTP's `POST /run` takes a
`respond_with: "archive"` field to get ZIP bytes back instead of a JSON
summary (`server/http.py`).

The `dos` task reuses steps 2-6 twice — once for `scf`, once for `nscf` — via
`service.advise_dos`/`check_dos`/`generate_dos` (`service/_dos.py`), rather
than any generic graph: nothing about resolving one step's own settings
changes between an scf task and dos's scf/nscf steps, so the same
single-step pipeline runs twice with nscf-specific overrides. `relax`/
`vc-relax` similarly get `advise_relax`/`check_relax`/`generate_relax`
(`service/_relax.py`), calling the single-step pipeline once and adding one
more resolved decision (`relax`) on top.

## Understand the pipeline phases

Every task runs the same fixed order — `advise` -> `check` -> `generate` ->
`render_submission` -> `to_bundle_input` -> `publish` — described above.
Inside `advise()` itself, the three phases are:

- **Analyze** (`service/_analysis.py`, backed by `analysis/`: `composition.py`,
  `geometry.py`, `is_magnetic.py`, `is_metal.py`, `needs_correlation.py`,
  `needs_soc.py`, `symmetry.py`) reports structure facts.
- **System advice** (`service/_system.py`, backed by system-scoped advisors in
  `advisors/` such as `functional.py`, `hubbard_u.py`, `vdw_method.py`,
  `boundary.py`, `magnetic_config.py`) recommends system-wide parameters,
  including pseudopotential-table selection via `advisors/pseudo_selection.py`
  and `service/_pseudo.py`.
- **Step advice** (`service/_step.py`, split into `service/_step_kpoints.py`
  for occupations/k-mesh/bands/convergence and `service/_step_resources.py`
  for the resource envelope/job/parallelisation, backed by per-step advisors
  such as `advisors/k_sampling.py`, `advisors/nbnd.py`, `advisors/occupations.py`,
  `advisors/convergence.py`, `advisors/cutoffs.py`, `advisors/job_resources.py`,
  `advisors/parallelisation.py`, `advisors/n_irr_k.py`, `advisors/relax.py`)
  recommends per-step parameters.

There is no generic "stage" or "graph executor" abstraction. Each phase is a
plain typed function call; `Advice` composes the three phases' own
dataclasses (`analysis`, `system`, `step`) rather than one flat dataclass, so
no single module needs to import every advisor's types directly. Every
advisor degrades independently (`Resolved`/`Unavailable`/`Blocked`, see
`resolution.py`) instead of raising; only `checks.py` decides whether a
request may proceed to `generate()`.

## Find the code to change

Paths below are relative to `src/goldilocks_core/` unless stated otherwise.

| Area                | Files and responsibility                                                                                                                                                                                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Core types          | `resolution.py`: the `Resolved`/`Unavailable`/`Blocked` tri-state and `Provenance`/`Source`, used everywhere; `steps.py`: `Step`/`SharedContext`; `step_settings.py`/`system_settings.py`: the settings dataclasses `generate()` builds; `plan.py`: `PlannedStep`/`expand_task`, multi-step task planning. There is no centralized `contracts` package any more — importing `goldilocks_core.contracts` is a ruff-banned API; import from the module that defines the name instead. |
| Orchestration       | `service/_pipeline.py`: `advise()`/`check()`, the single-step orchestration; `service/_generate.py`: `generate()`, the hard-fail boundary; `service/_dos.py` / `service/_relax.py`: the multi-step and relax task variants; `service/_bundle.py`: submission rendering and bundle assembly; `checks.py`: the sole cross-parameter hard-fail gateway.                                                                     |
| Input and discovery | `inputs/structure.py`: structure normalization and inspection; `capabilities.py`: available codes, tasks, settings, pseudopotential tables, HPC profiles, and warnings — reflected from the override dataclasses and advisor warning catalogues, not hand-listed.                                                        |
| Scientific behavior | `analysis/`: structure facts (`composition.py`, `geometry.py`, `is_magnetic.py`, `is_metal.py`, `needs_correlation.py`, `needs_soc.py`, `symmetry.py`); `advisors/`: the recommendation layer (`boundary.py`, `convergence.py`, `cutoffs.py`, `electron_count.py`, `functional.py`, `hubbard_u.py`, `job_resources.py`, `k_sampling.py`, `magnetic_config.py`, `n_irr_k.py`, `nbnd.py`, `occupations.py`, `parallelisation.py`, `pseudo_selection.py`, `relax.py`, `size.py`, `vdw_method.py`, `dos.py`, `warning_catalogue.py`); top-level `kmesh.py`: k-point mesh ladder math. |
| Assets              | `assets/`: installation and integrity (`AssetStore`); `ml/models.py`: model declarations (loading plumbing only — no advisor calls it yet, see below); `assets/pseudopotentials/registry.py` and `assets/pseudopotentials/importers.py`: table declarations and provider normalization; `service/_pseudo.py`: the one place in the pipeline that touches the asset store/filesystem, resolving the installed asset, loading its manifest, and narrowing it to a structure's elements. |
| Output              | `generation/`: target-code writers (`generation/quantum_espresso/scf.py`, `relax.py`, `dos.py`, `namelists.py`); top-level `bundle.py`: manifest assembly and directory/ZIP publication, replacing v1's separate `input_data.py`/`publication.py`/`serialization.py` roles.                                                                                                                        |
| Transports          | `cli/core.py`: local commands; `server/documents.py`: request/response pydantic models; `server/_handlers.py`: the one shared request-validation/dispatch implementation both `server/http.py` and `server/mcp.py` call; `server/readiness.py`: cached asset checks.                                                     |
| Browser             | Repository `web/src/api/`: HTTP client and generated types; `web/src/workspace/`: draft, request, result, and download state.                                                                                                                                                                                              |

## Extend a workflow

### Add a task

There is no generic task-registration API or graph to define. To add a new
task, add a `service/_<task>.py` module that reuses the existing single-step
phases (`_analysis.analyze`, `_system.system_advice`, `_step.step_advice`,
and `_pipeline.advise`/`_generate.generate`) — calling them once for a
single-step task, or more than once for a multi-step task, the way
`service/_dos.py` calls the single-step pipeline twice (once for `scf`, once
for `nscf`) instead of rebuilding `Advice`/`StepAdvice` as "one-or-many".
Multi-step ordering is described by `plan.py`'s `PlannedStep`/`expand_task`,
not a graph.

Export the new `advise_<task>`/`check_<task>`/`generate_<task>`/
`render_submission_<task>`/`to_bundle_input_<task>` functions from
`service/__init__.py`'s `__all__`, and wire the CLI's `--task` choices
(`cli/_run.py`) and `capabilities()`'s `tasks` list (`capabilities.py`) to
match.

### Add an input writer

Implement the writer as a function in `generation/quantum_espresso/` (or a
new `generation/<code>/` package for a new code) taking the resolved
`SystemSettings`/`PwSettings` and a `SharedContext`, and returning `Step`s
(see `steps.py`). There is no `(code, task, writer)` registry — dispatch is a
direct Python reference in `service/_generate.py`
(`write_qe_relax if purpose in ("relax", "vc-relax") else write_qe_scf`).
Wire a new writer into `service/_generate.py` or the relevant
`service/_<task>.py` module the same way. Writers should reject unsupported
combinations before rendering, rather than inventing scientific choices —
per `generation/quantum_espresso/scf.py`'s own governing invariant,
"generation never makes a scientific choice, only translation and
formatting." The writers today cover all four tasks: `write_qe_scf`,
`write_qe_relax`, and a DOS writer (`generation/quantum_espresso/dos.py`).

### Add a model or pseudopotential table

Keep scientific metadata and provider-specific preparation in the
pseudopotential registry (`assets/pseudopotentials/registry.py`). Table
selection and requirements-building live in `advisors/pseudo_selection.py`
(`select_pseudopotential_table`, `pseudo_requirements`,
`select_metadata_for_elements`); `service/_pseudo.py` is the one place that
combines table selection with `AssetStore` acquisition, manifest loading, and
per-element narrowing behind one function, keeping asset-store-shaped types
out of `_system.py`'s own import surface. The per-request result lives on
`Advice.system.pseudo` (a `PseudoAdvice`) as an ordinary `FieldState`.

`AssetStore` (`assets/store.py`) handles acquisition, integrity, on-disk
staging, and verified path resolution for any installable asset — table
pinning today is only via `--set pseudo_table_id=...` against the registry's
15 built-in tables; there is no user-supplied local UPF directory feature.
Installation is explicit through `goldilocks assets install`/`status`/
`verify` or the CLI's `--fetch-missing` retry; scientific operations never
install assets themselves.

The `default` asset profile also installs two ML models
(`models/qrf-kpoints`, `models/metallicity-cgcnn`) alongside the
pseudopotential table, but no advisor calls `ml.models.load_model` yet —
every advisor's heuristic tier is the only tier that runs today (e.g.
`analysis/is_metal.py` decides `metal`/`Unavailable` from composition alone;
`advisors/k_sampling.py` picks a flat k-point spacing from `is_metal`). ML
selection is a separate, still-empty pluggable model registry reserved for a
later epic (`goldilocks models list` reports none installed even after
installing the `default` profile) — do not describe model-backed selection
as live behavior.

Independent assets install concurrently with at most eight workers. Results
retain profile order; each asset retains its own lock, checksum verification,
staging directory, and atomic publication. See
[pseudopotential tables](pseudopotentials.md) for layout and licensing
requirements.

## Preserve the boundaries

- **Validate at entry and side effects.** Input constructors validate domain
  controls; transport models (`server/documents.py`) reject unknown fields
  and incorrect types; provider adapters validate imported data. Internal
  advisor/generation functions are trusted to return coherent values.
- **Keep pipeline state request-local.** There is no long-lived service
  object whose lifecycle needs managing: each HTTP/MCP request calls plain
  functions from `service/`, building whatever `AssetStore`/`HpcProfile` it
  needs from configuration for that call. `server/_handlers.py` holds the one
  shared request-validation/dispatch implementation both `http.py` and
  `mcp.py` call; `server/readiness.py`'s `AssetReadiness` caches
  asset-availability checks (used by `GET /ready`) across requests.
- **Publish through one implementation.** `bundle.py`'s `assemble_bundle_input`
  reads each selected pseudopotential's bytes straight off disk
  (`Path(metadata.filepath).read_bytes()`) once, hashes every artifact while
  assembling the `goldilocks.json` manifest, and writes the same in-memory
  bytes it hashed — never rereading a source after hashing. Publication uses
  a completion-marker protocol, not v1's staging-directory-plus-atomic-rename:
  `target.mkdir()` (directories) or an exclusive `open(path, "xb")`
  (archives) claims the destination, raising `FileExistsError` if it already
  exists; every byte is written; a `.complete` marker file is written
  strictly last. Any consumer must call `bundle.is_complete(target)` before
  trusting a directory/archive — a killed process (SIGKILL/OOM) can leave a
  real-but-incomplete directory behind with no exception ever raised, an
  explicitly accepted trade-off versus v1's staging approach, which made that
  physically impossible.
- **Citations come from pseudopotentials, not models.** A published bundle's
  `citations` are pulled exclusively from the selected pseudopotential
  table's registry metadata (`metadata.pseudo_info.get("citation")`); there
  is currently no ML-model citation/licence path in the bundle, consistent
  with the model registry being empty.
- **Keep remote callers away from local paths.** HTTP and MCP accept inline
  structures and registered table IDs, not filesystem sources, model
  locations, or publication paths. Keep authentication and deployment
  controls explicit; see [CLI transport security](cli.md#http-security).
- **Keep imports direct.** Package code imports from the module defining a
  name. The top-level `goldilocks_core` package deliberately exports
  nothing; library users import directly from `goldilocks_core.service`
  (`advise`/`check`/`generate` and the `dos`/`relax` task variants),
  `goldilocks_core.cli`, or `goldilocks_core.server`.
- **Classify expected failures once.** Domain exceptions implement
  `ExpectedFailure` from `failures.py` with a category and safe public
  description (e.g. `AdviceIncomplete` in `service/_generate.py`,
  `AssetNotInstalled`/`AssetCorrupt` in `assets/store.py`). Adapters classify
  failures for remote callers from this vocabulary; unexpected execution
  defects propagate.

## Complexity gates

`scripts/check_complexity.py` runs in contributor checks, hooks, and CI. Its AST
import ceiling applies to every production owner: 12 project origin modules and
24 imported symbols by default, with stricter limits for `cli.core`, `cli._run`,
`server.http`, `server.mcp`, `service._system`, `service._dos`, and
`server._handlers` (the modules whose own orchestration role needs more
imports than the default ceiling allows).
It counts local and type-only imports, resolves re-exports, and counts accessed
module-alias members. Pure export packages are transparent to consumer counts.
The same gate runs Ruff's McCabe check: at most 10 per production function,
ignoring `noqa` suppressions. Reduce decisions and duplication; moving import
blocks or extracting shallow helper fleets does not deepen an interface.

## Change the HTTP or browser contract

`server/documents.py` converts strict request models directly into native Core
inputs and derives response schemas from domain-owned shapes and `Portable`
annotations. `server/_handlers.py` serializes trusted Core results without
reconstructing response models — both `server/http.py` and `server/mcp.py`
call the same handler functions. HTTP's `POST /run` returns a JSON summary by
default, or ZIP bytes when the request body sets `respond_with: "archive"`.
Workbench reads file hashes and sizes from the ZIP's `goldilocks.json`.

There is no per-transport service object to own. `server/readiness.py`'s
`AssetReadiness` caches asset-availability checks (used by `GET /ready`)
across requests within one running process.

Workbench keeps the editable draft and reviewed response in its browser store.
Editing a draft marks the existing result out of date and prevents downloading
its archive. Download uses the bytes returned with that result rather than
recomputing it.

After changing the wire contract, run these from the repository root with the
HTTP extra installed:

```bash
npm --prefix web run generate:api
```

`web/openapi.json` and `web/src/api/schema.d.ts` are ignored build products.
Frontend lifecycle commands regenerate them from the local Python package;
generation needs neither a running server nor installed model assets. Commit
the domain declarations, not generated copies. Docker exports OpenAPI in its
Python stage and generates TypeScript in its Node stage.

## Run browser tests

Start the [built Workbench](../web/README.md#run-locally) in another terminal,
then run:

```bash
npm --prefix web exec -- playwright install chromium
npm --prefix web run test:e2e
```

Tests use `http://127.0.0.1:8000`; they do not start a server.
`WORKBENCH_BASE_URL` selects another address.
`PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` selects an existing Chromium installation.
