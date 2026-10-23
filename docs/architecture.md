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
   `_system.system_advice` (system-wide advisors, starting with the simulated
   cell and pseudopotential-table selection), and `_step.steps_advice`
   (per-step k-point/occupation advisors, then the task's job and each step's
   parallelisation).
   `advise()` returns an `Advice` even for an incomplete request: a field it
   cannot decide, because an input is missing, something is not installed or
   configured, or the request leaves it open, is `Unavailable`/`Blocked`. A
   fault raises and fails the request instead: a model that is set up but
   fails to load or predict, an enumeration error that is not the
   enumerator's own answer about the structure, or a worker process that
   dies.
3. `service.check(advice, *, purpose="scf", relax=None)` calls
   `checks.check_all(...)`, the one place that refuses a request whose
   decisions are incomplete or inconsistent. It returns a `CheckReport`
   whose `.ok`/`.blocking` decide whether `generate()` may proceed.
4. `service.generate(advice, report, *, ctx, purpose, relax)` raises
   `AdviceIncomplete` (an `ExpectedFailure`, see `failures.py`) if
   `report.ok` is false; otherwise it renders one or more `Step`s by calling
   `generation.quantum_espresso.scf.write_qe_scf` or
   `generation.quantum_espresso.relax.write_qe_relax` directly. There is no
   dispatch table.
5. `service.render_submission(advice, hpc, code, steps)` renders the
   SLURM script via `submission/slurm.py`. Each step is followed by the
   completion check its writer declared (`Step.completion`), and the script
   stops with a `goldilocks:` message when one fails.
6. `service.to_bundle_input(advice, steps, submission_script, ctx)` assembles
   a `bundle.BundleInput`: the rendered input file(s), the pseudopotential
   bytes read straight off disk for each selected element, the submission
   script, every advisor decision as a tri-state record (`advice.records()`),
   and the advice's warnings (`advice.warnings()`). Records stay typed until
   they are published: `resolution.records_document` projects them once, for
   the HTTP/MCP responses, CLI `--json` and `goldilocks.json` alike, each with
   its whole `Provenance`, and `resolution.collect_citations` reads what the
   bundle cites off the same records.
7. `bundle.publish(bundle_input, output)` / `bundle.bundle_files(bundle_input)`
   / `bundle.archive_bytes(bundle_input)` write a directory (`DirectoryOutput`)
   or ZIP (`ArchiveOutput`), or return an in-memory preview.

`explain` (CLI, HTTP `POST /explain` and the MCP `explain` tool) is
`advise()` -> `check()` (`check_dos()`/`check_relax()` for those tasks): every
decision plus `blocking`, the reasons `run` would refuse to generate, without
rendering anything. CLI's `run` is `advise()` -> `check()` -> `generate()` ->
`render_submission()` -> `to_bundle_input()` -> `bundle.publish()`/preview.
Omitting `-o/--out` gives a memory-only preview (file names and byte sizes
only, nothing written); a directory path publishes a directory; a path
ending in `.zip` publishes an archive. HTTP's `POST /run` takes a
`respond_with: "archive"` field to get ZIP bytes back instead of a JSON
summary (`server/http.py`).

`advise()` is the one-step case of `service/_pipeline.py`'s internal
`advise_steps`, which diagnoses every `pw.x` step of a task together. The
`dos` task asks it for an `scf` and an `nscf` step via
`service.advise_dos`/`check_dos`/`generate_dos` (`service/_dos.py`), rather
than any generic graph: nothing about resolving one step's own settings
changes between an scf task and dos's scf/nscf steps, so both steps get the
same per-step advisors, the nscf with nscf-specific overrides. Analysis and
system advice run once, and the steps share one job, because they run one
after another in one allocation: it is sized over both steps, and each
`pw.x` step gets its own `max_seconds` share of the walltime, since `pw.x`
counts `max_seconds` from its own start. `dos.x` runs in what is left.
`relax`/`vc-relax` similarly get `advise_relax`/`check_relax`/
`generate_relax` (`service/_relax.py`), asking `advise_steps` for their one
step and adding one more resolved decision (`relax`) on top.

## Understand the pipeline phases

Every task runs the same fixed order — `advise` -> `check` -> `generate` ->
`render_submission` -> `to_bundle_input` -> `publish` — described above.
Inside `advise()` itself, the three phases are:

- **Analyze** (`service/_analysis.py`, backed by `analysis/`: `composition.py`,
  `geometry.py`, `is_magnetic.py`, `is_metal.py`, `local_moments.py`,
  `needs_correlation.py`, `needs_soc.py`, `symmetry.py`) reports structure
  facts. `local_moments.py` is the open-shell model behind `is_magnetic`'s
  heuristic tier.
- **System advice** (`service/_system.py`, backed by system-scoped advisors in
  `advisors/` such as `functional.py`, `hubbard_u.py`, `vdw_method.py`,
  `boundary.py`, `magnetic_config.py`) recommends system-wide parameters,
  including pseudopotential-table selection via `advisors/pseudo_selection.py`
  and `service/_pseudo.py`. It starts with the simulated cell
  (`service/_magnetic.py`): `magnetic_config.magnetic_cell` resolves spin
  polarization, spin-orbit coupling and a magnetic spin-orbit run's moment
  direction, and applies the requested magnetic ordering, the only enumlib
  call, which can replace the input with a different cell;
  `magnetic_config.framed_cell` then writes that cell in the frame where pw.x
  keeps the most of its point group (`pw_symmetry.pw_frame`). Table selection
  then reads the cell's spin-orbit switch, and
  `magnetic_config.magnetic_config` seeds the cell's moments from the
  selected z_valence. Everything after that -- the electron count, the step
  advice, the gate and the writer -- reads that one cell object, so no count
  describes a cell other than the one written.
- **Step advice** (`service/_step.py`, split into `service/_step_kpoints.py`
  for occupations/k-mesh/bands/convergence and `service/_step_resources.py`
  for the job/parallelisation, backed by per-step advisors such as
  `advisors/k_sampling.py`, `advisors/nbnd.py`, `advisors/occupations.py`,
  `advisors/convergence.py`, `advisors/cutoffs.py`, `advisors/job_resources.py`,
  `advisors/parallelisation.py`, `advisors/n_irr_k.py`, `advisors/relax.py`)
  recommends per-step parameters. Resources are task-level: one job
  (`advisors/job_resources.py`) takes the fewest nodes on which pw.x's
  estimated memory per process fits for the layout every step will run
  (`advisors/size.py`, a port of pw.x's own memory report), splits the
  walltime into per-step `max_seconds` shares, and publishes that memory
  in its record; each step's `npool`/`ndiag` then follow from that job's
  `ntasks`.

There is no generic "stage" or "graph executor" abstraction. Each phase is a
plain typed function call; `Advice` composes the three phases' own
dataclasses (`analysis`, `system`, `step`) rather than one flat dataclass, so
no single module needs to import every advisor's types directly. Every
advisor degrades independently (`Resolved`/`Unavailable`/`Blocked`, see
`resolution.py`) when it cannot decide its field, and lets a fault
propagate; only `checks.py` decides whether a request may proceed to
`generate()`.

## Find the code to change

Paths below are relative to `src/goldilocks_core/` unless stated otherwise.

| Area                | Files and responsibility                                                                                                                                                                                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Core types          | `resolution.py`: the `Resolved`/`Unavailable`/`Blocked` tri-state and `Provenance`/`Source`, used everywhere; `steps.py`: `Step`/`SharedContext`; `step_settings.py`/`system_settings.py`: the settings dataclasses `generate()` builds; `plan.py`: `PlannedStep`/`expand_task`, multi-step task planning. There is no centralized `contracts` package any more — importing `goldilocks_core.contracts` is a ruff-banned API; import from the module that defines the name instead. |
| Orchestration       | `service/_pipeline.py`: `advise()`/`check()`, and `advise_steps`, which advises every `pw.x` step of a task with one shared job; `service/_generate.py`: `generate()`, the hard-fail boundary; `service/_dos.py` / `service/_relax.py`: the multi-step and relax task variants; `service/_bundle.py`: submission rendering and bundle assembly; `checks.py`: the sole cross-parameter hard-fail gateway. |
| Input and discovery | `inputs/structure.py`: structure normalization and inspection; `capabilities.py`: available codes, tasks, settings, pseudopotential tables, HPC profiles, and warnings — reflected from the override dataclasses and advisor warning catalogues, not hand-listed.                                                        |
| Scientific behavior | `analysis/`: structure facts (`composition.py`, `geometry.py`, `is_magnetic.py`, `is_metal.py`, `local_moments.py`, `needs_correlation.py`, `needs_soc.py`, `symmetry.py`); `advisors/`: the recommendation layer (`boundary.py`, `convergence.py`, `cutoffs.py`, `electron_count.py`, `functional.py`, `hubbard_u.py`, `job_resources.py`, `k_sampling.py`, `magnetic_config.py`, `n_irr_k.py`, `nbnd.py`, `occupations.py`, `parallelisation.py`, `pseudo_selection.py`, `relax.py`, `size.py`, `species_split.py`, `vdw_method.py`, `dos.py`, `warning_catalogue.py`); top-level `kmesh.py`: k-point mesh ladder math; top-level `pw_symmetry.py`: the symmetry operations pw.x keeps in a cell as written (which `advisors/n_irr_k.py` counts k-points with) and the frame the cell is written in. |
| Assets              | `assets/`: installation and integrity (`AssetStore`); `ml/models.py`: model declarations (loading plumbing only — no advisor calls it yet, see below); `assets/pseudopotentials/registry.py` and `assets/pseudopotentials/importers.py`: table declarations and provider normalization; `service/_pseudo.py`: the one place in the pipeline that touches the asset store/filesystem, resolving the installed asset, loading its manifest, narrowing it to a structure's elements, and reading the selected UPF headers' projector counts for the memory estimate. |
| Output              | `generation/`: target-code writers (`generation/quantum_espresso/scf.py`, `relax.py`, `dos.py`, `namelists.py`); top-level `bundle.py`: manifest assembly and directory/ZIP publication, replacing v1's separate `input_data.py`/`publication.py`/`serialization.py` roles.                                                                                                                        |
| Transports          | `cli/core.py`: local commands; `server/documents.py`: request/response pydantic models; `server/_handlers.py`: the one shared request-validation/dispatch implementation both `server/http.py` and `server/mcp.py` call; `server/readiness.py`: cached asset checks.                                                     |
| Browser             | Repository `web/src/api/`: HTTP client and generated types; `web/src/workspace/`: draft, request, result, and download state.                                                                                                                                                                                              |

## Extend a workflow

### Add a task

There is no generic task-registration API or graph to define. To add a new
task, add a `service/_<task>.py` module that calls `_pipeline.advise_steps`
with one entry per `pw.x` step (purpose -> that step's `StepOverrides`, in
run order) — one entry for a single-step task, like `service/_relax.py`,
or several, like `service/_dos.py`'s `scf` and `nscf` — and then
`_generate.generate` once per step with that step's purpose. `advise_steps`
returns one `Advice` per step, all sharing the same analysis, system advice
and job, so the submission script is rendered from that one job, which
holds every step's `max_seconds` share. Multi-step ordering is described by
`plan.py`'s `PlannedStep`/`expand_task`, not a graph.

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
A writer sets `Step.ntasks` when its program cannot use the whole allocation
(the DOS writer sets 1); `submission/slurm.py` then launches that step on that
many tasks instead of every task the job allocates. Every `Step` also declares
how the script decides it finished (`Step.completion`, from `steps.py`):
`pw.x` steps use `ExitStatusCheck`, which passes only if
`<outdir>/<prefix>.save/data-file-schema.xml` was rewritten during the step
with `exit_status` 0, and `dos.x`, which writes no status file, uses
`StdoutMarkerCheck` (`JOB DONE` in its stdout).

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

The `default` asset profile also installs four ML models
(`models/qrf-kpoints`, `models/metallicity-cgcnn`,
`models/is-metal-classifier`, `models/is-magnetic-classifier`) alongside the
pseudopotential table. `analysis/is_metal.py`, `analysis/is_magnetic.py`, and
`advisors/k_sampling.py` all call through `ml.predict` (`predict()` for the
two classifiers, `predict_k_distance()` for `qrf-kpoints`), which resolves
the registered asset via `AssetStore` and loads it once per process through
`goldilocks_ml.inference.load_model()`. Each caller falls back to its next
tier only when the model is not set up: its asset is not installed or fails
checksum verification, or, for `is_magnetic`, the mMACE backbone
(`GOLDILOCKS_MACE_BACKBONE`) is not configured or `mace`, `e3nn` or
`sphericart` is not installed. Any other error loading or running a model
fails the request (see `ml/predict.py`).

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
- **Publish through one implementation.** `service/_bundle.py`'s
  `assemble_bundle_input` takes the pseudopotential files from
  `service/_pseudo.py`'s `pseudo_artifacts`, the only producer of
  `pseudopotential` artifacts: each selected UPF and, beside them, the table's
  `LICENSE.txt` as `licences/<table-id>.txt` (role `licence`), so no bundle
  ships a UPF without its licence. It reads each file once through the
  installation's verified inventory (`InstalledAsset.read_bytes`) and raises
  `AssetCorrupt` for one that no longer matches it, so a UPF's bytes are those
  its `pseudopotentials` record hashes. `bundle.py` hashes every artifact while
  assembling the `goldilocks.json` manifest, and writes the same in-memory
  bytes it hashed — never rereading a source after hashing. Publication uses a
  completion-marker protocol, not v1's staging-directory-plus-atomic-rename:
  `target.mkdir()` (directories) or an exclusive `open(path, "xb")` (archives)
  claims the destination, raising `FileExistsError` if it already exists;
  every byte is written; a `.complete` marker file is written strictly last.
  Any consumer must call `bundle.is_complete(target)` before trusting a
  directory/archive — a killed process (SIGKILL/OOM) can leave a
  real-but-incomplete directory behind with no exception ever raised, an
  explicitly accepted trade-off versus v1's staging approach, which made that
  physically impossible.
- **Cite what the records cite.** Every `Provenance` field is published
  (`ResolvedField` forbids keys it does not declare, so a new one cannot be
  dropped silently). A record an ML model decided carries the model's
  evidence (`ml.predict.MlPrediction`: goldilocks-ml id, registry asset,
  details and warnings as returned) and its `ml/registry.toml` citation; the
  `pseudo_table` record carries the selected table's. A bundle's
  `CITATIONS.md` and manifest `citations` are every record's citations;
  `CITATIONS.md` also names each `licence` file, which carries per-family
  references the records do not. ML model licence material is not shipped:
  bundles do not contain the models.
- **Keep remote callers away from local paths.** HTTP and MCP accept inline
  structures and registered table IDs, not filesystem sources, model
  locations, or publication paths. Published records name no local path
  either: a field annotated `Portable()` (`PseudoMetadata.filepath`,
  `ModelSpec.location`) never reaches them, and a `pseudopotentials` record
  names each UPF by `table_id`, `asset_path` and `content_sha256`, so two
  machines with the same assets publish the same records. Keep authentication
  and deployment controls explicit; see
  [CLI transport security](cli.md#http-security).
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

Documents are strict JSON. HTTP's `/explain`, `/magnetic-orderings` and `/run`
bodies, CLI `--json` output and `goldilocks.json` raise on a non-finite float
instead of writing `Infinity`, `NaN` or a silent `null`. Records therefore
never hold one: a bound or value with no finite number is `None` (for example
the upper end of the Gamma-only mesh's `k_distance_interval`), and the
override models reject `inf` and `nan`. MCP's text content, which pydantic
serializes and would write as `Infinity`, depends on the same rule.

There is no per-transport service object to own. `server/readiness.py`'s
`AssetReadiness` caches asset-availability checks (used by `GET /ready`)
across requests within one running process.

Requests run concurrently on threads (FastAPI's threadpool, MCP's
`asyncio.to_thread`), so no request may change process-wide state such as the
working directory. pymatgen's enumlib caller `chdir`s the whole process, so
magnetic-ordering enumeration runs on one long-lived worker process
(`worker_process.py`), one call at a time, in a private temporary directory
(`advisors/magnetic_enumeration.py`). mace sets torch's default dtype for the
whole process, which breaks float32 models built on other threads, so the
server and CLI processes never import it: `is_magnetic` predictions run on a
second long-lived worker (`ml/mace_process.py`), and each mMACE ordering
ranking runs in a fresh process, one ranking at a time
(`advisors/magnetic_ordering_ml.py`). The Workbench static root is resolved to
an absolute path once, when the app is created.

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
