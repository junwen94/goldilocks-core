# CLI reference

For installation and a first calculation, follow the
[quickstart](quickstart.md).

## Common commands

Replace `structure.cif` with your CIF or POSCAR:

```bash
uv run goldilocks inspect structure.cif --json
uv run goldilocks explain structure.cif --json
uv run goldilocks run structure.cif --out silicon-run
```

`inspect` reports structure metadata. `explain` runs analysis and every
advisor, printing each decision and its source (`human`, `ml`, `llm`, or
`heuristic`), then the checks `run` applies, printing one `blocked: <reason>`
line for each reason `run` would refuse to generate an input. It never writes
anything, even without `--json`, and exits 0 whether or not it is blocked.
`run` generates a complete runnable input (Quantum ESPRESSO by default) and
publishes it to `silicon-run` because `-o/--out` was given; omit `-o/--out`
for a memory-only preview instead. Use `uv run goldilocks --help` or a
subcommand's `--help` for full syntax.

`run` and `explain` share the same seven flags: the `structure` path,
`--code`, `--task {scf_single_point,dos,relax,vc-relax}`, `--hpc`, `--set
KEY=VALUE` (repeatable), `--config FILE.toml`, and `--json`; both also accept
`--fetch-missing`, and only `run` additionally accepts `-o/--out`. `--task` is
validated against the four listed choices; `--code` is passed through
unchecked — only `quantum_espresso` is a registered code today, so an
unrecognised `--code` silently produces a Quantum-ESPRESSO-shaped bundle
instead of failing.

`uv run goldilocks capabilities --json` dumps the same full payload HTTP's
`GET /capabilities` and the MCP `capabilities` tool return: `codes`, `tasks`
(with descriptions, step counts, and executables), `facts`, `settings`,
`pseudopotential_tables`, `hpc_profiles`, `models`, `warnings`, and `sources`.
Narrower slices are also available directly:

- `uv run goldilocks settings --json` — every `--set`-able key, its type,
  default, and sources (the same objects as `capabilities`'s `settings[]`;
  run this command for the current key count rather than trusting a number
  in these docs).
- `uv run goldilocks models --json` — every model `ml/registry.toml`
  registers, whether or not it is actually installed (see [Scientific
  controls](#scientific-controls)).
- `uv run goldilocks assets status --json` — installed pseudopotential-table
  and model assets.

See [Serve HTTP](#serve-http) and [Serve local MCP](#serve-local-mcp) for the
HTTP/MCP equivalents. `uv run goldilocks examples path` locates the bundled
example structures.

## Save output or print JSON

`explain` never writes anything, regardless of flags — it has no `-o/--out`.
`run` writes only when you pass it:

| Flag                  | Output                                                        |
| --------------------- | -------------------------------------------------------------- |
| `-o/--out DIRECTORY`  | Publish a new calculation directory (must not already exist)   |
| `-o/--out FILE.zip`   | Publish the same contents as a ZIP archive                     |
| (omit `-o`/`--out`)   | Memory-only preview to stdout; nothing is written to disk      |

Explicit destinations must not already exist; `run` raises an error rather
than overwrite one. Without `-o/--out`, `run` always builds the full bundle in
memory and prints a preview listing each file's path and byte size — there is
no auto-picked output directory (v1's `goldilocks_out`, `goldilocks_out_1`, …
naming no longer exists in any form).

`--json` changes the shape of `run`'s output rather than being combined with
an output flag: without `-o/--out` it prints `{"files": [...], "records":
{...}, "warnings": [...]}`; with `-o/--out` it prints `{"files": [...],
"kind": "directory"|"archive", "path": "..."}`. `explain --json` prints
`{"records": {...}, "warnings": [...], "blocking": [...]}`, where `blocking`
lists the distinct reasons `run` would refuse to generate an input and is empty
when its checks pass.

`--json` output and the published `goldilocks.json` are strict JSON: a value
with no finite number is `null`, never `Infinity` or `NaN`. For example, the
Gamma-only mesh (`k_index` 1) holds for every k-distance above its lower
bound, so its `k_distance_interval` is `[low, null]`.

A command writes only its own output to stdout: the `--json` document, or the
plain-text report without `--json`. Anything else printed while it runs, such
as the notice `mace` prints when it loads, output from a C extension, or a
child process's output, goes to stderr, so `--json` output always parses as
one JSON document. `serve` is the exception: `serve mcp` speaks JSON-RPC on
stdout itself.

## Scientific controls

Defaults are Quantum ESPRESSO single-point SCF, PBEsol, and efficiency-tier
pseudopotentials. See [scientific conventions](conventions.md) for numerical
defaults and override rules.

Every scientific parameter is reached through `--set KEY=VALUE` (repeatable)
or `--config FILE.toml` (same `KEY=VALUE` pairs, TOML-format, merged the same
way) — there are no dedicated per-parameter flags. `uv run goldilocks
settings` (or `--json`) is the single source of truth for every override key,
its type, default, and description; a selection of commonly used keys:

| Setting                             | Meaning                                                              |
| ------------------------------------ | --------------------------------------------------------------------- |
| `functional`                        | Exchange-correlation functional (default `PBEsol`)                    |
| `pseudo_table_id`                   | Pin a registered pseudopotential table id instead of automatic selection |
| `k_grid`                            | Explicit Monkhorst-Pack mesh, e.g. `--set 'k_grid=[8,8,8]'`            |
| `k_distance`                        | Target k-point spacing (heuristic default 0.15 for metals, 0.30 otherwise) |
| `smearing_type`                     | Smearing function when `occupations=smearing` (default `cold`)        |
| `degauss`                           | Smearing width in Ry (default 0.01)                                   |
| `spin_polarized`, `spin_orbit_coupling` | Force spin polarization / spin-orbit coupling                     |
| `use_vdw`, `method`                 | Force a dispersion correction; only `d3bj` is implemented, for PBE and PBEsol |
| `isolation`                         | Electrostatic boundary: `periodic`, `slab` (Quantum ESPRESSO's 2D truncation, `assume_isolated='2D'`) or `isolated` (Martyna-Tuckerman); see [boundary conditions](science.md#check-boundary-conditions-and-vacuum) |
| `conv_thr`, `mixing_beta`, `electron_maxstep` | SCF energy-convergence threshold, density-mixing factor, max SCF iterations |
| `u_by_element`                      | Hubbard U_eff per element in eV, replacing the built-in table, e.g. `--set 'u_by_element={"Ti": 4.0}'` |
| `hubbard_needs_correlation`         | `false` runs without Hubbard +U; `true` applies the built-in U table  |

Quote `--set` values that contain brackets or braces (e.g. `--set
'k_grid=[8,8,8]'`) so the shell does not glob-expand them; array and object
values must be valid JSON. Numbers must be finite: `inf`, `nan` and values
that overflow, such as `1e999`, are rejected as invalid settings. See
[Pseudopotentials](pseudopotentials.md) for how tables are chosen, pinned, and
checked for compatibility.

`is_metal`, `is_magnetic`, and `k_distance` (v2 epic 11) each resolve through
a `human > ml > llm > heuristic` priority: `explain --json`'s
`records[...].source` says which tier actually produced a value.
`is_metal`/`k_distance` become ML-backed once their PSDI-hosted asset is
installed (`uv run goldilocks assets install <id>` — see [Install and check
assets](#install-and-check-assets)); `is_magnetic` additionally needs `mace`,
`e3nn`, `sphericart`, and `ase`, none of which install via a `pip`/`uv`
extra of this project or of goldilocks-ml -- see [mMACE setup](mmace-setup.md)
for a complete, copy-pasteable walkthrough (also covered in [goldilocks-ml's
own README](https://github.com/stfc/goldilocks-ml#use-the-is_magnetic-classifier))
for the manual install (the `mace` fork it needs has no PyPI release, and
PyPI's own upload validation rejects a package that declares a direct git
dependency regardless, so this can never become an automatic extra). It
also needs the same `GOLDILOCKS_MACE_BACKBONE` checkpoint the [magnetic
orderings](#magnetic-orderings) `--rank-with-mmace` flag uses below --
confirmed empirically (2026-09-21): the published record declares no
automatic download for its mMACE backbone, so `is_magnetic` reuses
whichever checkpoint that variable already points at rather than fetching
a second copy. A model that is not set up (its asset not installed or
failing verification, or for `is_magnetic` the packages or checkpoint
missing) leaves the value to the next tier. Once a model is set up, an error
loading or running it fails the command instead of being replaced by the
heuristic answer. `is_magnetic` runs in a separate worker process, so mMACE
never changes the torch settings of the command or server that asked for
it. There is still no flag to point `run`/`explain` at a custom
model file of your own. `uv run goldilocks models --json` reports every
model, installed or not, so it also confirms whether one you expect to
be usable actually is.

## Magnetic orderings

```bash
uv run goldilocks magnetic-orderings structure.cif --json
uv run goldilocks magnetic-orderings structure.cif --rank-with-mmace
```

Lists a ferromagnetic guess plus any compensated antiferromagnetic
candidates `enumlib` finds (see [Check magnetism and spin-orbit
coupling](science.md#check-magnetism-and-spin-orbit-coupling)) for a
human or agent to choose among before generating any input file. Each
listed candidate's `--json` output carries its own `structure_content`
and `overrides` (a fresh CIF, plus a remapped `starting_magnetization`
for AFM candidates), enough to hand straight to `goldilocks run`/`explain`
without re-deriving anything.

`--rank-with-mmace` relaxes every candidate's magnetic moments on a frozen
mMACE potential energy surface and recommends the lowest energy-per-atom
one -- needs both a `GOLDILOCKS_MACE_BACKBONE` environment variable
pointing at a downloaded mMACE checkpoint file, and the same manual
`mace`/`e3nn`/`sphericart`/`ase` install described above. Missing either,
or an error inside the relaxation itself, degrades to an unranked listing
with a warning. Each ranking runs in its own process, one ranking at a time,
so a long ranking never holds up `explain` or `run`; a ranking process that
dies fails the request.

## Install and check assets

```bash
uv run goldilocks assets install default
uv run goldilocks assets status default
uv run goldilocks assets verify default
```

Use a profile, asset ID, or table ID. `default` installs the four prediction
models (`qrf-kpoints`, `metallicity-cgcnn`, `is-metal-classifier`,
`is-magnetic-classifier`) and a PBEsol efficiency PseudoDojo table;
`workbench` installs all registered models and tables. `install` also
repairs corrupt installations.

`run --fetch-missing` (or `explain --fetch-missing`) installs a missing
pseudopotential table instead of failing, then retries; it does not apply to
any other kind of missing dependency. See
[asset storage](pseudopotentials.md#find-stored-assets) for locations.

## Serve HTTP

```bash
uv run goldilocks assets install workbench
uv run --extra http goldilocks serve http
```

The server defaults to **http://127.0.0.1:8000**. `--host` and `--port` change
the address. `--static-root DIRECTORY` also serves a built Workbench; a relative
`DIRECTORY` is resolved against the launch directory once, at startup.

| Endpoint            | Purpose                                                                          |
| ------------------- | --------------------------------------------------------------------------------- |
| `GET /capabilities` | Codes, tasks, facts, settings, pseudopotential tables, HPC profiles, models, warnings |
| `POST /inspect`     | Structure inspection (inline `structure_content`, not a path)                     |
| `POST /explain`     | Analysis and advisor decisions, plus `blocking`: the reasons `/run` would refuse; never writes anything |
| `POST /run`         | Generate a runnable input; `respond_with: "json"` (default) returns a JSON summary, `respond_with: "archive"` returns ZIP bytes |
| `GET /health`       | Process health                                                                     |
| `GET /ready`        | Asset readiness; 503 for missing or corrupt assets                                |
| `GET /openapi.json` | Request and response schemas                                                      |

Interactive API documentation is at `/docs`.

### HTTP security

Keep the default localhost binding for local use. The server has no built-in
authentication; restrict network access with a firewall or authenticated reverse
proxy before exposing it. Configure TLS and request limits there.

HTTP and MCP accept inline structures and registered table IDs, not local
structure paths or publication paths. Server operators control installed
assets. `/run` never writes to the server's disk: on request it can return
ZIP bytes directly (`respond_with: "archive"`) or a JSON summary (the
default), but it always builds and serves the bundle in memory.

## Serve local MCP

Configure your MCP client to launch this command from the repository root:

```bash
uv run --extra mcp goldilocks serve mcp
```

The stdio tools are `capabilities`, `inspect_structure`, `explain`, and `run`.
MCP's `explain` tool returns the same `records`, `warnings`, and `blocking` as
`POST /explain`. MCP's `run` tool never writes to the server's filesystem — it
always returns only the published file list, every resolved decision (with its
source), and a warnings array the caller must relay verbatim. To obtain actual
file bytes, use the CLI (`goldilocks run ... -o/--out`) or the HTTP transport's
`/run` endpoint with `respond_with: "archive"`.
