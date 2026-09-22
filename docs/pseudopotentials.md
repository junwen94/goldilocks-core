# Pseudopotential tables

A table is a named collection of pseudopotential files in UPF format and
recommended energy cutoffs. Goldilocks supports tables from PseudoDojo and SSSP.
Start with the default table in the [quickstart](quickstart.md); use this page
to choose another registered table, or to find the assets and licence
material Goldilocks installs for you.

## Choose a table

List the registered choices and their supported elements, functional, accuracy,
and relativistic treatment with `uv run goldilocks capabilities --json` (see
the `pseudopotential_tables` array), or filter it directly in Python:

```python
from goldilocks_core.capabilities import capabilities

for table in capabilities()["pseudopotential_tables"]:
    print(table["id"], table["functional"], table["accuracy"], table["relativistic"])
```

(Or `GET /capabilities` once `goldilocks serve http` is running, or the MCP
`capabilities` tool — all four return the same payload.)

The `pseudopotential_tables` array lists the registered tables. Each table's
`elements` field lists elements permitted by Core's selection policy, not just
those present in the upstream library. `uv run goldilocks assets status` with
no argument only reports the `default` profile's single table; run
`uv run goldilocks assets status workbench` to see every registered table's
install state.

Without an explicit source, Goldilocks chooses a compatible registered table,
preferring PseudoDojo unless the structure contains lanthanides or actinides. It
does not choose a different table merely because that table is installed. The
`default` asset profile includes the scalar-relativistic PBEsol efficiency
table, plus the default `qrf-kpoints`, `metallicity-cgcnn`,
`is-metal-classifier`, and `is-magnetic-classifier` models.

To choose a table explicitly, pin `pseudo_table_id`: `--set
pseudo_table_id=<table-id>` on the CLI, `build_overrides({"pseudo_table_id":
"<table-id>"})` in Python (see the [Python tutorial](tutorial.md)), or
`{"overrides": {"pseudo_table_id": "<table-id>"}}` over HTTP/MCP. The pinned
table's functional, accuracy tier, relativistic treatment, and element
coverage must still match the request. In particular:

- Only `efficiency`-tier tables can currently be selected, automatically or by
  pinning: the required accuracy tier is hardcoded to `efficiency` throughout
  selection today, with no override anywhere. Pinning a `precision` table
  (for example `pseudodojo-pbesol-precision-sr`) fails with `pseudopotential
  table '...' does not satisfy the request: accuracy is precision, requested
  efficiency; matching tables: ...`. `precision` tables are registered but not
  yet reachable, automatically or explicitly.
- Table suffixes `sr` and `fr` mean scalar relativistic and fully relativistic.
  Explicit spin-orbit coupling (SOC) needs fully relativistic files; the default
  profile contains only an `sr` table.
- Lanthanides and actinides must use SSSP under Goldilocks' selection policy,
  with coverage checked for the actual elements. PseudoDojo's lanthanide table
  freezes 4f electrons in the core assuming trivalent ions, which is not
  suitable for all oxidation states; its tables do not cover actinides. No
  registered SSSP table is fully relativistic, so automatic or explicit
  registered-table selection cannot supply SOC for these elements.
- SSSP 1.3.0 PBEsol tables reuse PBE input parameters and cutoffs and were not
  tested with the SSSP convergence protocol. Do not treat these as
  PBEsol-validated cutoffs.
- Some files in SSSP scalar-relativistic tables declare non-relativistic
  treatment. Goldilocks permits this exception for scalar requests and
  preserves the per-file treatment; no warning is currently emitted for it.

In the Workbench, the table dropdown shows choices that support every element
in your structure at the selected functional; there is no accuracy or
relativistic-treatment filter or hint in the UI today (accuracy and
relativistic treatment appear only as descriptive text in each option's
label). Changing the functional clears any pinned table override. Selecting an
`fr` table does **not** enable SOC, and enabling `spin_orbit_coupling` does not
change the table selection — it is an independent setting, checked against
whichever table ends up selected. Direct Python, CLI, and HTTP requests must
still supply compatible requirements explicitly.

The [scientific guide](science.md#check-pseudopotentials-and-cutoffs) explains
what to check before using selected cutoffs.

## Install and select it

The commands below assume Goldilocks is installed. Replace `structure.cif` with
your ordered structure file. This example installs and selects fully
relativistic PBEsol files for an SOC recommendation:

```bash
uv run goldilocks assets install pseudodojo-pbesol-efficiency-fr
uv run goldilocks explain structure.cif --set spin_orbit_coupling=true --set pseudo_table_id=pseudodojo-pbesol-efficiency-fr
```

Use the [asset commands](cli.md#install-and-check-assets) if a required model or
table is missing.

Check installation state and verify file integrity:

```bash
uv run goldilocks assets status pseudodojo-pbesol-efficiency-fr
uv run goldilocks assets verify pseudodojo-pbesol-efficiency-fr
```

Status is `installed`, `missing`, or `corrupt`. Repeat `assets install` to
repair a corrupt installation. Verification checks stored files, not scientific
accuracy.

### Read installed metadata in Python

For ordinary calculations, pin the table ID with `pseudo_table_id` (see
[Choose a table](#choose-a-table) above, and the
[Python tutorial](tutorial.md)). To inspect its installed metadata directly:

```python
from goldilocks_core.assets.store import AssetStore
from goldilocks_core.assets.pseudopotentials.importers import load_installed_table
from goldilocks_core.assets.pseudopotentials.registry import load_tables

table = load_tables()["pseudodojo-pbesol-efficiency-fr"]
installed = AssetStore().resolve_spec(table.asset)
metadata = load_installed_table(installed, table=table)
for pseudo in metadata:
    print(pseudo.element, pseudo.filename, pseudo.cutoffs)
```

Managed PseudoDojo tables convert the provider's high cutoff hint from Hartree
to Ry and derive the charge-density cutoff using the registered multiplier
(currently 4). SSSP tables supply separate wavefunction and charge-density
cutoffs. Neither path optimizes cutoffs for your structure.

## Use your own UPF files

Not currently supported. There is no `--pseudo-root` flag, no custom-UPF-root
scanning, and no user-facing way to supply local UPF files, licence sidecars,
or scientific metadata for an arbitrary directory — `SystemOverrides` has no
`pseudo_root`/`pseudo_metadata` field, and the CLI, HTTP, and MCP surfaces only
accept a `pseudo_table_id` pinned to a *registered* table. The only way to
supply pseudopotentials today is automatic selection or `pseudo_table_id`,
described above.

(`parse_upf_metadata` in
`goldilocks_core.assets.pseudopotentials.upf` still exists, but it is an
internal UPF-header parser used by the asset-install pipeline, not a
public entry point for bringing your own files.)

## Find stored assets

The default store is `$XDG_DATA_HOME/goldilocks/assets`, or
`~/.local/share/goldilocks/assets` when `XDG_DATA_HOME` is unset.
`GOLDILOCKS_ASSET_ROOT` overrides it. Table installations live at:

```text
<asset-store>/pseudopotentials/<table-id>/<version>/
```

Treat managed files as read-only; use `assets install`, `status`, and `verify`.

## Licences and citations

UPFs retain their upstream licences; the Goldilocks BSD licence does not apply
to them. UPFs are downloaded separately, not bundled in the package.

- [PseudoDojo](https://www.pseudo-dojo.org/): registered tables use CC BY 4.0.
  Cite van Setten et al., _Computer Physics Communications_ 226, 39–54 (2018).
- [SSSP 1.3.0](https://archive.materialscloud.org/records/rcyfm-68h65): cite
  Prandini et al., _npj Computational Materials_ 4, 72 (2018), and the data
  record. The record's CC BY 4.0 licence does not replace the individual UPF
  licences. Read its mixed-family
  [`LICENSE.txt`](https://archive.materialscloud.org/records/rcyfm-68h65/files/LICENSE.txt?download=1)
  before redistribution.

Installations store licence material in `LICENSE.txt`: the CC BY 4.0 notice for
PseudoDojo, or the upstream mixed-family licence file for SSSP. Published
calculation outputs include that material with the selected UPFs.
