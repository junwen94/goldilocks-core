# Scientific conventions

Use this reference for units, defaults, and override rules. For what to check
before trusting a recommendation, see [Check your recommendations](science.md).

## Units

| Quantity                  | Unit          | Field                                              |
| ------------------------- | ------------- | --------------------------------------------------- |
| k-point spacing           | 1/Å           | setting `k_distance` (group `k_sampling`)          |
| Smearing width            | Rydberg (Ry)  | setting `degauss` (group `occupations`)            |
| Wavefunction cutoff       | Ry            | setting `ecutwfc_ry` (group `cutoffs`)             |
| Charge-density cutoff     | Ry            | setting `ecutrho_ry` (group `cutoffs`)             |
| SCF convergence threshold | Ry            | setting `conv_thr` (group `convergence`)           |
| Density-mixing strength   | Dimensionless | setting `mixing_beta` (group `convergence`)        |
| SCF iteration limit       | Integer       | setting `electron_maxstep` (group `convergence`)   |

Energies use Quantum ESPRESSO's Rydberg convention, not Hartree: 1 Ha = 2 Ry.
The SCF threshold controls the estimated self-consistency error; it is not a
k-point or cutoff convergence tolerance.

Every setting above is listed by `goldilocks settings` (with its group, type,
and default) and, once resolved for a structure, shown in the matching record
from `goldilocks explain`/the published `goldilocks.json` (`k_sampling`,
`occupations`, `cutoffs`, and `convergence` respectively). `k_distance`
defaults to `0.15` for metals and `0.30` otherwise; `degauss` defaults to
`0.01` Ry for metals; `ecutwfc_ry`/`ecutrho_ry` are decided together by the
`cutoffs` advisor from the selected pseudopotentials' own published
recommendations, deriving `ecutrho_ry` from `ecutwfc_ry` when a table (e.g.
PseudoDojo) does not publish it directly.

## K-point spacing and shifts

`k_distance` uses solid-state reciprocal lattice vectors, including the 2π
factor (matching AiiDA-QuantumESPRESSO's k-distance convention). For each
reciprocal vector `b_i`, the grid size is:

```text
N_i = max(1, ceil(round(|b_i| / k_distance, 5)))
```

Rounding to five decimal places avoids numerical noise at integer boundaries. A
smaller `k_distance` gives a denser mesh. The human-override, `k_distance`,
and `k_grid` paths all default to shift `[0, 0, 0]` (Gamma-centered) when no
explicit `shift` is given; there is no ML-based k-point model wired yet
(planned for epic #11). In Quantum ESPRESSO's `K_POINTS automatic` convention
this includes Γ (the reciprocal-space origin), even for even-sized meshes; a
shift flag of `1` means a half-grid shift on that axis.

## Defaults

| Setting                                       | Default                                                        |
| ---------------------------------------------- | ---------------------------------------------------------------- |
| Target                                        | Quantum ESPRESSO, SCF single point                             |
| Exchange-correlation functional               | `PBEsol`                                                       |
| Pseudopotential accuracy tier                 | `efficiency` (only usable tier today, see below)               |
| SCF threshold / total-energy threshold        | `conv_thr = nat × 0.2e-9` Ry / `etot_conv_thr = nat × 1e-5` Ry |
| Mixing / iteration limit                      | `mixing_beta 0.4` / `electron_maxstep 80`                      |
| Confirmed metal, or metallicity unconfirmed   | `smearing`, `cold`, `degauss` `0.01` Ry                        |
| Confirmed non-metal                           | `fixed`, no width                                              |

`conv_thr`/`etot_conv_thr` scale with the structure's atom count `nat`, not a
fixed constant: for Si (`nat=8`) this gives `conv_thr = 1.6e-9` Ry and
`etot_conv_thr = 8e-5` Ry, matching real `goldilocks explain` output.
`mixing_beta` and `electron_maxstep` are fixed defaults.

Today `efficiency` is the only usable pseudopotential accuracy tier: there is
no setting to request `precision`, and pinning a `precision`-tier
`pseudo_table_id` makes pseudopotential resolution `Unavailable` rather than
switching tiers (verified:
`--set pseudo_table_id=pseudodojo-pbesol-precision-sr` on `Si.cif` errors
with `accuracy is precision, requested efficiency`). This is a genuine
current gap, not a doc omission.

Metallicity comes from `is_metal` (`FieldState[Metallicity]`): a *confirmed*
non-metal gets `fixed` occupations; a confirmed metal, or a structure where
composition alone cannot confirm metallic character (`is_metal` is
`Unavailable`), gets `smearing`. This is deliberate -- a wrong-but-plausible
fixed-occupations run is worse than an unnecessary but correct smearing run.

Each per-step advisor module under `src/goldilocks_core/advisors/` documents
and owns its own heuristic default (e.g. `occupations.py` for smearing,
`convergence.py` for SCF thresholds, `functional.py` for the default
functional); there is no single parameters/policy module. `code`/`task`
defaults live on `inputs/task.py`'s `Task` class.

Spin, dispersion, and electronic-character heuristics are described in
[Check your recommendations](science.md). Pseudopotential compatibility
exceptions belong to the [table guide](pseudopotentials.md#choose-a-table).

## Override precedence

Fields below are `--set`-able keys (see `goldilocks settings` for the full,
live list) that `set_overrides.build_overrides` turns into a `RunOverrides`
object consumed by `service.advise()`/`generate()`. Leaving a key unset
leaves the choice to Goldilocks's heuristics (or, once wired, its ML/LLM
tiers -- currently none are wired; every setting today only has
`human`/`heuristic` sources, per `goldilocks settings`).

- `k_grid` wins over `k_distance`; either one bypasses the heuristic
  k-point sizing entirely (the `k_sampling.grid_and_distance_conflict` info
  warning fires if both are given).
- `smearing_type` and `degauss` are independent overrides in the
  `occupations` group; `degauss` defaults to `0.01` Ry (the metallic
  cold-smearing default) if smearing is chosen without an explicit value.
  `occupations='fixed'` always carries no `smearing_type`/`degauss`.
- `conv_thr`, `etot_conv_thr`, `mixing_beta`, and `electron_maxstep`
  override independently; an unspecified `conv_thr`/`etot_conv_thr` falls
  back to the per-atom-scaled formula (see Defaults above), not a fixed
  constant.
- The functional comes from the `functional` setting (default `PBEsol`).
  There is currently no setting to choose the pseudopotential accuracy
  tier -- it is hardcoded to `efficiency`; pinning a `precision`-tier
  `pseudo_table_id` will make pseudopotential resolution `Unavailable`
  rather than switch tiers (see Defaults above). This is a known current
  gap, not a doc omission.
- `spin_polarized` and `spin_orbit_coupling` are independent settings
  (group `magnetic`) consumed together by one combined advisor,
  `magnetic_config.py`, specifically so that enabling one can never
  silently clear or ignore the other.
- `use_vdw` explicitly enables or disables dispersion. `method` alone
  changes the method only if dispersion is otherwise enabled; it does not
  enable dispersion for a 3D or unknown structure. Enabled dispersion
  defaults to `d3bj` if no method is given.
- The single setting `pseudo_table_id` pins an explicit table id instead
  of automatic selection. An explicit table must still satisfy the
  already-decided functional/accuracy/relativistic-treatment/element-coverage
  requirements, or the whole pseudopotential resolution becomes
  `Unavailable` (naming which tables would have matched); pinning a table
  never changes the requested functional, and (today) accuracy can't be
  changed via any setting at all -- see the accuracy-tier gap noted above.

## Relativistic modes

`relativistic` is a derived value, not a settable field -- it is shown in
`goldilocks explain`/`goldilocks.json`, and as each pseudopotential table's
own `relativistic` classification in `goldilocks assets`/`capabilities`.

| `relativistic` value | Meaning                                           |
| --------------------- | -------------------------------------------------- |
| `scalar`              | Scalar relativistic effects, without explicit SOC |
| `full`                | Fully relativistic data, needed for explicit SOC  |
| `non-relativistic`    | No relativistic treatment                         |

It is always derived from `spin_orbit_coupling`: Goldilocks requests `full`
when SOC is enabled and `scalar` otherwise. There is no direct override for
the relativistic treatment itself. Pinning a `pseudo_table_id` whose own
relativistic classification does not match what `spin_orbit_coupling`
implies makes pseudopotential resolution `Unavailable` (it does not override
the derived requirement). Fully relativistic files alone do not enable SOC.
With SOC enabled, QE generation writes `noncolin = .true.` and
`lspinorb = .true.`; otherwise spin-polarized advice produces `nspin = 2`.
