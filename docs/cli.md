# CLI Reference

## Installation

```bash
uv sync
```

After that, `gl` is available via `uv run gl ...`.

---

## Commands

### `gl` — interactive wizard

Running `gl` with no arguments (or `gl wizard`) opens the interactive menu:

```
1) Pre-Analysis    — load a structure file and display analysis results
2) Input Kit       — collect task/accuracy/hints and show QE parameter recommendations
3) Submit Playground  — coming soon
4) Results Lab        — coming soon
0) Quit
```

**Pre-Analysis** (menu option 1): first asks how to source the structure:

- **Load local file** — load and analyse a CIF/POSCAR/XSF file directly.
- **Search database** — query Materials Cloud (MC3D) and NOMAD in parallel via their public OPTIMADE endpoints, or provide a structure file to extract the formula automatically. Displays a results table with entry IDs and clickable URLs. No external API key required.

**Input Kit** (menu option 2): optionally preceded by Pre-Analysis if no structure has been loaded yet. Prompts for:
- Calculation task (default: `scf`)
- Accuracy tier (default: `balanced`)
- Optional parameter hints (`key=value`)

Then runs the advise pipeline and displays the full QE parameter recommendation.

---

### `gl input` — non-interactive command

Runs the full pipeline from a single command. Useful for scripting.

```
gl input [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--structure` | `-s` | **required** | Structure file (CIF, POSCAR, XSF, …) |
| `--task` | `-t` | `scf` | Calculation task: `scf` \| `relax` \| `vc-relax` \| `nscf` \| `bands` \| `md` \| `vc-md` |
| `--accuracy` | `-a` | `balanced` | Accuracy tier: `fast` \| `balanced` \| `accurate` |
| `--code` | `-c` | `qe` | DFT code (currently only `qe`) |
| `--xc` | | `pbesol` | XC functional (currently only `pbesol`) |
| `--pseudo` / `--pp` | | PseudoDojo SR | Pseudo family. Defaults to SR; auto-upgrades to FR for SOC structures when not set explicitly. |
| `--hints` | `-H` | | Parameter overrides as `key=value`, repeatable |
| `--explain` | `-e` | off | Print full rationale for every decision |

#### Examples

Minimal — just the structure file:

```bash
gl input -s Fe.cif
```

Relaxation with accurate settings and full rationale:

```bash
gl input -s Fe.cif -t relax -a accurate -e
```

Force fully-relativistic pseudopotentials:

```bash
gl input -s FeBi.cif --pseudo PseudoDojo/0.4/PBEsol/FR/standard/upf
```

Override spin treatment and wavefunction cutoff:

```bash
gl input -s Fe.cif -H spin_treatment=collinear -H ecutwfc_ev=680
```

All options explicit:

```bash
gl input --structure Fe.cif --task scf --accuracy balanced --code qe --xc pbesol
```

---

## Pseudo families

| Family string | Relativistic | Notes |
|--------------|-------------|-------|
| `PseudoDojo/0.4/PBEsol/SR/standard/upf` | Scalar | Default |
| `PseudoDojo/0.4/PBEsol/FR/standard/upf` | Fully relativistic | Auto-selected when SOC is relevant |

When `--pseudo` is not set, the advise layer picks SR by default and upgrades to FR automatically if the structure contains heavy elements (period 5+).

---

## Hints reference

Hints override specific parameters in the advise pipeline. Pass them with `-H key=value`.

| Key | Values | Effect |
|-----|--------|--------|
| `spin_treatment` | `non_magnetic` \| `collinear` \| `non_collinear` \| `non_collinear_soc` | Override spin/SOC treatment |
| `ecutwfc_ev` | float (eV) | Override wavefunction cutoff |
| `ecutrho_ev` | float (eV) | Override charge density cutoff |
| `initial_magnetization` | dict literal | Override starting magnetic moments (μB) |

Example:

```bash
gl input -s Fe.cif -H spin_treatment=collinear -H ecutwfc_ev=680
```
