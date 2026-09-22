# goldilocks-core

Goldilocks recommends settings for density functional theory (DFT) calculations
and generates Quantum ESPRESSO input files (SCF, DOS, relaxation, and
variable-cell relaxation) from a crystal structure.

## Installation

Not published to PyPI yet -- clone the repository. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) first:

```bash
git clone https://github.com/junwen94/goldilocks-core.git
cd goldilocks-core
uv sync
```

## Try it

### Start the Workbench

With Node.js 24 or newer installed, run:

```bash
uv sync --extra http
npm --prefix web ci
uv run goldilocks assets install workbench
uv run --extra http poe workbench
```

The asset step installs the models and pseudopotential tables. Open
**http://127.0.0.1:5173**, upload a CIF or POSCAR, review the recommended
settings, and download the generated inputs.

For a built frontend instead, stop the development servers and run:

```bash
uv run --extra http poe stage
```

Then open **http://127.0.0.1:8000**. See the [Workbench guide](web/README.md)
for Docker and development checks.

#### With mMACE (ML-backed magnetism features)

Without any extra setup, magnetism classification (`is_magnetic`) and
magnetic-ordering ranking run at a heuristic/LLM tier. To get the real ML
tier, install `mace`/`e3nn`/`sphericart` and a checkpoint file once, manually
-- none of this can ever be a `pip`/`uv` extra (the `mace` fork it needs has
no PyPI release):

```bash
uv pip install ase==3.28.0 e3nn==0.4.4 sphericart==1.0.9 sphericart-torch==1.0.9
uv pip install "mace-torch @ git+https://github.com/CheukHinHoJerry/mace.git@19cdf6692c48e068a24e06cfe1ffc670e8aea3dd"
mkdir -p ~/.local/share/goldilocks/mmace
curl -L -o ~/.local/share/goldilocks/mmace/mace_matpes_pbe_baseline_run-3.model \
  https://data-collections.psdi.ac.uk/api/records/1g8rw-q8128/files/mace_matpes_pbe_baseline_run-3.model/content
export GOLDILOCKS_MACE_BACKBONE=~/.local/share/goldilocks/mmace/mace_matpes_pbe_baseline_run-3.model
```

Then start the Workbench as above **in the same shell** (the backend only
picks up `GOLDILOCKS_MACE_BACKBONE` if it's set before launch). Load a
magnetic structure (e.g.
`src/goldilocks_core/examples/structures/Fe_bcc.cif`) and check the
**Analysis** column's "is magnetic" field: its caption switches to
**"Goldilocks-ML prediction"** once the `ml` tier is live.

Two gotchas worth knowing up front: `uv sync` silently removes the two
manually-installed packages again (they're not in `uv.lock`) -- re-run the
`uv pip install` lines above after any `uv sync`; and the Workbench's own
"Run mMACE" ranking button is currently broken
([stfc/goldilocks-ml#95](https://github.com/stfc/goldilocks-ml/issues/95)) --
use `uv run goldilocks magnetic-orderings --rank-with-mmace` from the CLI for
ranking instead. Checksum verification and full troubleshooting:
[mMACE setup](docs/mmace-setup.md).

### Generate inputs from the command line

Download the prediction models and default pseudopotential table, then generate
inputs for the bundled silicon structure:

```bash
uv run goldilocks assets install default
uv run goldilocks run src/goldilocks_core/examples/structures/Si.cif --out si-run
```

Open `si-run/scf.in` to see the input. The directory also contains the
pseudopotential file, a submission script, and `goldilocks.json` (full
provenance for every setting).

Treat the recommended settings as a starting point: review warnings and check
convergence for your calculation. The [quickstart](docs/quickstart.md) explains
the output and how to run it.

## Guides and reference

- [First calculation](docs/quickstart.md) — generate, check, and run an input.
- [Python API](docs/tutorial.md) — use Goldilocks in a script.
- [Recommendations](docs/science.md) — understand the choices and their limits.
- [Pseudopotentials](docs/pseudopotentials.md) — choose a table and understand
  automatic selection.
- [CLI reference](docs/cli.md) — commands and options.
- [mMACE setup](docs/mmace-setup.md) — enable the ML-backed magnetism
  features.
- [Scientific conventions](docs/conventions.md) — units and numerical
  definitions.
- [Contributing](docs/architecture.md) — code layout and development checks.

## Licence

Code: [BSD 3-Clause](LICENSE). Documentation under `docs/` and example
structures: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Downloaded pseudopotentials retain their
[upstream licences](docs/pseudopotentials.md#licences-and-citations).
