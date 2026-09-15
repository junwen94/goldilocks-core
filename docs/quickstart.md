# First calculation

Generate a Quantum ESPRESSO SCF input for silicon, check the result, and run it.
For Python, see the [Python guide](tutorial.md).

Already generated `si-run` from the README? Continue with
[checking the settings](#3-check-the-settings).

## 1. Install

Follow the [repository setup](../README.md#try-it), then run these commands from
the repository directory:

```bash
uv run goldilocks assets install default
uv run goldilocks assets verify default
```

This downloads and checks the k-point model, the metallicity classifier, and the
PseudoDojo PBEsol efficiency pseudopotential table. The first installation needs
an internet connection.

## 2. Generate an input

```bash
uv run goldilocks run src/goldilocks_core/examples/structures/Si.cif --out si-run
```

Replace the bundled silicon path with your own CIF or POSCAR when you are ready.
Choose a new output directory each time; Goldilocks will not overwrite an
existing one. (Only one HPC profile is installed by default, so `--hpc` can be
omitted here; pass `--hpc <profile>` explicitly if more than one is available.)

The main files in `si-run/` are:

| Path                         | Contents                                                                                        |
| ---------------------------- | ------------------------------------------------------------------------------------------------ |
| `scf.in`                     | Quantum ESPRESSO input, named after the task (`relax.in`, `vc-relax.in`, or `scf.in`+`nscf.in`+`dos.in` for `--task dos`) |
| `pseudo/`                    | The selected UPF pseudopotential file(s)                                                         |
| `submit.sh`                  | Generated SLURM submission script for the chosen `--hpc` profile                                 |
| `goldilocks.json`            | Settings, provenance, and file hashes for every record and file                                  |
| `README.md`, `CITATIONS.md`  | Bundle contents summary and citation information                                                 |

## 3. Check the settings

Read any warnings printed by the command and open `si-run/scf.in`. Check
the k-point grid, energy cutoffs, occupations, and spin settings against what
you know about your material.

Model predictions and table cutoffs are starting points, not convergence tests.
See [Recommendations](science.md) for how each choice is made and
[Scientific conventions](conventions.md) for units.

To choose a grid yourself, generate a second input with `--set k_grid=...`:

```bash
uv run goldilocks run src/goldilocks_core/examples/structures/Si.cif --set 'k_grid=[4,4,4]' --out si-grid-4
```

The value must be a valid JSON array (`--set k_grid=4,4,4` fails with a JSON
parse error), and quoting the whole `key=value` pair keeps the shell from
trying to glob-expand the brackets. The `4 × 4 × 4` grid demonstrates an
override; it is not a converged value for silicon. Run
`uv run goldilocks settings` to see every other `--set`-able key, or see the
[CLI reference](cli.md#scientific-controls).

## 4. Run Quantum ESPRESSO

With Quantum ESPRESSO installed, run `pw.x` **from inside the output directory**
so it can find `./pseudo`:

```bash
cd si-run
pw.x -in scf.in > scf.out
```

Review `scf.out` to check that the SCF calculation converged. On an HPC
cluster, submit the generated `submit.sh` instead — it already runs `pw.x`
with the right MPI/OpenMP layout for the `--hpc` profile you chose.

## Next

- [Python API](tutorial.md) — inspect structures and generate inputs in a
  script.
- [Pseudopotentials](pseudopotentials.md) — change the functional or
  pseudopotential table.
- [Workbench](../web/README.md) — prepare and review inputs in a browser.
