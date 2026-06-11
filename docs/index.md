# goldilocks-core

AI-powered DFT input recommendation for crystal structures.

## What it does

Given a structure file and a calculation goal, `goldilocks-core` recommends DFT parameters — k-mesh, pseudopotentials, cutoffs, smearing, spin treatment — and generates ready-to-run QE input files.

```
structure + intent → Load → Analyse → Advise → Generate → gl-pw-scf.in
```

Every recommended value carries a provenance tag (`heuristic` / `ML` / `user_hint`) and a plain-English rationale.

## 30-second quick start

Install and sync:

```bash
git clone https://github.com/stfc/goldilocks-core
cd goldilocks-core
uv sync --group dev
```

Run the interactive wizard:

```bash
uv run gl
```

Or run non-interactively:

```bash
uv run gl input -s Fe.cif
uv run gl input -s Fe.cif -t relax -a accurate --explain
```

Generate QE input files:

```bash
uv run gl input -s Fe.cif --output ./run
# writes run/run_001/gl-pw-scf.in + copies pseudopotentials
```

## Documentation

- [CLI Reference](cli.md) — `gl` wizard and `gl input` options
- [Architecture](architecture.md) — pipeline design and module responsibilities
