# goldilocks-core

Upstream Python package for DFT input recommendation.

## What goldilocks-core Does

**One job**: turn (Structure + CalculationIntent + optional hints) into QE input files.

Pipeline (strict sequence, no stage reaches back):

```
Load → Analyse → Advise → Select → Generate
```

- **Load** — parse structure file, pure I/O
- **Analyse** — inspect structure, produce `StructureAnalysis` (metallicity, magnetic character, SOC relevance, dimensionality, disorder). Reports observations only, no recommendations.
- **Advise** — combine analysis + intent + hints → `AdviceBundle`, each decision carrying a provenance tag (`heuristic` / `ML` / `user_hint`)
- **Select** — translate `AdviceBundle` into code-specific concrete values (`QEParameterSet`)
- **Generate** — write QE `pw.x` / `ph.x` input files from `QEParameterSet`

Output is structured data (dataclasses) + optionally written input files.

## Ecosystem Position

Four-repo ecosystem: `goldilocks-data` → `goldilocks-models` → **`goldilocks-core`** → `goldilocks-webapp`

- **goldilocks-models** trains ML models and exports versioned artefacts (`manifest.json` + model file)
- **goldilocks-core** loads those artefacts and runs inference — it does not train models
- goldilocks-models **imports from goldilocks-core** — these APIs must stay stable:
  - `goldilocks_core.kmesh.build_kmesh_entries` — kindex schedule (changing requires retraining)
  - `goldilocks_core.infer_features` — structure feature extraction

## Code and Scope

- **First implementation**: Quantum ESPRESSO only
- **Architecture**: code-agnostic through Advise; Select and Generate are code-aware
- Optional extras: `goldilocks[aiida]` for AiiDA integration, `goldilocks[mlip]` for MACE pre-analysis

## Commands

```bash
uv sync --group dev                     # install with dev deps
uv run pytest                            # run tests
uv run ruff check src tests              # lint
uv run ruff format src tests             # format
uv run pre-commit run --all-files        # lint + test in one shot
```

## Code Style

- Ruff with `E`, `F`, `I` rules. Target Python 3.12.
- Dataclasses use `slots=True`. Frozen for immutable value objects.
- `from __future__ import annotations` at the top of every module.
- Domain modules, not generic buckets: no `helpers/`, no `utils/`, no `processing/`.
- Prefer one clear API over compatibility shims. Do not add legacy aliases, duplicate import paths, or wrapper modules unless the user explicitly asks for backward compatibility.
- `snake_case` for everything. No `CamelCase` except in string literals matching external formats.
- Type hints on public API surfaces. Internal functions can be looser.
- Docstrings: factual — what it does, what it returns, what it assumes. Not prose essays.

## What Doesn't Belong Here

- User auth, sessions, frontend code, WebSocket handlers, pod management — that's the application layer.
- AiiDA workflows, CalcJobs, execution/scheduler scripts — that's Runner.
- Jupyter notebooks — go in `notebooks/` (gitignored). Convert insights into tests.
- Large ML model files or pseudo libraries — `local_data/` is gitignored.

## Rules

- **Never push or merge directly to `main`.** All changes arrive through PRs.
- Every PR must close an issue (`Closes #N`).
- Track work status in GitHub Issues/PRs.
- Any GitHub issue, issue comment, PR description, or review comment written by an agent must explicitly say so and name the human it represents: `Written by an agent on behalf of <user>.`
- Use `uv`, not `pip`.

## Agent Workflow

Skills are in `.agents/skills/`: `catchup`, `plan`, `review`, `report`, `make-a-pr`, `write-a-test`, `write-docs`, `use-uv`, `dft-basics`, `github-cli`, `skill-creator`.

- Start sustained work with `catchup`.
- Use `plan` for multi-step changes; keep the issue body as the current plan.
- Use `review` before PRs or after substantial changes.
- Use `report` for handoff/progress comments.
- Use `make-a-pr` only after implementation, tests, and review are ready.
