# Project Instructions

## Session Start

At the beginning of every session working on this project, read:

1. `.agents/` directory — contains project skill definitions
2. `AGENTS.md` — contains agent/workflow guidance for this project

---

## What goldilocks-core Does

**One job**: turn (Structure + CalculationIntent + optional hints) into a structured parameter recommendation. It does not write DFT input files — that is done downstream.

Pipeline (strict sequence, no stage reaches back):

```
Load → Analyse → Advise → Select
```

- **Load** — parse structure file, pure I/O
- **Analyse** — inspect structure, produce structured facts (metallicity, magnetic character, SOC relevance, dimensionality, disorder). Reports observations only, no recommendations.
- **Advise** — combine analysis + intent + hints → parameter recommendations, each with a provenance tag (`analysis` / `default` / `user_hint`)
- **Select** — turn abstract advice into concrete values: integer k-grid, specific pseudo files, cutoffs derived from pseudo metadata

Output is structured data (dataclasses), not files.

## Ecosystem Position

Four-repo ecosystem: `goldilocks-data` → `goldilocks-models` → **`goldilocks-core`** → `goldilocks-webapp`

- **goldilocks-models** trains ML models and exports versioned artefacts (`manifest.json` + model file)
- **goldilocks-core** loads those artefacts and runs inference — it does not train models
- goldilocks-models **imports from goldilocks-core**:
  - `goldilocks_core.kmesh.build_kmesh_entries` — kindex schedule (must stay stable; changing it requires retraining)
  - `goldilocks_core.infer_features` — structure feature extraction (currently misnamed `extract_cslr_features`)

## Code and Scope

- **First implementation**: Quantum ESPRESSO only
- **Architecture**: code-agnostic from the start — `CalculationIntent` carries the `code` field; Select is the only stage that is code-aware
- Generate and Bundle stages are out of scope for this package

## Current Module State (as of 2026-06-10)

Working but needs restructuring:

| Module | Status |
|--------|--------|
| `io/structures.py` | Load stage — OK |
| `io/structures.py:analyze_structure()` | Analyse stage — too thin (only 3 boolean flags) |
| `kmesh.py` | k-distance math used by Select; must stay stable |
| `ml/features.py` | Feature extraction — public name should be `infer_features`, not `extract_cslr_features` |
| `ml/models.py` | Model loading — needs to read from manifest directory, not `ModelSpec` |
| `advisors/kmesh_advisor.py` | Conflates Advise + Select; needs splitting |
| `pseudo/` | Partial Select for pseudos; not yet connected to Advise stage |
| `shared/types.py` | All types in one file — will need splitting as scope grows |
| No `CalculationIntent` type | Missing first-class input object |
