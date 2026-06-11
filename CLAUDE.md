# Project Instructions

## Session Start

At the beginning of every session working on this project, read:

1. `.agents/` directory — contains project skill definitions
2. `AGENTS.md` — contains agent/workflow guidance for this project

---

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
