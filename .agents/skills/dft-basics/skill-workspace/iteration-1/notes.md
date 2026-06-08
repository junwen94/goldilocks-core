# Iteration 1 notes

## Baseline
The old `SKILL.md` held all DFT guidance in one file. It was useful, but dense: agents had to scan k-points, pseudopotentials, smearing, SOC, convergence, and codes even when only one topic was relevant.

## Feedback used
Junwen's PR comments asked for:

- more explicit k-point convention guidance
- no blanket "even grid misses Gamma" assumption
- clearer k-mesh representations used by the package
- pseudopotential guidance covering NC, ultrasoft, PAW, functional, relativistic treatment, SSSP families, selection chain, and SOC gotchas
- clarification that PAW is usually frozen-core and does not imply fully-relativistic

The requested direction was progressive disclosure with detailed supporting docs.

## Revision
- `SKILL.md` is now a lean trigger/orientation file.
- Detailed physics moved into `references/`:
  - `k-points.md`
  - `pseudopotentials.md`
  - `smearing-soc-convergence.md`
- The main workflow emphasizes analysis → advice → selection → generation boundaries and provenance.

## Expected improvement
For focused physics tasks, agents should load one detailed reference instead of carrying the whole DFT primer in immediate context. For cross-cutting tasks, the main file names exactly which references to combine.
