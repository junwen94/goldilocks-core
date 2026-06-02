---
name: dft-basics
description: Domain reference for DFT concepts used in this project — k-points, pseudopotentials, convergence, smearing, and spin-orbit coupling. Load when making decisions about physics-related code, adding new parameter advice, or understanding why certain defaults exist.
---

# DFT Basics

This project recommends DFT calculation inputs. Understanding the physics is required to make good recommendations. This reference covers the concepts that appear repeatedly in the codebase.

Load this when you need to understand *why* a parameter has a certain default, not just *what* the default is.

---

## k-points

DFT integrates over the Brillouin zone. k-points sample that integration. Too few → inaccurate. Too many → expensive.

**Grids:**
- Monkhorst-Pack grids are the standard. Specified as `(nk1, nk2, nk3)`.
- For Gamma-centered grids (common in Quantum ESPRESSO), each dimension should be **odd** so Gamma is included. Even grids miss Gamma and shift the sampling.
- k-point density is often specified as a spacing (Å⁻¹) rather than a grid — the code converts spacing to a grid based on the reciprocal lattice.

**Trade-offs:**
- Metals need denser k-points than insulators (the Fermi surface is sharp).
- 2D and 1D systems need fewer k-points in the non-periodic directions.
- The ML model in this package predicts a k-index that maps to a grid density.

## Pseudopotentials

Atoms have core electrons (tightly bound, chemically inert) and valence electrons (participate in bonding). Pseudopotentials replace the all-electron potential with an effective potential that freezes the core, reducing the basis set size.

**Key distinctions:**

| Property | Values | Meaning |
|----------|--------|---------|
| Pseudo type | NC (norm-conserving), ultrasoft, PAW | How the valence charge is represented. NC is simpler; PAW/ultrasoft are more efficient but need more cutoffs. |
| Relativistic treatment | scalar-relativistic, fully-relativistic | Scalar ignores spin-orbit coupling. Fully-relativistic includes it. |
| Functional | PBE, PBEsol, LDA, ... | The exchange-correlation functional the pseudo was generated for. |

**SSSP families:**
- **Efficiency** — softer pseudos, lower cutoffs, faster calculations. Good for initial testing and high-throughput.
- **Precision** — harder pseudos, higher cutoffs, more accurate. Good for publication-quality results.

**Selection chain (how this code works):**
1. Analyse structure → identify elements and properties (SOC candidates, etc.)
2. Advise → choose pseudo family (efficiency vs precision), relativistic mode
3. Select → pick specific files per element, derive cutoffs from metadata

**Gotcha:** Fully-relativistic pseudopotentials are needed for spin-orbit coupling calculations. You cannot mix scalar and fully-relativistic pseudos for different elements — the calculation must use one treatment consistently.

## Smearing

Metals have a Fermi surface where occupation changes abruptly from 1 to 0. This causes convergence problems in DFT because small changes in band energy cause large changes in occupation. Smearing spreads the transition over a width, making SCF convergence smooth.

**Types:**
- **Methfessel-Paxton (MP)** — good for metals, but can produce negative occupations at high order. First-order MP is standard. Variational energy correction exists.
- **Cold smearing (Marzari-Vanderbilt)** — nearly unbiased free energy, good general-purpose choice. Preferred for many systems.
- **Gaussian** — simple, always positive occupation. Very conservative. Useful for insulators or testing.

**Defaults by system type:**
- Metallic → MP or cold, width ~0.01–0.02 Ry
- Insulating → cold or Gaussian, width ~0.001–0.01 Ry (or no smearing at all)

**The width matters:**
- Too wide → free energy differs from internal energy, unphysical results
- Too narrow → SCF won't converge

## Spin-orbit Coupling (SOC)

Electrons in heavy atoms have significant spin-orbit interaction. This splits degeneracies and changes band structures. It's expensive — it doubles the number of spin components and requires fully-relativistic pseudopotentials.

**When SOC matters:**
- Elements with high Z (period 5 and below — Hf, Ta, W, Re, Os, Ir, Pt, Au, Bi, Pb, etc.)
- Even if only one heavy element is present, SOC can be important

**Cost:** SOC is roughly 4-8× more expensive than a scalar-relativistic calculation.

**Implications for this codebase:**
- If `StructureAnalysis.contains_heavy_elements` is true, advise SOC consideration
- SOC advice means the selector must pick fully-relativistic pseudopotentials
- SOC can be forced on or off via user hints

## Convergence

DFT calculations are iterative. Convergence parameters control when the SCF loop stops.

**Key parameters:**
- **Energy convergence threshold** — stop when total energy change between steps is below this. Typical: 10⁻⁶ Ry for well-converged, 10⁻⁴ for quick tests.
- **Force convergence** — stop when all forces are below this. Used for geometry optimization. Typical: 10⁻³ Ry/bohr.
- **Mixing** — how the new potential is mixed with the old. Bad mixing → oscillations → no convergence.
  - Beta (mixing parameter): too high → oscillations, too low → slow convergence
  - Mixing scheme: plain, TF (Thomas-Fermi), local-TF

**System-dependent defaults:**
- Large systems → lower mixing beta, more SCF steps
- Metals → may need more steps and wider smearing for convergence
- Magnetic systems → spin-polarized calculations converge slower

## Codes

This package generates input files for multiple DFT codes. Each has its own syntax and conventions.

- **Quantum ESPRESSO** — plane-wave pseudopotential code. Input is Fortran namelists. Well-supported in this package.
- **VASP** — plane-wave PAW code. Uses POTCAR, POSCAR, INCAR, KPOINTS files. POTCAR files are licensed — this package emits a POTCAR.spec instead.
- **CASTEP** — plane-wave pseudopotential code. Input is .cell and .param files.
- **ONESTEP** — plane-wave pseudopotential code, used in UK HPC community.

The generator for each code receives the same advice and selection records. Physics decisions are made before generation — generators are mechanical translators.