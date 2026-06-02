---
name: dft-basics
description: Domain reference for DFT concepts used in this project — k-points, pseudopotentials, convergence, smearing, and spin-orbit coupling. Load when making decisions about physics-related code, adding new parameter advice, or understanding why certain defaults exist.
---

# DFT Basics

This project recommends DFT calculation inputs. Understanding the physics is required to make good recommendations. This reference covers concepts that appear repeatedly in the codebase.

Load this when you need to understand *why* a parameter has a certain default, not just *what* the default is.

This is practical guidance, not a replacement for code-specific documentation. Always verify syntax and edge cases against the target DFT code.

---

## k-points

DFT integrates over the Brillouin zone. k-points sample that integration. Too few → inaccurate. Too many → expensive.

**Grids:**
- Monkhorst-Pack-style grids are standard for periodic systems. They are specified as `(nk1, nk2, nk3)`.
- Gamma-centered means the grid includes Γ. In Quantum ESPRESSO `K_POINTS automatic`, shift `0 0 0` is unshifted/Gamma-centered; shift `1 1 1` applies half-grid offsets.
- Do not blindly assume "even grid = misses Γ". That depends on the grid convention and shift. Odd grids are often convenient for symmetric unshifted meshes, but code-specific semantics matter.
- k-point density is often specified as a spacing (Å⁻¹) rather than a grid. The code converts spacing to a grid using the reciprocal lattice.

**Trade-offs:**
- Metals usually need denser k-points than insulators because the Fermi surface is sharp.
- 2D and 1D systems need fewer k-points in non-periodic directions.
- The ML model in this package predicts a k-index that maps to a grid density.

## Pseudopotentials

Atoms have core electrons (tightly bound, chemically inert) and valence electrons (participate in bonding). Pseudopotentials replace the all-electron potential near the nucleus with an effective potential, reducing computational cost.

**Key distinctions:**

| Property | Values | Meaning |
|----------|--------|---------|
| Pseudo type | NC (norm-conserving), ultrasoft, PAW | How the valence charge is represented. NC is simpler; ultrasoft/PAW often reduce wavefunction cutoffs but need augmentation/charge-density treatment. |
| Relativistic treatment | scalar-relativistic, fully-relativistic | Scalar includes scalar relativistic effects but not explicit spin-orbit coupling. Fully-relativistic includes spin-orbit terms. |
| Functional | PBE, PBEsol, LDA, ... | The exchange-correlation functional the pseudo was generated for. |

**SSSP families:**
- **Efficiency** — chosen to reduce computational cost while keeping acceptable accuracy for screening/high-throughput work.
- **Precision** — chosen for stricter accuracy, usually with higher cutoffs and greater cost.

**Selection chain:**
1. Analyse structure → identify elements and relevant properties (heavy elements, magnetic candidates, etc.)
2. Advise → choose pseudo family and relativistic mode
3. Select → pick specific files per element and derive cutoffs from metadata

**SOC gotcha:** Spin-orbit calculations generally require fully-relativistic pseudopotentials. Whether scalar and fully-relativistic pseudos can be mixed is code- and setup-dependent; do not assume it is safe. A conservative workflow selects a consistent relativistic treatment and validates it against the target code.

## Smearing

Metals have a Fermi surface where occupation changes abruptly from 1 to 0. This causes convergence problems because small band-energy changes can cause large occupation changes. Smearing smooths the transition, making SCF convergence easier.

**Types:**
- **Methfessel-Paxton (MP)** — common for metals, but higher-order MP can produce negative occupations. First-order MP is common.
- **Cold smearing (Marzari-Vanderbilt)** — good general-purpose smearing with small energy bias. Often preferred for metallic systems.
- **Gaussian** — simple and conservative. Useful for testing or when a less aggressive smearing is desired.
- **Fixed occupations / no smearing** — appropriate for many insulators and semiconductors when band occupations are unambiguous.

**Typical guidance:**
- Metallic → MP or cold smearing, width roughly 0.01–0.02 Ry as a starting point
- Insulating → fixed occupations, or narrow Gaussian/cold smearing if needed for convergence

**The width matters:**
- Too wide → occupations and energies become unphysical
- Too narrow → SCF may fail to converge

## Spin-orbit Coupling (SOC)

Electrons in heavy atoms can have significant spin-orbit interaction. SOC splits degeneracies and can change band structures, magnetic anisotropy, and topological character.

**When SOC matters:**
- Heavy elements, especially period 5 and heavier (`4d`, `5d`, lanthanides/actinides, post-transition metals such as Pb/Bi)
- Materials where band splittings, topology, magnetism, or heavy-element chemistry are important
- Even one heavy element can make SOC relevant

**Cost:** SOC often costs several times more than a scalar-relativistic calculation because it uses spinor wavefunctions, reduces usable symmetries, and requires fully-relativistic pseudos.

**Implications for this codebase:**
- If `StructureAnalysis.contains_heavy_elements` is true, advise SOC consideration rather than silently enabling it
- SOC advice should drive pseudopotential selection toward fully-relativistic pseudos
- SOC should be overridable by user hints because relevance depends on the science question

## Convergence

DFT calculations are iterative. Convergence parameters control when the SCF loop stops.

**Key parameters:**
- **Energy convergence threshold** — stop when total energy change between steps is below this. In Quantum ESPRESSO, `conv_thr` is in Ry; `1e-6` Ry is a common default, tighter values are used for precision work.
- **Force convergence** — used for geometry optimization. In Quantum ESPRESSO, `forc_conv_thr` is in Ry/bohr; `1e-3` Ry/bohr is a common default-level threshold.
- **Mixing** — how the new potential/density is mixed with the old. Bad mixing causes oscillations or slow convergence.
  - Mixing beta too high → oscillations
  - Mixing beta too low → slow convergence
  - Metallic, magnetic, or large systems often need gentler mixing

**System-dependent tendencies:**
- Large systems → lower mixing beta, more SCF steps
- Metals → may need smearing and denser k-points
- Magnetic systems → spin-polarized calculations can converge more slowly

## Codes

Goldilocks may target multiple DFT codes. Support must be verified in the codebase; do not assume a target is implemented just because it is listed here.

Common target codes:
- **Quantum ESPRESSO** — plane-wave pseudopotential code. Input is Fortran namelists plus cards.
- **VASP** — plane-wave PAW code. Uses POSCAR, INCAR, KPOINTS, and POTCAR. POTCAR files are licensed, so tools should not ship them.
- **CASTEP** — plane-wave pseudopotential code. Input is `.cell` and `.param` files.
- **ONETEP** — linear-scaling DFT code, common in UK materials modelling.

Physics decisions should be made before code-specific generation. Generators, when present, should translate completed advice/selection records into code syntax rather than inventing new physical defaults.