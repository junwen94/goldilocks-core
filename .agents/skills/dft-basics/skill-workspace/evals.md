# dft-basics skill evals

## eval-1: k-point convention change
Prompt: "I'm changing `k_distance_to_mesh`; what DFT convention details should I check?"
Why it matters: Should trigger the k-points reference and warn about reciprocal-lattice `2π`, Gamma shifts, and even-grid assumptions.

## eval-2: SOC pseudopotential selection
Prompt: "Add advice for heavy-element structures so pseudopotential selection handles SOC correctly."
Why it matters: Should load pseudopotentials plus SOC guidance and avoid assuming PAW means fully-relativistic.

## eval-3: baseline test coverage
Prompt: "Add tests around current k-mesh and pseudo behaviour before refactoring."
Why it matters: Should combine DFT basics with write-a-test and focus on public behaviour/regression tests.
