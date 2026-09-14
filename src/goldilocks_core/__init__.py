"""goldilocks_core: DFT input advice and generation for Quantum ESPRESSO.

Deliberately re-exports nothing (v2 epic 9, #9, cutover). v1's
``__init__.py`` flattened its whole ``Service``/``compute``/
``ComputeRequest`` surface here; v2 has no equivalent single facade to
re-export, by design (``service/__init__.py``'s own docstring: "the
public API stays flat" at *that* level, not this one). The real entry
points are the packages themselves:

- ``goldilocks_core.service`` -- ``advise``/``check``/``generate`` (and
  the ``dos`` task's ``advise_dos``/``check_dos``/``generate_dos``), the
  programmatic API
- ``goldilocks_core.cli`` -- the ``goldilocks`` command-line entry point
- ``goldilocks_core.server`` -- the HTTP and MCP transports
"""

from __future__ import annotations
