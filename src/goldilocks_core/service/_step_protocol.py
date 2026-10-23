"""``StepProtocol``: a task's own defaults for one of its pw.x steps (#174).

A protocol is a tier of its own for the step it is given to: it ranks
below the request's overrides and above the ML tier, for that step only
(``advisors/k_sampling.py``, ``advisors/occupations.py``). Its values are
not request values, so their records read ``heuristic`` with the protocol
named in ``source_note``; ``human`` only ever comes from the request. A
request value replaces the protocol's field by field: a ``shift`` alone
keeps the protocol's k_distance, a ``degauss`` alone its occupations.

Kept apart from ``StepOverrides`` (``_step.py``), which carries request
values only and is walked into ``capabilities``' settings.
"""

from __future__ import annotations

from dataclasses import dataclass

from goldilocks_core.advisors.k_sampling import KSamplingProtocol
from goldilocks_core.advisors.occupations import OccupationsProtocol


@dataclass(frozen=True, slots=True)
class StepProtocol:
    """The protocol fields a step's advisors read; ``None`` leaves that
    advisor to its own tiers."""

    k_sampling: KSamplingProtocol | None = None
    occupations: OccupationsProtocol | None = None


_DOS_NSCF_NOTE = (
    "Protocol default for the dos task's nscf step: "
    "aiida-quantumespresso's pdos.yaml, balanced."
)

DOS_NSCF_PROTOCOL = StepProtocol(
    k_sampling=KSamplingProtocol(k_distance=0.10, note=_DOS_NSCF_NOTE),
    occupations=OccupationsProtocol(occupations="tetrahedra_opt", note=_DOS_NSCF_NOTE),
)
"""The dos task's nscf step (``service/_dos.py``), from aiida-quantumespresso's
``workflows/protocols/pdos.yaml`` "balanced" protocol (confirmed against its
source 2026-09-14): ``kpoints_distance`` 0.10 1/angstrom, against the scf
heuristic's 0.15/0.30, and ``tetrahedra_opt`` occupations, a DOS-quality
integration grid. That protocol also sets ``nosym`` for the nscf step; this
one does not, because the generated input does not write ``nosym``/``noinv``
(``generation/quantum_espresso/scf.py``), and a protocol value with no effect
on the rendered file would mislead."""
