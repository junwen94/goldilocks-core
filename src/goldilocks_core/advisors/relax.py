"""relax: ion/cell relaxation parameters, for relax/vc-relax tasks.

New in v2 (v2 epic 7, #7). Previously these existed only as
directory-sketch names in the design doc -- `RelaxOptions`/
`VcRelaxOptions` marked as placeholders (goldilocks-qe-pw-parameter
-audit.md finding B1) -- so whatever eventually rendered a relax input
would have had to invent values on the spot, violating "generation only
translates". Every default here is copied verbatim from QE's own
`INPUT_PW.html` (checked 2026-09-11 per the design doc), not invented
or "improved" with unvalidated numbers.

**Minimal scope for this epic.** This epic's own generation rewrite
only touches DOS/SCF-scope rendering -- it does not render a relax or
vc-relax input at all. These two dataclasses exist so
`step_settings.py`'s `PwSettings.relax` field has a real type instead
of a placeholder `dict[str, float] | None`, closing that specific gap;
full relax parameter coverage (a real `advisors/relax.py` decision
cascade with human/llm overrides, `if_pos` per-atom constraints, etc.)
is v2 epic 10's job.

**Two real bugs from the QE parameter audit, fixed here, not just
noted:**

- **`cell_factor` direction (audit finding #3).** An earlier design-doc
  draft had this backwards: "raise it when the cell is expected to
  expand." QE's own docs say it must exceed the maximum linear
  *contraction* the cell will undergo -- the case that actually needs a
  larger value is compression (high-pressure relaxations, equation-of
  -state calculations), not expansion. Getting this backwards risks the
  pseudopotential table's radial grid being interpolated out of range
  under compression with no warning.
- **`ion_dynamics`/`cell_dynamics` coupling (audit finding B5).** QE
  requires `cell_dynamics='bfgs'` whenever `ion_dynamics='bfgs'` --
  vc-relax's only supported combination here, since this codebase does
  not model any alternate `ion_dynamics` value at all (that is e10's
  job). Rather than expose two independently settable fields that could
  be set inconsistently and only get caught by QE itself at runtime,
  `VcRelaxOptions.cell_dynamics` is a derived property of
  `ion_dynamics`, not its own field -- the invalid combination cannot
  be constructed.

`diagonalization='ppcg'` (audit finding #4: QE ended support for it in
December 2024) is not modelled anywhere in this codebase yet -- noted
here so whichever future epic adds an electron-diagonalization advisor
does not resurrect a deprecated option.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RelaxOptions:
    ion_dynamics: str = "bfgs"
    forc_conv_thr: float = 1.0e-3
    """a.u. (Ry/Bohr). QE's own official INPUT_PW default."""
    etot_conv_thr: float = 1.0e-5
    """a.u. QE's single-system default is 1.0e-4; this codebase instead
    scales with system size (`nat * etot_conv_thr_per_atom`, matching
    `advisors/convergence.py`) once a real `advisors/relax.py` exists
    to compute it -- this bare default is only used until then."""
    nstep: int = 50
    trust_radius_max: float = 0.8
    """Bohr."""
    trust_radius_min: float = 1.0e-3
    """Bohr."""
    trust_radius_ini: float = 0.5
    """Bohr."""
    remove_rigid_rot: bool = False


@dataclass(frozen=True, slots=True)
class VcRelaxOptions(RelaxOptions):
    cell_dofree: str = "all"
    press: float = 0.0
    """kbar."""
    press_conv_thr: float = 0.5
    """kbar."""
    cell_factor: float = 2.0
    """Must exceed the maximum linear *contraction* the cell will
    undergo, not expansion -- see this module's docstring, audit
    finding #3."""

    @property
    def cell_dynamics(self) -> str:
        """Always equals `ion_dynamics` -- QE requires this coupling
        (audit finding B5). A derived property, not an independent
        field, so the invalid combination cannot be constructed."""
        return self.ion_dynamics
