from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pymatgen.core import Structure


@dataclass(frozen=True)
class ParameterHints:
    """Typed parameter overrides for the goldilocks-core advise pipeline.

    All fields default to ``None``, meaning "no override".
    Use :meth:`from_dict` to construct from a raw key→value mapping (e.g.
    from CLI ``-H key=value`` parsing).
    """

    # Spin
    spin_treatment: Literal["non_magnetic", "collinear", "non_collinear", "non_collinear_soc"] | None = None
    initial_magnetization: dict[str, float] | str | None = None

    # Smearing
    smearing_method: Literal["marzari_vanderbilt", "methfessel_paxton", "fermi_dirac", "gaussian"] | None = None
    smearing_width_ev: float | None = None

    # K-points
    kpoints_grid: tuple[int, int, int] | None = None
    kpoints_shift: tuple[int, int, int] | None = None
    k_distance: float | None = None

    # Van der Waals
    use_vdw: bool | None = None
    vdw_method: Literal["d3", "d3bj", "ts", "mbd"] | None = None

    # Pseudopotentials
    pseudo_family: str | None = None
    pseudo_dir: str | None = None

    # Cutoffs
    ecutwfc_ev: float | None = None
    ecutrho_ev: float | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ParameterHints":
        """Parse raw key→value pairs into a validated :class:`ParameterHints`.

        Raises :exc:`ValueError` on unknown keys so CLI errors surface early.
        """
        _KNOWN: frozenset[str] = frozenset({
            "spin_treatment", "initial_magnetization",
            "smearing_method", "smearing_width_ev",
            "kpoints_grid", "kpoints_shift", "k_distance",
            "use_vdw", "vdw_method",
            "pseudo_family", "pseudo_dir",
            "ecutwfc_ev", "ecutrho_ev",
        })
        unknown = set(d) - _KNOWN
        if unknown:
            raise ValueError(
                f"Unknown hint key(s): {sorted(unknown)}. "
                f"Valid keys: {sorted(_KNOWN)}"
            )
        kwargs: dict[str, Any] = {}
        for k, v in d.items():
            if k in ("kpoints_grid", "kpoints_shift"):
                kwargs[k] = v if isinstance(v, tuple) else (int(v[0]), int(v[1]), int(v[2]))
            elif k in ("smearing_width_ev", "k_distance", "ecutwfc_ev", "ecutrho_ev"):
                kwargs[k] = float(v)  # type: ignore[arg-type]
            elif k == "use_vdw":
                if isinstance(v, str):
                    kwargs[k] = v.lower() not in ("false", "0", "no", "off")
                else:
                    kwargs[k] = bool(v)
            else:
                kwargs[k] = v
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class CalculationIntent:
    """First-class input object for the goldilocks-core pipeline.

    Args:
        structure: Input crystal structure.
        code: DFT code.
            Supported: ``"qe"``.
            Planned: ``"vasp"``.
        task: Calculation type. Follows pw.x ``calculation`` parameter:
            ``"scf"``, ``"nscf"``, ``"bands"``, ``"relax"``,
            ``"md"``, ``"vc-relax"``, ``"vc-md"``.
            Other codes may use different task identifiers.
        xc: Exchange-correlation functional.
            Supported: ``"pbesol"``.
            Planned: ``"pbe"``, ``"hse06"``, ``"scan"``, ``"r2scan"``.
        pseudo_family: Pseudopotential family label (aiida-pseudo format).
            Supported: ``"PseudoDojo/0.4/PBEsol/SR/standard/upf"``,
            ``"PseudoDojo/0.4/PBEsol/FR/standard/upf"``,
            ``"SSSP/1.3/PBEsol/efficiency"``,
            ``"SSSP/1.3/PBEsol/accurate"``.
            Planned: ``"SSSP/1.3/PBE/efficiency"``,
            ``"SSSP/1.3/PBE/accurate"``.
        accuracy: Calculation accuracy tier.
        hints: Typed parameter overrides (provenance = ``"user_hint"``).
            Pass :class:`ParameterHints` directly or use
            ``ParameterHints.from_dict({"key": value, ...})``.
    """

    structure: Structure
    code: str = "qe"
    task: Literal["scf", "nscf", "bands", "relax", "md", "vc-relax", "vc-md", "ph"] = "scf"
    xc: str = "pbesol"
    pseudo_family: str = "PseudoDojo/0.4/PBEsol/SR/standard/upf"
    accuracy: Literal["fast", "balanced", "accurate"] = "accurate"
    hints: ParameterHints = field(default_factory=ParameterHints)
