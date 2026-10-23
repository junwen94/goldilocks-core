"""capabilities: the v2 capabilities contract (v2 epic 8, #8;
goldilocks-core-design.md S4.2).

Consumers: the frontend (render forms), the agent (know what's tunable),
the CLI (``goldilocks settings``' data source, ``--set`` key validation),
and cross-repo vocabulary version checks.

**`settings[]` is reflected from the override dataclasses `service/`
already built, not hand-listed** -- this is design point (1)'s "add an
advisor, its settings automatically appear in capabilities" promise,
made real via `pydantic`'s own `model_fields` plus `typing.get_type_hints`
on the plain dataclasses `service/_system_overrides.py`/`_step_kpoints.py`/
`_step_resources.py` already compose. Concretely: for each of those three
classes, every non-``_llm`` field is either (a) a plain type (like
``pseudo_table_id: str | None``), which becomes one setting directly, or
(b) a ``HumanInput`` subclass (like ``cutoffs: CutoffsHumanInput | None``),
whose *own* pydantic fields each become one setting, grouped under the
outer field's name. This walk only needs to import the three composing
classes themselves, not any of the ~15 individual advisor modules behind
them -- which is also why this module comfortably fits this project's
import-surface ceiling (``scripts/check_complexity.py``) despite
covering every advisor: reflection at runtime doesn't cost an `import`
statement the way a hand-written list of every advisor's types would.

**`group`/`scope`/`programs` are derived structurally, not annotated
per-field**: `group` is the outer composing field's own name (e.g. every
field inside ``CutoffsHumanInput`` gets ``group="cutoffs"`` for free);
`scope` is which of the three composing classes a field came from
(``SystemOverrides`` -> ``"system"``, ``KpointsOverrides``/
``ResourceOverrides`` -> ``"per_step"``); `programs` is ``None`` (all)
for system-level settings and ``["pw.x"]`` for per-step ones, since
today there is exactly one program in the whole codebase (v2 epics 1-7
render only QE's ``scf`` step) -- revisit once a second program/task
exists, per this codebase's own repeated "don't build a lookup table
for one entry" rule (e.g. ``inputs/task.py``'s own docstring).

**What genuinely can't be derived** -- physical unit, a confidently-known
fixed default (many fields have no fixed default at all: cutoffs and
electron count are pseudopotential-dependent, `nbnd` is formula-derived,
walltime defaults to the partition ceiling), a declared `ml_target`, and
the one `enum_from` -- lives in `_SETTING_META` below, one small opt-in
table, the same pattern this codebase already uses for
``scripts/check_complexity.py``'s ``LIMITS`` dict. This is not "hand
-written instead of auto-generated": the *existence*, *key*, *type*, and
*grouping* of every setting is still 100% reflected; this table only
supplies the handful of facts pydantic's type system cannot know
(what unit a ``float`` is in, whether an ml model is planned for it).

**One real key collision, resolved by renaming, not by luck**:
``advisors/hubbard_u.py``'s own ``HubbardUHumanInput.needs_correlation``
and ``analysis/needs_correlation.py``'s fact-level override both use the
field name ``needs_correlation`` -- but they are genuinely different
knobs (the fact is a general "does this structure need correlation
treatment" judgement; hubbard's own field is a stronger, hubbard
-specific escape hatch that overrides even a human-set fact, see
``hubbard_u()``'s own priority order). Exposed as flat ``--set`` keys
they would collide, so hubbard's is renamed to
``hubbard_needs_correlation`` in ``_SETTING_META`` -- the Python
attribute name is untouched, only the exposed capabilities key differs.

**`facts[]` lists the four scalar analysis facts** (``is_metal``,
``is_magnetic``, ``needs_soc``, ``needs_correlation``) in the design
doc's illustrative shape (``{"key", "type": "enum", "values",
"ml_target", "overridable", "description"}``). Only which facts exist,
their ``ml_target`` and their description are hand-written (``_FACTS``).
``type``/``values`` come from the fact's record type on
``service.AnalysisFacts`` (``FieldState[Literal[...]]`` is an enum of the
Literal's members, ``FieldState[bool]`` a boolean), and ``overridable``
is whether the key has an analysis binding. The overrides are typed with
the same record Literals, so every advertised value is an accepted one
(#139: ``is_metal`` advertised ``metal``/``non_metal`` while its override
took only booleans). ``composition``/``geometry``/``symmetry`` are
structural, multi-field records with no override at all
(``analysis/composition.py``'s and ``analysis/symmetry.py``'s own
docstrings say so explicitly) and don't fit this shape; they are not
exposed here.

**`warnings[]` is aggregated, not hand-listed**, the same reflection
-over-imports-elsewhere-in-the-codebase spirit as `settings[]`: every
advisor that can emit a ``resolution.Warning`` declares its own
``WARNING_CATALOGUE`` constant (code/level/category/a generic
description), and ``advisors/warning_catalogue.py`` -- a dedicated
aggregator, not this module -- imports every one of them so this module
itself only needs one import to get the full list without blowing its
own import-surface ceiling.

**`models[]`/`approaches` (v2 epic 11, #11).** ``models[]`` lists every
model ``ml/registry.toml`` registers (``ml.models.registered_models``),
whether or not it is actually installed; ``approaches`` (design point
(1)-b) is the honest "is ml actually usable right now" signal -- a
fact/setting's ``ml_target`` only gains ``"ml"`` once
``ml.predict.model_ready`` passes the checks a prediction makes first: the
model's asset installed and verified and, for ``is_magnetic``, the mMACE
backbone configured and the mace packages installed (#130).
``is_metal``/``is_magnetic`` (``ml.models.ML_CLASSIFIER_ROLES``) and
``k_distance`` (QRF95) are wired this way; ``k_index``'s own
``ml_target`` names a real, published goldilocks-ml *ladder-rung* model
that is a distinct thing from QRF95 and is not registered/wired yet
(#90) -- it stays ``["human", "heuristic"]`` honestly rather than
claiming support that would raise on every real call. That model predicts
rungs on goldilocks-data's ladder (``ladder_1based.v2``), which core's
ladder no longer matches rung for rung (#163, ``kmesh``).
"""

from __future__ import annotations

import types
import typing
from dataclasses import dataclass
from importlib.metadata import version as package_version
from typing import Literal, TypedDict

from goldilocks_core.advisors.warning_catalogue import (
    WARNING_CATALOGUE as _ADVISOR_WARNING_CATALOGUE,
)
from goldilocks_core.assets.pseudopotentials.registry import load_tables
from goldilocks_core.inputs.hpc import list_hpc_profiles, load_hpc_profile
from goldilocks_core.ml.predict import model_ready
from goldilocks_core.service import (
    AnalysisFacts,
    AnalysisOverrides,
    KpointsOverrides,
    RelaxOverrides,
    ResourceOverrides,
    SystemOverrides,
)

VOCABULARY_VERSION = "2"
"""Bump when the settings/facts vocabulary changes in a way other
ecosystem repos (ml/agent/data) need to know about -- goldilocks-core
-design.md S4.2's cross-repo validation antidote."""

SOURCES: tuple[Literal["human", "ml", "llm", "heuristic"], ...] = (
    "human",
    "ml",
    "llm",
    "heuristic",
)
"""Order is priority, matching ``resolution.Source`` -- closed, four
tiers, per design point (5)."""

_CODE = "quantum_espresso"
_TASK = "scf_single_point"
_DOS_TASK = "dos"
_RELAX_TASK = "relax"
_VC_RELAX_TASK = "vc-relax"
_PROGRAM = "pw.x"


def _approaches(ml_target: str | None) -> list[str]:
    """Design point (1)-b: ``ml_target`` is a static declaration;
    ``approaches`` is what's *actually* usable right now, computed as
    ``["human"] + (["ml"] if that target's model would run) +
    ["heuristic"]``. v2 epic 11 (#11): ``is_metal``/``is_magnetic`` are
    registered, real, ``goldilocks_ml.inference``-loadable classifiers
    (``ml.models.ML_CLASSIFIER_ROLES``), and ``k_distance`` is QRF95
    (#92) -- ``"ml"`` appears for any of these once
    ``ml.predict.model_ready`` finds it set up, not merely declared.
    ``k_index``'s ``ml_target`` names a different, not-yet-registered/wired
    ladder-rung model (#90), which ``model_ready`` never reports ready, so
    it keeps resolving to ``["human", "heuristic"]`` honestly rather than
    claiming ml support that would raise ``MlModelUnavailable`` on every
    real call."""
    if ml_target is not None and model_ready(ml_target):
        return ["human", "ml", "heuristic"]
    return ["human", "heuristic"]


class Setting(TypedDict, total=False):
    key: str
    group: str
    type: str
    enum: list[str]
    unit: str | None
    default: object
    enum_from: str
    codes: list[str] | None
    tasks: list[str] | None
    programs: list[str] | None
    scope: Literal["system", "per_step"]
    ml_target: str | None
    approaches: list[str]
    description: str


class Fact(TypedDict):
    key: str
    type: str
    values: list[str] | None
    ml_target: str | None
    approaches: list[str]
    overridable: bool
    description: str


class Capabilities(TypedDict):
    core_version: str
    vocabulary_version: str
    codes: list[dict[str, object]]
    tasks: list[dict[str, object]]
    facts: list[Fact]
    settings: list[Setting]
    pseudopotential_tables: list[dict[str, object]]
    hpc_profiles: list[dict[str, object]]
    models: list[dict[str, object]]
    warnings: list[dict[str, object]]
    sources: list[str]


class _SettingExtra(TypedDict, total=False):
    key: str
    unit: str | None
    default: object
    enum_from: str
    ml_target: str | None
    description: str


_SETTING_META: dict[str, _SettingExtra] = {
    "functional": {
        "default": "PBEsol",
        "enum_from": "pseudopotential_tables.functional",
        "description": "Exchange-correlation functional label (e.g. PBE, PBEsol).",
    },
    "pseudo_table_id": {
        "description": (
            "Pin an explicit pseudopotential table id instead of automatic selection."
        ),
    },
    "ecutwfc_ry": {
        "unit": "Ry",
        "description": "Plane-wave wavefunction kinetic-energy cutoff.",
    },
    "ecutrho_ry": {
        "unit": "Ry",
        "description": "Charge-density cutoff; auto-derived from ecutwfc_ry if unset.",
    },
    "nelec": {
        "description": (
            "Total valence electron count of the uploaded cell (otherwise "
            "summed from pseudopotential z_valence), written as tot_charge, "
            "its difference from the neutral count; refused when "
            "magnetic_ordering writes a cell of another size."
        ),
    },
    "spin_polarized": {
        "description": (
            "Force spin polarization on or off: a collinear nspin=2 run, or a "
            "magnetic noncollinear one with spin_orbit_coupling."
        ),
    },
    "spin_orbit_coupling": {
        "description": (
            "Enable noncollinear spin-orbit-coupling mode (always opt-in, "
            "never auto-enabled)."
        ),
    },
    "tot_magnetization": {
        "description": (
            "Total per-cell magnetization target of a collinear spin-polarized "
            "run (nspin=2); refused with spin_orbit_coupling, whose noncollinear "
            "run takes starting_magnetization instead, and ignored when "
            "spin_polarized is false. Fixed occupations need an integer of the "
            "electron count's parity; a non-metal gets them by default only "
            "then, and smearing otherwise."
        ),
    },
    "starting_magnetization": {
        "description": (
            "Starting spin-polarization fraction (each value in [-1, 1]) per "
            "site label -- the uploaded structure's labels, or Ni1/Ni2/O-style "
            "labels of an applied magnetic_ordering -- replacing the "
            "valence-electron-based heuristic; labels left out start at 0, and "
            "sites with different values become separate species."
        ),
    },
    "magnetic_ordering": {
        "description": (
            "Opt in to an antiferromagnetic ordering instead of the "
            "ferromagnetic default ('fm' or unset), over the sites that carry "
            "a local d or f moment: 'afm' applies the first candidate the "
            "magnetic-orderings listing gives (the smallest cell, then the "
            "most symmetric), 'afm-<n>' the listed candidate of that label; "
            "records.magnetic names the criterion and the other candidates of "
            "that size. A slab, wire or boxed molecule gets only candidates "
            "that keep its vacuum vectors. Any value, 'fm' included, turns spin "
            "polarization on unless spin_polarized is false, which ignores it "
            "with a warning. 'afm' degrades to ferromagnetic with a warning if "
            "there is no candidate; an 'afm-<n>' that is not listed is refused."
        ),
    },
    "moment_direction": {
        "description": (
            "Direction the moments of a spin-polarized spin_orbit_coupling run "
            "start along, as a lattice direction [u, v, w] of the uploaded "
            "cell (non-zero), written as QE's angle1/angle2 and used to count "
            "the k-points by magnetic symmetry. A physical choice: unset, the "
            "moments start along [0, 0, 1], with a notice. Ignored with a "
            "warning by any other run."
        ),
    },
    "use_vdw": {
        "description": (
            "Force whether a van der Waals dispersion correction is applied. "
            "D3(BJ) needs parameters for the functional, which Quantum "
            "ESPRESSO has only for PBE and PBEsol: with any other functional, "
            "such as LDA, the default is no correction and true is refused."
        ),
    },
    "method": {
        "description": (
            "Dispersion-correction method when use_vdw is set (only d3bj is "
            "implemented)."
        ),
    },
    "isolation": {
        "description": (
            "How the electrostatics treat periodic images. 'periodic': no "
            "truncation, the default for any cell periodic along at least one "
            "lattice vector, including molecular crystals, slabs and wires "
            "(pw.x has no 1D truncation). 'slab': Quantum ESPRESSO's 2D Coulomb "
            "truncation (assume_isolated='2D', Sohier et al. 2017), never the "
            "default; refused unless the vacuum lies along c only, a and b lie "
            "in the xy plane, c lies along z, the cell is at least 11.11 Å "
            "(21 bohr) high and the vacuum is at least 10.58 Å (20 bohr) wider "
            "than the slab, and a vc-relax needs cell_dofree='2Dxy' or '2Dshape'. "
            "'isolated': Martyna-Tuckerman (assume_isolated='martyna-tuckerman'), "
            "the default with vacuum along all three lattice vectors and refused "
            "otherwise; pw.x computes no stress under it, so vc-relax is refused."
        ),
    },
    "needs_correlation": {
        "key": "hubbard_needs_correlation",
        "description": (
            "Force whether a Hubbard +U correction is applied, overriding even "
            "a human-set needs_correlation fact: false runs without +U, true "
            "applies the built-in U table to the structure's d and f elements."
        ),
    },
    "u_by_element": {
        "unit": "eV",
        "description": (
            "Dudarev U_eff in eV (> 0) per element symbol; replaces the built-in "
            "table, so only listed elements present in the structure get +U. "
            "La and Ac are rejected: no installed pseudopotential has an "
            "occupied La 4f or any Ac 5f channel, so pw.x stops on any U for them."
        ),
    },
    "occupations": {
        "description": (
            "Electronic-occupation scheme (fixed, smearing, or tetrahedra_opt). "
            "Default: smearing for a metal or unconfirmed metallicity; fixed for "
            "a non-metal where fixed occupations can run -- an even electron "
            "count without spin polarization (spin_orbit_coupling or not), any "
            "count on a spin-polarized spin_orbit_coupling run, and, for a "
            "collinear spin-polarized run, an integer tot_magnetization of the "
            "count's parity (one set, or 0 for a magnetic_ordering compensated "
            "by symmetry) -- and smearing otherwise. An explicit fixed on a "
            "collinear run with no such moment is refused."
        ),
    },
    "smearing_type": {
        "description": (
            "Smearing function used when occupations=smearing; defaults to 'cold'."
        ),
    },
    "degauss": {
        "unit": "Ry",
        "description": (
            "Smearing width used when occupations=smearing; heuristic default "
            "0.01 Ry for metals."
        ),
    },
    "k_grid": {
        "description": (
            "Explicit Monkhorst-Pack mesh dimensions (nk1, nk2, nk3); wins "
            "outright over k_index/k_distance if more than one is set."
        ),
    },
    "k_index": {
        "ml_target": "k_index",
        "description": (
            "A specific 1-based rung on this structure's own k-mesh ladder, "
            "the distinct meshes its point group maps onto itself (rung 1 is "
            "Gamma-only); wins over k_distance if both are set."
        ),
    },
    "k_distance": {
        "unit": "1/Angstrom",
        "ml_target": "k_distance",
        "description": (
            "Target k-point spacing along each periodic axis; axes the point "
            "group mixes share the larger count, and a vacuum axis always gets "
            "one k-point. Heuristic default is 0.15 for metals or when "
            "metallicity is unknown, 0.30 for a confirmed non-metal."
        ),
    },
    "shift": {
        "description": (
            "K-mesh origin offset per axis (0 or 1); defaults to Gamma-centered."
        ),
    },
    "nosym": {
        "default": False,
        "description": "Disable symmetry reduction of the k-point mesh entirely.",
    },
    "nbnd": {
        "description": (
            "Total number of Kohn-Sham bands (per spin channel with nspin=2), "
            "overriding the default written otherwise: pw.x's setup.f90 rule, "
            "with a collinear smearing run's majority channel raised for its "
            "seeded moment. Refused when below the electrons a "
            "spin channel holds (pw.x's 'too few bands'); a count that holds "
            "the moment below its seed gets a warning."
        ),
    },
    "extra_bands": {
        "description": (
            "Extra bands added on top of the default band count (pw.x's "
            "setup.f90 rule, with a collinear smearing run's majority channel "
            "raised for its seeded moment); a total below the electrons a spin "
            "channel holds is refused."
        ),
    },
    "conv_thr": {
        "unit": "Ry",
        "description": (
            "SCF self-consistency energy-convergence threshold; otherwise "
            "scaled per atom."
        ),
    },
    "etot_conv_thr": {
        "unit": "Ry",
        "description": (
            "Total-energy convergence threshold between steps; otherwise "
            "scaled per atom."
        ),
    },
    "mixing_beta": {
        "default": 0.4,
        "description": "Charge-density mixing parameter for SCF convergence.",
    },
    "electron_maxstep": {
        "default": 80,
        "description": "Maximum number of SCF iterations allowed before giving up.",
    },
    "mixing_mode": {
        "description": (
            "SCF density-mixing algorithm; heuristic default is local-TF for a "
            "cell with vacuum along any lattice vector (a slab, wire or isolated "
            "system) and plain otherwise, including bulk layered crystals."
        ),
    },
    "mixing_fixed_ns": {
        "description": (
            "Number of initial SCF iterations during which pw.x holds the "
            "Hubbard occupation matrix at its starting value (+U runs only). "
            "Unset by default (QE's 0). pw.x can report convergence inside "
            "this window, so energies, forces and stresses from a nonzero "
            "value are not self-consistent DFT+U results; use it only for a "
            "preparatory run whose output seeds a separate run with ns free."
        ),
    },
    "partition": {
        "description": "HPC queue/partition to submit to.",
    },
    "nodes": {
        "description": (
            "Number of compute nodes requested; otherwise the fewest on which "
            "pw.x's estimated memory per process, for the layout each step "
            "runs, times the tasks per node and a 1.2 safety factor fits a "
            "node."
        ),
    },
    "ntasks": {
        "description": (
            "Total number of MPI tasks requested; otherwise nodes x cores_per_node."
        ),
    },
    "walltime_h": {
        "unit": "hours",
        "description": (
            "Requested job walltime; heuristic default is the partition's own "
            "maximum. 95% of it is split across the task's pw.x steps as "
            "max_seconds, by estimated workload."
        ),
    },
    "account": {
        "description": (
            "HPC accounting/billing code; personal to the submitter, never guessed."
        ),
    },
    "npool": {
        "description": (
            "Number of k-point pools passed to pw.x as -npool; must divide "
            "ntasks. Default: the most pools, up to one per k-point/spin "
            "pair (two per k-point with collinear spin), that keep the pools "
            "at least 80% busy."
        ),
    },
    "ndiag": {
        "description": (
            "Linear-algebra group size passed to pw.x as -ndiag; unset by "
            "default so pw.x chooses it from nbnd. Refused when its s x s "
            "grid, s = isqrt(min(ndiag, ntasks/npool)), cannot distribute "
            "nbnd bands (needs nbnd > (s-1)^2)."
        ),
    },
    "ion_dynamics": {
        "default": "bfgs",
        "description": (
            "Ionic relaxation algorithm for relax/vc-relax; vc-relax only "
            "accepts 'bfgs' in this codebase (QE couples cell_dynamics to it)."
        ),
    },
    "forc_conv_thr": {
        "unit": "Ry/Bohr",
        "default": 1.0e-3,
        "description": "Force-convergence threshold for relax/vc-relax.",
    },
    "nstep": {
        "default": 50,
        "description": "Maximum number of ionic/cell relaxation steps.",
    },
    "trust_radius_max": {
        "unit": "Bohr",
        "default": 0.8,
        "description": "Maximum BFGS ionic-displacement trust radius (bfgs only).",
    },
    "trust_radius_min": {
        "unit": "Bohr",
        "default": 1.0e-3,
        "description": (
            "Minimum BFGS ionic-displacement trust radius; BFGS resets below "
            "this (bfgs only)."
        ),
    },
    "trust_radius_ini": {
        "unit": "Bohr",
        "default": 0.5,
        "description": "Initial BFGS ionic-displacement trust radius (bfgs only).",
    },
    "remove_rigid_rot": {
        "default": False,
        "description": (
            "Cancel spurious rigid-body torque for isolated-system relaxation; "
            "trades total-energy/force self-consistency for speed."
        ),
    },
    "fix_bottom_layers": {
        "description": (
            "Fix the bottom N atomic layers of a slab (if_pos) during "
            "relax/vc-relax, counted up from the vacuum; refused unless the cell "
            "to be written has one vacuum gap, along exactly one lattice vector, "
            "and at least N layers (atoms within 0.5 Å along the normal share "
            "one)."
        ),
    },
    "cell_dofree": {
        "default": "all",
        "description": (
            "Which cell degrees of freedom vc-relax may move. Default 'all'; "
            "'2Dxy' for a slab with its vacuum along c, which is written with a "
            "along x and a, b in the xy plane. A wire or a slab with its vacuum "
            "along a or b has no default. '2Dxy' and '2Dshape' are refused unless "
            "a and b lie in the xy plane and any vacuum lies along c; the masks "
            "of Cartesian components ('a', 'b', 'c', 'x', 'y', 'z', 'xy', 'xz', "
            "'yz', 'xyz') are refused when the cell is written rotated."
        ),
    },
    "press": {
        "unit": "kbar",
        "default": 0.0,
        "description": "Target external pressure for vc-relax.",
    },
    "press_conv_thr": {
        "unit": "kbar",
        "default": 0.5,
        "description": "Pressure-convergence threshold for vc-relax.",
    },
    "cell_factor": {
        "default": 2.0,
        "description": (
            "Pseudopotential-table interpolation headroom; must exceed the "
            "maximum linear cell contraction expected during vc-relax."
        ),
    },
}


@dataclass(frozen=True, slots=True)
class _FactMeta:
    """What a fact's record type cannot say: its ML model target and its
    description. The description says what the fact means, never how it
    was resolved -- that is the record's ``source`` and provenance, and it
    differs per tier. ``_facts()`` derives the rest."""

    key: str
    ml_target: str | None
    description: str


_FACTS: tuple[_FactMeta, ...] = (
    _FactMeta(
        key="is_metal",
        ml_target="is_metal",
        description="Whether the structure is metallic (zero DFT band gap).",
    ),
    _FactMeta(
        key="is_magnetic",
        ml_target="is_magnetic",
        description="Whether the structure is expected to be magnetic.",
    ),
    _FactMeta(
        key="needs_soc",
        ml_target=None,
        description=(
            "Whether spin-orbit coupling is likely relevant for this structure."
        ),
    ),
    _FactMeta(
        key="needs_correlation",
        ml_target=None,
        description="Whether a Hubbard +U (or hybrid) correction is likely needed.",
    ),
)


def _unwrap_optional(annotation: object) -> object:
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _json_type(annotation: object) -> dict[str, object]:
    annotation = _unwrap_optional(annotation)
    origin = typing.get_origin(annotation)
    if origin is Literal:
        return {"type": "string", "enum": sorted(typing.get_args(annotation))}
    if origin is tuple:
        args = typing.get_args(annotation)
        return {
            "type": "array",
            "items": _json_type(args[0])["type"] if args else "integer",
            "minItems": len(args) or None,
            "maxItems": len(args) or None,
        }
    if origin is dict:
        return {"type": "object"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    return {"type": "string"}


@dataclass(frozen=True, slots=True)
class SettingBinding:
    """One ``--set``-able key's construction recipe, alongside its
    display metadata -- the single walk ``_leaves()`` below performs
    once, shared by ``capabilities()`` (projects to JSON) and the
    shared request-validation layer, so the two can never drift apart
    by walking the override tree two different ways.

    ``branch`` says which of ``RunOverrides``' three sub-trees this key
    belongs to; ``outer_field`` is that sub-tree's own field name
    (``SystemOverrides.cutoffs``, say); ``inner_field`` is the field
    *inside* the ``HumanInput`` class for a wrapped setting, or ``None``
    for a plain-typed one (``pseudo_table_id``) where ``outer_field``
    *is* the setting. ``json_type`` is ``_json_type(annotation)``,
    computed once here rather than re-derived by every consumer --
    the shared request-validation layer's CLI string-coercion reads
    ``json_type["type"]`` directly instead of importing this module's
    own private type-mapping helper.
    """

    key: str
    group: str
    branch: Literal["analysis", "system", "kpoints", "resources", "relax"]
    outer_field: str
    inner_field: str | None
    human_input_cls: type | None
    annotation: object
    json_type: dict[str, object]
    scope: Literal["system", "per_step"]
    programs: list[str] | None


def _leaves_from_human_input(
    branch: Literal["analysis", "system", "kpoints", "resources", "relax"],
    outer_field: str,
    human_input_cls: type,
    *,
    scope: Literal["system", "per_step"],
    programs: list[str] | None,
) -> list[SettingBinding]:
    # `include_extras=False` (the default) strips any per-element
    # `Annotated[int, Field(gt=0)]` metadata a field's own tuple members
    # carry (e.g. KSamplingHumanInput.k_grid) back down to the plain
    # `tuple[int, int, int]` shape `_json_type` below expects -- reading
    # `model_fields[name].annotation` directly would leave `Annotated`
    # wrappers in place instead, since pydantic keeps per-element
    # constraint metadata on the type itself, not lifted to `FieldInfo`.
    hints = typing.get_type_hints(human_input_cls)
    leaves = []
    for name in human_input_cls.model_fields:
        # Analysis leaves always keep their bare field name: they must
        # match `_FACTS`'s own keys exactly (`_settings()` filters them
        # out and `_facts()` derives `overridable` by that same key), and
        # `_SETTING_META`'s one rename entry ("needs_correlation" ->
        # "hubbard_needs_correlation") exists for hubbard_u's *own* field
        # of the same inner name, not this one -- see this module's own
        # docstring on that collision.
        extra = _SETTING_META.get(name, {}) if branch != "analysis" else {}
        leaves.append(
            SettingBinding(
                key=extra.get("key", name),
                group=outer_field,
                branch=branch,
                outer_field=outer_field,
                inner_field=name,
                human_input_cls=human_input_cls,
                annotation=hints[name],
                json_type=_json_type(hints[name]),
                scope=scope,
                programs=programs,
            )
        )
    return leaves


def _leaves_from_overrides(
    overrides_cls: type,
    *,
    branch: Literal["analysis", "system", "kpoints", "resources", "relax"],
    scope: Literal["system", "per_step"],
) -> list[SettingBinding]:
    hints = typing.get_type_hints(overrides_cls)
    programs = None if scope == "system" else [_PROGRAM]
    leaves: list[SettingBinding] = []
    for name, annotation in hints.items():
        if name.endswith("_llm"):
            continue
        inner = _unwrap_optional(annotation)
        if isinstance(inner, type) and hasattr(inner, "model_fields"):
            leaves.extend(
                _leaves_from_human_input(
                    branch, name, inner, scope=scope, programs=programs
                )
            )
        else:
            extra = _SETTING_META.get(name, {})
            leaves.append(
                SettingBinding(
                    key=extra.get("key", name),
                    group=name,
                    branch=branch,
                    outer_field=name,
                    inner_field=None,
                    human_input_cls=None,
                    annotation=inner,
                    json_type=_json_type(inner),
                    scope=scope,
                    programs=programs,
                )
            )
    return leaves


def _leaves() -> list[SettingBinding]:
    return [
        *_leaves_from_overrides(AnalysisOverrides, branch="analysis", scope="system"),
        *_leaves_from_overrides(SystemOverrides, branch="system", scope="system"),
        *_leaves_from_overrides(KpointsOverrides, branch="kpoints", scope="per_step"),
        *_leaves_from_overrides(
            ResourceOverrides, branch="resources", scope="per_step"
        ),
        *_leaves_from_overrides(RelaxOverrides, branch="relax", scope="per_step"),
    ]


def bindings() -> dict[str, SettingBinding]:
    """Every ``--set``-able key's construction recipe, keyed by its
    exposed capabilities key -- used by the shared request-validation
    layer to turn ``{key: value}`` into a real ``RunOverrides``, and by
    nothing inside this module (``_settings()`` below projects the same
    ``_leaves()`` call to JSON instead)."""
    return {leaf.key: leaf for leaf in _leaves()}


def _setting_from_leaf(leaf: SettingBinding) -> Setting:
    extra = _SETTING_META.get(leaf.inner_field or leaf.outer_field, {})
    ml_target = extra.get("ml_target")
    setting: Setting = {
        "key": leaf.key,
        "group": leaf.group,
        **leaf.json_type,
        "unit": extra.get("unit"),
        "codes": None,
        "tasks": None,
        "programs": leaf.programs,
        "scope": leaf.scope,
        "ml_target": ml_target,
        "approaches": _approaches(ml_target),
        "description": extra.get("description", ""),
    }
    if "default" in extra:
        setting["default"] = extra["default"]
    if "enum_from" in extra:
        setting["enum_from"] = extra["enum_from"]
    return setting


def _settings() -> list[Setting]:
    """Every ``--set``-able leaf *except* the four analysis facts
    (``is_metal``/``is_magnetic``/``needs_soc``/``needs_correlation``):
    ``facts()`` already describes them, with the type and values of the
    record their override is typed by. They still walk through
    ``_leaves()`` so ``bindings()`` (the `--set`/override construction
    path) accepts them; only this JSON projection skips them."""
    fact_keys = {fact.key for fact in _FACTS}
    return [_setting_from_leaf(leaf) for leaf in _leaves() if leaf.key not in fact_keys]


def _fact_schema(record: object) -> tuple[str, list[str] | None]:
    """A fact's published ``type`` and ``values`` from its ``AnalysisFacts``
    record type ``FieldState[T]``: ``("enum", members)`` for a ``Literal``
    ``T``, ``("boolean", None)`` for ``bool``. Raises ``TypeError`` for any
    other ``T``, which has no scalar fact shape."""
    (value_type,) = typing.get_args(record)
    json_type = _json_type(value_type)
    if "enum" in json_type:
        return "enum", list(json_type["enum"])
    if json_type["type"] == "boolean":
        return "boolean", None
    raise TypeError(f"fact record type {value_type!r} is neither a Literal nor bool")


def _facts() -> list[Fact]:
    """``_FACTS`` projected to the published shape. ``approaches`` is
    computed per call, not at import time, since ``_approaches`` checks
    live installed-asset state (v2 epic 11, #11). A fact is overridable
    exactly when its key has an analysis binding (a ``HumanInput`` on
    ``AnalysisOverrides``)."""
    records = typing.get_type_hints(AnalysisFacts)
    overridable = {leaf.key for leaf in _leaves() if leaf.branch == "analysis"}
    facts: list[Fact] = []
    for meta in _FACTS:
        fact_type, values = _fact_schema(records[meta.key])
        facts.append(
            Fact(
                key=meta.key,
                type=fact_type,
                values=values,
                ml_target=meta.ml_target,
                approaches=_approaches(meta.ml_target),
                overridable=meta.key in overridable,
                description=meta.description,
            )
        )
    return facts


def _pseudopotential_tables() -> list[dict[str, object]]:
    return [
        {
            "id": table.id,
            "provider": table.provider,
            "version": table.version,
            "functional": table.functional,
            "relativistic": table.relativistic,
            "accuracy": table.accuracy,
            "elements": list(table.elements),
            "licence": table.licence,
            "citation": table.citation,
            "default": table.default,
        }
        for table in load_tables().values()
    ]


def _hpc_profiles() -> list[dict[str, object]]:
    profiles = []
    for name in list_hpc_profiles():
        profile = load_hpc_profile(name)
        profiles.append(
            {
                "id": name,
                "name": profile.name,
                "scheduler": profile.scheduler,
                "partitions": sorted(profile.partitions),
            }
        )
    return profiles


def _codes() -> list[dict[str, object]]:
    return [
        {
            "id": _CODE,
            "name": "Quantum ESPRESSO",
            "tasks": [_TASK, _DOS_TASK, _RELAX_TASK, _VC_RELAX_TASK],
        }
    ]


def _tasks() -> list[dict[str, object]]:
    return [
        {
            "id": _TASK,
            "name": "Single-point SCF",
            "description": "One self-consistent-field calculation, no relaxation.",
            "codes": [_CODE],
            "step_count": 1,
            "executables": [_PROGRAM],
        },
        {
            "id": _DOS_TASK,
            "name": "Density of states",
            "description": (
                "scf, then a denser nscf pass, then dos.x -- three steps "
                "sharing one prefix/outdir (v2 epic 9, #9)."
            ),
            "codes": [_CODE],
            "step_count": 3,
            "executables": [_PROGRAM, _PROGRAM, "dos.x"],
        },
        {
            "id": _RELAX_TASK,
            "name": "Ionic relaxation",
            "description": (
                "One pw.x run, calculation='relax': scf plus BFGS/damped/FIRE "
                "ionic-position optimization (v2 epic 10, #10)."
            ),
            "codes": [_CODE],
            "step_count": 1,
            "executables": [_PROGRAM],
        },
        {
            "id": _VC_RELAX_TASK,
            "name": "Variable-cell relaxation",
            "description": (
                "One pw.x run, calculation='vc-relax': scf plus BFGS ionic "
                "and cell relaxation together (v2 epic 10, #10)."
            ),
            "codes": [_CODE],
            "step_count": 1,
            "executables": [_PROGRAM],
        },
    ]


def _warnings() -> list[dict[str, object]]:
    return [warning.model_dump() for warning in _ADVISOR_WARNING_CATALOGUE]


def _models() -> list[dict[str, object]]:
    """Every ML model registry.toml declares (v2 epic 11, #11) --
    registered, not necessarily installed; a fact's own ``approaches``
    (``_approaches`` above) is the honest "is ml actually usable right
    now" signal, checked per asset. This listing is what registered
    against which target, for a caller that wants to know which models
    exist at all."""
    from goldilocks_core.ml.models import registered_models

    return [
        {
            "id": model.id,
            "role": model.role,
            "name": model.spec.name,
            "version": model.spec.version,
            "target": model.spec.target,
        }
        for model in registered_models()
    ]


def capabilities() -> Capabilities:
    """Everything the design doc's S4.2 shape lists. Cacheable in-process
    per that section's own note ("static... only `models` needs to query
    ml") -- callers needing a fresh read after installing a new
    pseudopotential table or HPC profile file should call this again
    rather than expect a long-lived cache here; this function itself
    does no caching."""
    return {
        "core_version": package_version("goldilocks-core"),
        "vocabulary_version": VOCABULARY_VERSION,
        "codes": _codes(),
        "tasks": _tasks(),
        "facts": _facts(),
        "settings": _settings(),
        "pseudopotential_tables": _pseudopotential_tables(),
        "hpc_profiles": _hpc_profiles(),
        "models": _models(),
        "warnings": _warnings(),
        "sources": list(SOURCES),
    }
