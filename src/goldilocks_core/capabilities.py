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

**`facts[]` is hand-written, not reflected**: only the four analysis
facts that are both scalar-valued *and* overridable (``is_metal``,
``is_magnetic``, ``needs_soc``, ``needs_correlation``) are listed, per
the design doc's own illustrative shape (``{"key", "type": "enum",
"values", "ml_target", "overridable", "description"}``) -- ``composition``/
``geometry``/``symmetry`` are structural, multi-field records with no
override at all (``analysis/composition.py``'s and
``analysis/symmetry.py``'s own docstrings say so explicitly) and don't
fit this shape; they are not exposed here.

**`warnings[]` is aggregated, not hand-listed**, the same reflection
-over-imports-elsewhere-in-the-codebase spirit as `settings[]`: every
advisor that can emit a ``resolution.Warning`` declares its own
``WARNING_CATALOGUE`` constant (code/level/category/a generic
description), and ``advisors/warning_catalogue.py`` -- a dedicated
aggregator, not this module -- imports all ten of them so this module
itself only needs one import to get the full list without blowing its
own import-surface ceiling.

**`models[]` ships empty.** ml integration is deliberately last (v2 epic
11, #11); until then no ``target`` has an installed model, so
``approaches`` (design point (1)-b) never includes ``"ml"`` -- this is
the expected, honest state, not a bug.
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
from goldilocks_core.analysis.is_magnetic import Magnetism
from goldilocks_core.analysis.is_metal import Metallicity
from goldilocks_core.assets.pseudopotentials.registry import load_tables
from goldilocks_core.inputs.hpc import list_hpc_profiles, load_hpc_profile
from goldilocks_core.service import (
    AnalysisOverrides,
    KpointsOverrides,
    RelaxOverrides,
    ResourceOverrides,
    SystemOverrides,
)

VOCABULARY_VERSION = "1"
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
    ``["human"] + (["ml"] if that target has an installed model) +
    ["heuristic"]``. No target has an installed model yet -- ml
    integration is deliberately last (v2 epic 11, #11) -- so this
    always resolves to ``["human", "heuristic"]`` today regardless of
    ``ml_target``; the parameter is threaded through now so epic 11
    only has to change this one function's body, not any caller."""
    del ml_target  # unused until epic 11 wires a real installed-model check
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
            "Total valence electron count; otherwise summed from "
            "pseudopotential z_valence."
        ),
    },
    "spin_polarized": {
        "description": (
            "Force spin-polarized (nspin=2) vs non-spin-polarized calculation."
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
            "Total per-cell magnetization target; not valid together with "
            "spin_orbit_coupling."
        ),
    },
    "starting_magnetization": {
        "description": (
            "Per-element starting spin-polarization fraction (each value in "
            "[-1, 1]) fed directly to the generated input, bypassing the "
            "valence-electron-based heuristic."
        ),
    },
    "magnetic_ordering": {
        "description": (
            "Opt in to a compensated antiferromagnetic ordering search "
            "('afm') instead of the ferromagnetic default ('fm' or unset); "
            "degrades to ferromagnetic with a warning if no ordering can "
            "be found."
        ),
    },
    "use_vdw": {
        "description": (
            "Force whether a van der Waals dispersion correction is applied."
        ),
    },
    "method": {
        "description": (
            "Dispersion-correction method when use_vdw is set (only d3bj is "
            "implemented)."
        ),
    },
    "assume_isolated": {
        "description": (
            "Force the electrostatic boundary-condition treatment for a "
            "periodic cell (e.g. 'martyna-tuckerman' for an isolated "
            "molecule/wire); 'none' for the normal fully-periodic case."
        ),
    },
    "needs_correlation": {
        "key": "hubbard_needs_correlation",
        "description": (
            "Force whether a Hubbard +U correction is applied, overriding even "
            "a human-set needs_correlation fact -- see the analysis-layer "
            "needs_correlation fact for the general judgement this escalates past."
        ),
    },
    "u_by_element": {
        "unit": "eV",
        "description": (
            "Explicit Dudarev effective-U value per element, overriding the "
            "built-in table."
        ),
    },
    "occupations": {
        "description": (
            "Electronic-occupation scheme (fixed, smearing, or tetrahedra_opt)."
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
            "over k_distance if both set."
        ),
    },
    "k_distance": {
        "unit": "1/Angstrom",
        "ml_target": "k_index",
        "description": (
            "Target k-point spacing; heuristic default is 0.15 for metals, "
            "0.30 otherwise."
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
            "Total number of Kohn-Sham bands, overriding the formula-based "
            "value entirely."
        ),
    },
    "extra_bands": {
        "description": (
            "Extra bands added on top of the formula-computed base band count."
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
            "SCF density-mixing algorithm; heuristic default is local-TF for "
            "2D/molecule geometries."
        ),
    },
    "mixing_fixed_ns": {
        "description": (
            "Initial SCF iterations that freeze Hubbard occupation-matrix "
            "mixing (+U calculations)."
        ),
    },
    "partition": {
        "description": "HPC queue/partition to submit to.",
    },
    "nodes": {
        "description": (
            "Number of compute nodes requested; otherwise sized from the "
            "estimated memory footprint."
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
            "Requested job walltime; heuristic default is the partition's own maximum."
        ),
    },
    "account": {
        "description": (
            "HPC accounting/billing code; personal to the submitter, never guessed."
        ),
    },
    "npool": {
        "description": (
            "Number of k-point parallelization pools; must divide ntasks evenly."
        ),
    },
    "ndiag": {
        "description": (
            "ScaLAPACK linear-algebra process-grid size; only meaningful "
            "when has_scalapack is true."
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
    "cell_dofree": {
        "default": "all",
        "description": (
            "Which cell degrees of freedom vc-relax may move; heuristic "
            "default is hexagon-aware for 2D structures ('ibrav+2Dxy' vs "
            "bare '2Dxy')."
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

_FACTS: tuple[Fact, ...] = (
    Fact(
        key="is_metal",
        type="enum",
        values=list(typing.get_args(Metallicity)),
        ml_target="is_metal",
        approaches=_approaches("is_metal"),
        overridable=True,
        description="Whether the structure is metallic, from composition alone.",
    ),
    Fact(
        key="is_magnetic",
        type="enum",
        values=list(typing.get_args(Magnetism)),
        ml_target="is_magnetic",
        approaches=_approaches("is_magnetic"),
        overridable=True,
        description="Whether the structure is expected to be magnetic.",
    ),
    Fact(
        key="needs_soc",
        type="boolean",
        values=None,
        ml_target=None,
        approaches=_approaches(None),
        overridable=True,
        description=(
            "Whether spin-orbit coupling is likely relevant for this structure."
        ),
    ),
    Fact(
        key="needs_correlation",
        type="boolean",
        values=None,
        ml_target=None,
        approaches=_approaches(None),
        overridable=True,
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
        # out by that same key), and `_SETTING_META`'s one rename entry
        # ("needs_correlation" -> "hubbard_needs_correlation") exists
        # for hubbard_u's *own* field of the same inner name, not this
        # one -- see this module's own docstring on that collision.
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
    those are already fully described in ``facts()`` (their real type is
    the fact's own enum/boolean meaning, not the raw ``bool`` their
    ``HumanInput`` happens to store it as) -- listing them a second time
    here, with a different declared ``type``, would contradict ``facts()``
    instead of complementing it. They still walk through ``_leaves()``
    so ``bindings()`` (the `--set`/override construction path) accepts
    them; only this JSON projection skips them."""
    fact_keys = {fact["key"] for fact in _FACTS}
    return [_setting_from_leaf(leaf) for leaf in _leaves() if leaf.key not in fact_keys]


def _facts() -> list[Fact]:
    return list(_FACTS)


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
        "models": [],
        "warnings": _warnings(),
        "sources": list(SOURCES),
    }
