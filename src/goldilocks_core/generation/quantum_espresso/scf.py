"""write_qe_scf: render one ``pw.x`` scf ``Step`` from resolved settings.

New in v2 (v2 epic 7, #7). Rewrites ``generation/qe/scf.py`` (v1, 270
lines) as pure translation: every value written here traces to an
already-resolved ``SystemSettings``/``PwSettings`` field, and every
``if`` below formats a value or validates input shape -- none of them
choose one, per this epic's own invariant
(goldilocks-core-design.md:1049: "generation never makes a scientific
choice, only translation and formatting").

Card-writing knowledge (``ATOMIC_SPECIES``/``ATOMIC_POSITIONS``/
``CELL_PARAMETERS``/``K_POINTS``) is reused as content from v1's
``atomic_species``/``atomic_positions``/``cell_parameters``/
``k_points`` (``generation/qe/scf.py:225-262``), restructured against
the new types; namelist formatting (``&CONTROL``/``&SYSTEM``/
``&ELECTRONS``) is delegated to ``namelists.render_namelist`` instead of
v1's own hand-written ``_control_section``/``_system_section``/
``_electrons_section``.

Takes a ``JobDecision`` (``advisors/job_resources.py``) alongside
``SystemSettings``/``PwSettings`` for exactly one field: ``max_seconds``
(&CONTROL). This has to be baked into the input file at generation time,
not injected by ``submission/slurm.py`` at runtime -- editing the input
file after generation to add a dynamically-shrinking time budget would
mean the published ``goldilocks.json`` sha256 no longer describes the
bytes actually run (goldilocks-core-design.md:1230-1237, publication
guarantee 3). The rest of ``JobDecision`` (``nodes``/``ntasks``/
``walltime_h``/``partition``) is ``submission/slurm.py``'s concern, not
this writer's.

Structure always comes from ``system.magnetic.relabeled_structure``,
never a separately-passed ``Structure`` -- goldilocks-qe-pw-parameter-audit
finding F18 (also goldilocks-core-design.md:2854): every card in a
multi-step task must be generated from the *same* relabeled structure,
or a later step's ``ATOMIC_SPECIES`` can silently disagree with an
earlier one's. Taking it as a separate parameter would let a caller pass
a different (unlabeled) structure by mistake; reading it off ``system``
makes that impossible by construction.

**v1 bugs this fixes (issue #7, section (c)):**

- Bracket-access into a plain ``dict``, bare ``KeyError`` on a
  missing/renamed key (``generation/qe/scf.py:167,186-188,194-195,
  217-219``) -- every lookup here is a typed dataclass attribute, and
  every value-shape problem raises ``GenerationError`` at this
  boundary, never a raw ``KeyError``.
- Unconditional hardcoded literals with no provenance
  (``generation/qe/scf.py:134-135,153``: ``pseudo_dir``/``outdir``/
  ``ibrav``) -- ``pseudo_dir``/``outdir``/``prefix`` now come from a
  constructed ``SharedContext`` (``steps.py``); ``ibrav = 0`` is the one
  literal kept unconditional, because it is not a scientific choice but
  a fixed consequence of this writer always emitting an explicit
  ``CELL_PARAMETERS`` card (QE requires ``ibrav=0`` whenever the cell is
  given explicitly rather than derived from Bravais-lattice parameters).
- ``if`` branches that pick a value (``_smearing_lines``/``_spin_lines``/
  ``_vdw_lines``, ``generation/qe/scf.py:166-211``) -- every decision
  those made now happens upstream, in ``advisors/occupations.py``/
  ``magnetic_config.py``/``vdw_method.py``. What looks like branching
  below only decides *whether a keyword applies at all* given an
  already-resolved fact (e.g. ``vdw_corr`` is only meaningful when
  ``vdw.use_vdw`` is true) or *which QE spelling* an already-chosen
  value maps to (``_QE_VDW_CORR``) -- never which value to choose.

**Hubbard ``HUBBARD`` card (v2 #88).** ``advisors/hubbard_u.py``'s own
docstring once claimed QE >= 7.1's real card is atom-index-keyed --
verified against QE 7.3's own ``Doc/Hubbard_input.tex`` and
``Modules/read_cards.f90``'s ``card_hubbard`` parsing, that is only true
of the inter-site ``V`` line; the onsite ``U`` line (the only Hubbard
parameter ``HubbardUDecision`` models) is keyed by atomic *type*
(species label) plus its Hubbard manifold, e.g. ``U Mn-3d 5.0`` -- no
atom index at all. ``hubbard_card`` below renders exactly that, for
``plan="table"`` only.

**Still not in this epic's scope, on purpose:**

- ``plan="self_consistent_calibration_needed"``. This needs an ``hp.x``
  self-consistent-linear-response calibration *input* this codebase has
  never written (``advisors/hubbard_u.py``'s own ``CalibrationRequest``
  only carries what such a writer would need, not a writer itself) --
  real, separate, larger scope than one card. ``write_qe_scf`` still
  raises ``GenerationError`` for this plan rather than silently dropping
  a +U decision the advisor layer already made.
- ``ConvergenceDecision.etot_conv_thr``/``RelaxOptions``. QE only
  consults ``etot_conv_thr``/``forc_conv_thr`` for ionic minimization
  (``calculation`` in ``{relax, vc-relax, md, ...}``); a plain ``scf``
  calculation has no ionic loop to compare energies across, so this
  writer never emits either -- ``generation/quantum_espresso/relax.py``
  (v2 epic 10, #10) is the writer that does, reusing this module's own
  ``control_keywords``/``system_keywords``/``electrons_keywords``/
  ``atomic_species``/``cell_parameters``/``atomic_positions``/
  ``k_points``/``validated_pseudo_by_element`` (un-privatized for that
  reuse) rather than duplicating them.
- ``nosym``/``noinv``/``no_t_rev``. ``advisors/n_irr_k.py``'s ``nosym``
  input only affects *that advisor's own* irreducible-k-point count; no
  field on ``KSamplingDecision``/``PwSettings`` carries a resolved
  ``nosym`` decision forward, so there is nothing to translate into the
  actual QE input yet -- a pre-existing, documented gap in epic 6's
  output, not something this epic re-decides.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Literal

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element

from goldilocks_core.advisors.hubbard_u import expand_hubbard_label, manifold_for
from goldilocks_core.advisors.job_resources import JobDecision
from goldilocks_core.generation.errors import GenerationError
from goldilocks_core.generation.quantum_espresso.namelists import render_namelist
from goldilocks_core.step_settings import PwSettings
from goldilocks_core.steps import SharedContext, Step
from goldilocks_core.system_settings import SystemSettings

_QE_VDW_CORR: dict[str, tuple[str, int | None]] = {
    "d3bj": ("grimme-d3", 4),
}
"""Our own vdW-method vocabulary (``advisors/vdw_method.py``'s ``VdwMethod``)
-> QE's own keyword spelling. A name translation, not a decision: which
method to use was already decided upstream; only its QE spelling is
resolved here (matches v1's ``_QE_VDW_CORR``, ``generation/qe/scf.py:16-21``,
restricted to the one method ``VdwMethod`` currently allows)."""

_SAFE_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")


def canonicalize_system_for_generation(system: SystemSettings) -> SystemSettings:
    """The ``SystemSettings`` to actually render QE cards from: one QE
    species label per *(element, starting_magnetization value)* pair
    genuinely present, not one per whatever raw per-atom label the input
    structure happened to carry.

    Real input files often carry their own per-atom labels that have
    nothing to do with magnetism or with each other -- many CIFs assign
    one crystallographic label per site (``Si1``..``Si40`` for a plain,
    non-magnetic 40-atom silicon cell, or ``Fe0``/``Fe1`` for a 2-atom BCC
    iron cell, are real examples seen in practice, not hypotheticals).
    Naively trusting ``site.label`` -- or gating on whether AFM relabeling
    ran -- both under- and over-split: a non-magnetic structure's labels
    are pure noise (should always collapse to one label per element,
    regardless of magnetism), while a *spin-polarized but non-AFM*
    structure's default heuristic gives every site of one element the
    identical value (``advisors/magnetic_config.py``'s
    ``_starting_magnetization_by_label`` docstring is explicit about
    this) and should collapse too -- but a human-supplied override that
    hands two of that same real structure's labels genuinely different
    values, without ever opting into AFM, must **not** collapse. Grouping
    by the actual resolved value (same algorithm as
    ``magnetic_config._label_by_spin`` uses for AFM candidates --
    most-positive value first, deterministic) handles all three cases
    uniformly, with no need to know *why* two sites differ.

    Re-keys ``starting_magnetization``/``angle1``/``angle2`` to match the
    new labels in the same pass, so ``_magnetic_keywords`` below (which
    looks values up by ``species_index``'s own label) never sees a label
    it doesn't recognise. Shared by ``write_qe_scf`` and ``write_qe_relax``
    (``relax.py``) so the two don't drift."""
    magnetic = system.magnetic
    structure = magnetic.relabeled_structure
    if not structure.is_ordered:
        # A disordered structure has no single `site.specie` to group by --
        # leave it untouched and let the caller's own `is_ordered` check
        # (immediately after this call) raise its real, clean error instead
        # of an unrelated crash from here.
        return system

    old_to_new_label = _species_relabeling(structure, magnetic.starting_magnetization)
    relabeled = Structure(
        structure.lattice,
        structure.species,
        structure.frac_coords,
        labels=[old_to_new_label[site.label] for site in structure],
        site_properties=structure.site_properties,
    )

    def _rekeyed(values: dict[str, float] | None) -> dict[str, float] | None:
        if not values:
            return values
        return {old_to_new_label[label]: value for label, value in values.items()}

    return replace(
        system,
        magnetic=replace(
            magnetic,
            relabeled_structure=relabeled,
            starting_magnetization=_rekeyed(magnetic.starting_magnetization),
            angle1=_rekeyed(magnetic.angle1),
            angle2=_rekeyed(magnetic.angle2),
        ),
    )


def _species_relabeling(
    structure: Structure, magnetization: dict[str, float] | None
) -> dict[str, str]:
    """Map each real (old) site label to the QE species label it should
    use instead: one per *(element, resolved value)* pair actually
    present, most-positive value numbered first when an element needs
    more than one (same convention as ``magnetic_config._label_by_spin``).
    A missing/empty ``magnetization`` defaults every site to ``0.0``,
    which collapses every element to one label -- exactly the plain,
    non-spin-polarized case."""
    magnetization = magnetization or {}
    values_by_element: dict[str, list[float]] = {}
    for site in structure:
        value = magnetization.get(site.label, 0.0)
        seen = values_by_element.setdefault(site.specie.symbol, [])
        if value not in seen:
            seen.append(value)

    label_by_key: dict[tuple[str, float], str] = {}
    for element, values in values_by_element.items():
        if len(values) == 1:
            label_by_key[(element, values[0])] = element
            continue
        for index, value in enumerate(sorted(values, reverse=True), start=1):
            label_by_key[(element, value)] = f"{element}{index}"

    return {
        site.label: label_by_key[
            (site.specie.symbol, magnetization.get(site.label, 0.0))
        ]
        for site in structure
    }


def write_qe_scf(
    system: SystemSettings,
    step: PwSettings,
    job: JobDecision,
    ctx: SharedContext,
    purpose: Literal["scf", "nscf"] = "scf",
) -> list[Step]:
    """``purpose`` (v2 epic 9, #9) selects QE's own ``&CONTROL
    calculation`` value and this step's name/filenames -- both real,
    documented ``pw.x`` values, not a new scientific choice this writer
    makes (the *decision* to run an nscf step at all, with what k-mesh/
    occupations, is ``service/_dos.py``'s job, upstream of here). Added
    once ``dos``'s nscf step became a second real caller; ``scf`` stays
    the default so every existing caller is unaffected.
    """
    system = canonicalize_system_for_generation(system)
    structure = system.magnetic.relabeled_structure
    if not structure.is_ordered:
        raise GenerationError(
            "cannot generate Quantum ESPRESSO input for a disordered structure"
        )
    if step.relax is not None:
        raise GenerationError(
            "write_qe_scf does not render a relax/vc-relax input; "
            "PwSettings.relax must be None for an scf step"
        )
    for name, value in (
        ("occupations", step.occupations),
        ("k_sampling", step.k_sampling),
        ("nbnd", step.nbnd),
        ("convergence", step.convergence),
        ("parallel", step.parallel),
    ):
        if value is None:
            raise GenerationError(
                f"PwSettings.{name} is required to generate {purpose}.in"
            )
    elements = sorted({site.specie.symbol for site in structure})
    pseudo_by_element = validated_pseudo_by_element(elements, system)
    species_labels = sorted({site.label for site in structure})
    label_to_element = {site.label: site.specie.symbol for site in structure}
    species_index = {label: index + 1 for index, label in enumerate(species_labels)}

    keywords: dict[str, object] = {}
    keywords.update(control_keywords(ctx, step, job, purpose))
    keywords.update(
        system_keywords(structure, system, step, len(species_labels), species_index)
    )
    keywords.update(electrons_keywords(step))

    lines = [render_namelist(keywords)]
    lines.append(atomic_species(species_labels, label_to_element, pseudo_by_element))
    lines.append(cell_parameters(structure))
    lines.append(atomic_positions(structure))
    lines.append(k_points(step))
    lines.extend(hubbard_lines(system, structure, label_to_element))
    content = "\n".join(lines)

    args = ["-npool", str(step.parallel.npool)]
    if step.parallel.ndiag is not None:
        args += ["-ndiag", str(step.parallel.ndiag)]
    args += ["-in", f"{purpose}.in"]

    return [
        Step(
            name=purpose,
            executable="pw.x",
            args=tuple(args),
            files={f"{purpose}.in": content},
            stdout=f"{purpose}.out",
        )
    ]


def validated_pseudo_by_element(
    elements: list[str], system: SystemSettings
) -> dict[str, object]:
    pseudo_by_element = {pseudo.element: pseudo for pseudo in system.pseudopotentials}
    if len(pseudo_by_element) != len(system.pseudopotentials):
        raise GenerationError("pseudopotential selection contains duplicate elements")
    missing = sorted(set(elements) - set(pseudo_by_element))
    extra = sorted(set(pseudo_by_element) - set(elements))
    if missing or extra:
        raise GenerationError(
            "pseudopotential selection coverage mismatch; "
            f"missing: {', '.join(missing) or 'none'}; "
            f"extra: {', '.join(extra) or 'none'}"
        )
    for symbol, pseudo in pseudo_by_element.items():
        if pseudo.functional != system.functional:
            raise GenerationError(
                f"pseudopotential functional mismatch for {symbol}: "
                f"calculation requires {system.functional}, selected "
                f"{pseudo.functional or 'unknown'}"
            )
        if _SAFE_FILENAME.fullmatch(pseudo.filename) is None:
            raise GenerationError(
                f"unsafe pseudopotential filename: {pseudo.filename!r}"
            )
    return pseudo_by_element


def control_keywords(
    ctx: SharedContext,
    step: PwSettings,
    job: JobDecision,
    purpose: Literal["scf", "nscf", "relax", "vc-relax"],
) -> dict[str, object]:
    keywords: dict[str, object] = {
        "calculation": purpose,
        "prefix": ctx.prefix,
        "outdir": ctx.outdir,
        "pseudo_dir": ctx.pseudo_dir,
        "tprnfor": True,
        "tstress": True,
        "max_seconds": job.max_seconds,
    }
    if step.disk_io is not None:
        keywords["disk_io"] = step.disk_io
    if step.relax is not None:
        # forc_conv_thr/etot_conv_thr/nstep are real &CONTROL keys, not
        # &IONS ones (INPUT_PW.txt, ASE's own ALL_KEYS['pw']['control']
        # registry, both checked 2026-09-14) -- only reachable when
        # step.relax is set, i.e. only from write_qe_relax
        # (generation/quantum_espresso/relax.py, v2 epic 10, #10);
        # write_qe_scf itself still rejects a non-None step.relax.
        keywords["forc_conv_thr"] = step.relax.forc_conv_thr
        keywords["etot_conv_thr"] = step.relax.etot_conv_thr
        keywords["nstep"] = step.relax.nstep
    return keywords


def system_keywords(
    structure,
    system: SystemSettings,
    step: PwSettings,
    ntyp: int,
    species_index: dict[str, int],
) -> dict[str, object]:
    keywords: dict[str, object] = {
        "ibrav": 0,
        "nat": len(structure),
        "ntyp": ntyp,
        "ecutwfc": system.cutoffs.ecutwfc_ry,
        "ecutrho": system.cutoffs.ecutrho_ry,
        "nbnd": step.nbnd.nbnd,
    }
    keywords.update(_occupations_keywords(step))
    keywords.update(_magnetic_keywords(system, species_index))
    keywords.update(_charge_keywords(structure, system))
    if system.vdw.use_vdw:
        if system.vdw.method not in _QE_VDW_CORR:
            raise GenerationError(
                f"unsupported Quantum ESPRESSO vdW method: {system.vdw.method!r}"
            )
        vdw_corr, dftd3_version = _QE_VDW_CORR[system.vdw.method]
        keywords["vdw_corr"] = vdw_corr
        if dftd3_version is not None:
            keywords["dftd3_version"] = dftd3_version
    if system.boundary.assume_isolated != "none":
        keywords["assume_isolated"] = system.boundary.assume_isolated
    return keywords


def _occupations_keywords(step: PwSettings) -> dict[str, object]:
    decision = step.occupations
    keywords: dict[str, object] = {"occupations": decision.occupations}
    if decision.occupations == "smearing":
        keywords["smearing"] = decision.smearing_type
        keywords["degauss"] = decision.degauss
    return keywords


def _magnetic_keywords(
    system: SystemSettings, species_index: dict[str, int]
) -> dict[str, object]:
    magnetic = system.magnetic
    keywords: dict[str, object] = {}
    if magnetic.spin_orbit_enabled:
        if magnetic.tot_magnetization is not None:
            raise GenerationError(
                "tot_magnetization is not valid for a noncolinear (spin-orbit) "
                "calculation -- QE restricts it to the LSDA/collinear case; "
                "use starting_magnetization instead"
            )
        keywords["noncolin"] = True
        keywords["lspinorb"] = True
    elif magnetic.spin_polarized:
        keywords["nspin"] = 2
        if magnetic.tot_magnetization is not None:
            keywords["tot_magnetization"] = magnetic.tot_magnetization
    for field_name, values in (
        ("starting_magnetization", magnetic.starting_magnetization),
        ("angle1", magnetic.angle1),
        ("angle2", magnetic.angle2),
    ):
        if not values:
            continue
        for label, value in values.items():
            keywords[f"{field_name}({species_index[label]})"] = value
    return keywords


def _charge_keywords(structure, system: SystemSettings) -> dict[str, object]:
    """``system.electron_count.nelec`` (#34, v2 epic 9, #9) is this
    codebase's own resolved *total electron count* -- but QE's real
    ``&SYSTEM`` namelist has no ``nelec`` input keyword at all (``nelec``
    is a QE-computed/reported quantity, confirmed against ASE's own
    ``ase.io.espresso_namelist.keys.ALL_KEYS['pw']['system']`` registry,
    which lists ``tot_charge``, not ``nelec``). The real input is
    ``tot_charge``, the *deviation* from the neutral, pseudopotential
    -summed electron count -- so this recomputes that neutral baseline
    from ``system.pseudopotentials`` (already available here, no new
    dependency) and translates the difference, rather than emitting a
    keyword QE would reject outright. Omitted entirely when the
    requested count matches the neutral baseline exactly (the heuristic
    -default case, since ``advisors/electron_count.py``'s own heuristic
    branch uses this identical formula) -- writing ``tot_charge = 0.0``
    would be harmless but is pure noise."""
    z_valence_by_element = {
        pseudo.element: pseudo.z_valence for pseudo in system.pseudopotentials
    }
    counts = structure.composition.get_el_amt_dict()
    natural_nelec = sum(
        z_valence_by_element[element] * count for element, count in counts.items()
    )
    tot_charge = natural_nelec - system.electron_count.nelec
    if tot_charge == 0.0:
        return {}
    return {"tot_charge": tot_charge}


def electrons_keywords(step: PwSettings) -> dict[str, object]:
    decision = step.convergence
    keywords: dict[str, object] = {
        "conv_thr": decision.conv_thr,
        "mixing_beta": decision.mixing_beta,
        "electron_maxstep": decision.electron_maxstep,
        "mixing_mode": decision.mixing_mode,
    }
    if decision.mixing_fixed_ns is not None:
        keywords["mixing_fixed_ns"] = decision.mixing_fixed_ns
    return keywords


def cell_parameters(structure) -> str:
    lines = ["CELL_PARAMETERS angstrom"]
    lines.extend(
        "  " + "  ".join(_format_float(value) for value in vector)
        for vector in structure.lattice.matrix
    )
    return "\n".join(lines) + "\n"


def atomic_species(
    species_labels: list[str],
    label_to_element: dict[str, str],
    pseudo_by_element: dict[str, object],
) -> str:
    """One line per QE species *label* (``Fe1``/``Fe2``/``O``), not per
    real element -- an AFM-relabeled structure needs two ``ATOMIC_SPECIES``
    entries for one physical element, both pointing at the same
    pseudopotential file (mass/pseudopotential lookup still needs the real
    element underneath; ``Element("Fe1")`` isn't a real periodic-table
    symbol)."""
    lines = ["ATOMIC_SPECIES"]
    for label in species_labels:
        element = label_to_element[label]
        pseudo = pseudo_by_element[element]
        mass = _format_float(float(Element(element).atomic_mass))
        lines.append(f"  {label}  {mass}  {pseudo.filename}")
    return "\n".join(lines) + "\n"


def atomic_positions(
    structure, fixed_site_indices: frozenset[int] | None = None
) -> str:
    """``fixed_site_indices`` (#44): 0-based indices, in ``structure``'s
    own site order, of atoms to pin with an ``if_pos`` column. Appended
    only to fixed atoms' rows, never to free atoms' -- matching ASE's
    own ``ase.io.espresso`` writer convention (confirmed by reading its
    source 2026-09-14: it omits the mask entirely for any atom with
    nothing fixed, relying on QE's per-row default of ``1 1 1`` rather
    than writing it out for every atom)."""
    fixed_site_indices = fixed_site_indices or frozenset()
    lines = ["ATOMIC_POSITIONS crystal"]
    for index, site in enumerate(structure):
        coords = "  ".join(_format_float(value) for value in site.frac_coords)
        line = f"  {site.label}  {coords}"
        if index in fixed_site_indices:
            line += "  0  0  0"
        lines.append(line)
    return "\n".join(lines) + "\n"


def hubbard_card(u_by_label: dict[str, float], label_to_element: dict[str, str]) -> str:
    """``plan="table"`` only -- ``u_by_label`` is ``expand_hubbard_label``'s
    already species-label-keyed output (module docstring's own ``U``-line
    citation for why this is label-keyed, not atom-index-keyed).
    ``ortho-atomic`` is QE's own recommended projector choice
    (Doc/Hubbard_input.tex: "It is recommended to use ortho-atomic
    whenever possible") -- this writer has no resolved fact to pick a
    different one from, so it is the one fixed literal here, the same
    role ``ibrav = 0`` plays above."""
    lines = ["HUBBARD (ortho-atomic)"]
    for label in sorted(u_by_label):
        manifold = manifold_for(label_to_element[label])
        lines.append(f"  U  {label}-{manifold}  {_format_float(u_by_label[label])}")
    return "\n".join(lines) + "\n"


def hubbard_lines(
    system: SystemSettings, structure, label_to_element: dict[str, str]
) -> list[str]:
    """Every line ``system.hubbard`` contributes to the card list, shared
    by ``write_qe_scf``/``write_qe_relax`` so neither writer's own branch
    count carries this a second time. Raises for
    ``plan="self_consistent_calibration_needed"`` (see ``hubbard_card``'s
    own docstring for why); returns ``[]`` for ``plan="not_needed"``."""
    if system.hubbard.plan == "self_consistent_calibration_needed":
        raise GenerationError(
            "a Hubbard +U correction needs self-consistent calibration "
            "(plan='self_consistent_calibration_needed'), but rendering an "
            "hp.x calibration input is not implemented by this writer yet -- "
            "see advisors/hubbard_u.py's own CalibrationRequest"
        )
    if system.hubbard.plan != "table":
        return []
    u_by_label = expand_hubbard_label(system.hubbard.u_by_element, structure)
    return [hubbard_card(u_by_label, label_to_element)]


def k_points(step: PwSettings) -> str:
    grid = step.k_sampling.mesh
    shift = step.k_sampling.shift
    return (
        "K_POINTS automatic\n"
        f"  {grid[0]}  {grid[1]}  {grid[2]}  {shift[0]}  {shift[1]}  {shift[2]}\n"
    )


def _format_float(value: float) -> str:
    return f"{value:.10g}"
