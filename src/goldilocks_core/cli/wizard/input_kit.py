"""Wizard Input Kit: collect intent, build CalculationIntent, show parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from goldilocks_core.cli.wizard._context import WizardContext

_TASKS: list[tuple[str, str]] = [
    ("scf",      "self-consistent field"),
    ("relax",    "ionic relaxation, fixed cell"),
    ("vc-relax", "variable-cell relaxation"),
    ("nscf",     "non-self-consistent field  (DOS / Fermi surface)"),
    ("bands",    "band structure along k-path"),
    ("ph",       "phonon calculation (ph.x)"),
    ("md",       "Born-Oppenheimer MD, fixed cell"),
    ("vc-md",    "variable-cell MD"),
]
_ACCURACY = ["fast", "balanced", "accurate"]
_RY_TO_EV: float = 13.605693122994


def run(
    console: Console,
    ctx: WizardContext | None = None,
    mode: str = "qe",
) -> None:
    """Collect task/accuracy/hints, run the advise pipeline, and display results.

    *mode* is ``"agnostic"`` for a code-independent summary or ``"qe"`` for
    Quantum ESPRESSO parameters.  If *ctx* is None the Pre-Analysis step runs
    first to obtain the structure and analysis.
    """
    if ctx is None:
        from goldilocks_core.cli.wizard.pre_analysis import run as pre_run

        ctx = pre_run(console)
        if ctx is None:
            return

    # --- task ------------------------------------------------------------
    console.print()
    console.print("  [bold]Calculation task[/bold]")
    for i, (name, desc) in enumerate(_TASKS, 1):
        default_tag = "  [dim](default)[/dim]" if name == "scf" else ""
        console.print(f"    [cyan]{i})[/cyan]  {name:<10} — {desc}{default_tag}")
    t_choice = Prompt.ask(
        "  Select  [dim](press Enter for default)[/dim]",
        choices=[str(i) for i in range(1, len(_TASKS) + 1)],
        default="1",
        show_default=False,
    )
    task = _TASKS[int(t_choice) - 1][0]

    # --- accuracy --------------------------------------------------------
    console.print()
    console.print("  [bold]Accuracy tier[/bold]")
    for i, a in enumerate(_ACCURACY, 1):
        suffix = "  [dim](default)[/dim]" if a == "balanced" else ""
        console.print(f"    [cyan]{i})[/cyan] {a}{suffix}")
    a_choice = Prompt.ask(
        "  Select  [dim](press Enter for default)[/dim]",
        choices=["1", "2", "3"],
        default="2",
        show_default=False,
    )
    accuracy = _ACCURACY[int(a_choice) - 1]

    # --- hints -----------------------------------------------------------
    hints: dict[str, Any] = {}
    console.print()
    if Confirm.ask("  Add parameter hints? (key=value overrides)", default=False):
        from goldilocks_core.cli._parse import coerce_hint_value

        console.print("  Enter key=value pairs, blank line to finish:")
        while True:
            h = Prompt.ask("  hint", default="")
            if not h:
                break
            k, _, v = h.partition("=")
            if k.strip():
                hints[k.strip()] = coerce_hint_value(v.strip())

    # --- ph-specific: setup advise + q-grid confirm ----------------------
    ph_nq: tuple[int, int, int] | None = None
    ph_setup = None
    if task == "ph":
        from goldilocks_core.advise.phonon import advise_ph_setup

        # Need base k-grid: build a temporary intent for the advise pipeline
        from goldilocks_core.advise.pipeline import advise as _advise
        from goldilocks_core.intent import CalculationIntent as _CI
        from goldilocks_core.select.qe import build_qe_parameter_set as _bqps

        _tmp_intent = _CI(
            structure=ctx.structure, code="qe", task="scf",
            accuracy=cast(Literal["fast", "balanced", "accurate"], accuracy),
        )
        _tmp_params = _bqps(_advise(ctx.analysis, _tmp_intent))
        ph_setup = advise_ph_setup(
            ctx.structure, ctx.analysis,
            cast(Literal["fast", "balanced", "accurate"], accuracy),
            _tmp_params.kpoints_grid,
        )

        _display_ph_setup(console, ph_setup)

        nq = ph_setup.q_grid.nq
        q_raw = Prompt.ask(
            "  Q-grid",
            default=f"{nq[0]} {nq[1]} {nq[2]}",
            show_default=True,
        ).strip()
        if q_raw:
            parts = q_raw.split()
            if len(parts) == 3 and all(p.isdigit() for p in parts):
                ph_nq = (int(parts[0]), int(parts[1]), int(parts[2]))
            else:
                console.print(f"  [yellow]⚠[/yellow] Invalid — using recommended {nq[0]} {nq[1]} {nq[2]}")
                ph_nq = nq

    # --- build intent and run pipeline -----------------------------------
    from goldilocks_core.advise.pipeline import advise
    from goldilocks_core.intent import CalculationIntent, ParameterHints
    from goldilocks_core.select.qe import build_qe_parameter_set

    _task = cast(
        Literal["scf", "nscf", "bands", "relax", "md", "vc-relax", "vc-md", "ph"],
        task,
    )
    _accuracy = cast(Literal["fast", "balanced", "accurate"], accuracy)

    try:
        typed_hints = ParameterHints.from_dict(hints)
    except ValueError as exc:
        console.print(f"\n  [red]Error:[/red] invalid hint — {exc}")
        return

    intent = CalculationIntent(
        structure=ctx.structure,
        code="qe",
        task=_task,
        xc="pbesol",
        pseudo_family="PseudoDojo/0.4/PBEsol/SR/standard/upf",
        accuracy=_accuracy,
        hints=typed_hints,
    )

    k_distance_ml: float | None = None

    with console.status("  Running ML inference and advise pipeline…", spinner="dots"):
        # K-points ML: CGCNN + QRF predictor; graceful fallback to heuristic
        try:
            from goldilocks_core.ml.loader import try_load_kpoints_predictor

            kp = try_load_kpoints_predictor()
            if kp is not None:
                kdist, _upper, _lower = kp.predict(ctx.structure)
                k_distance_ml = max(0.001, float(kdist))
        except Exception:
            pass

        bundle = advise(ctx.analysis, intent, k_distance_ml=k_distance_ml)
        params = build_qe_parameter_set(bundle)

    console.print()
    if mode == "agnostic":
        _display_agnostic(console, ctx.analysis, intent, params, k_distance_ml=k_distance_ml)
    else:
        from goldilocks_core.cli.commands.input import _display

        explain = Confirm.ask("  Show full rationale for each decision?", default=False)
        _display(console, ctx.structure_path, ctx.analysis, intent, params, explain)

    # ── Generate QE input files ─────────────────────────────────────────────
    console.print()
    if Confirm.ask("  Generate QE input files?", default=True):
        raw_dir = Prompt.ask(
            "  Output directory  [dim](press Enter for default)[/dim]",
            default="./goldilocks_output",
            show_default=False,
        )
        _generate_qe(console, params, ctx.structure, intent, raw_dir, ph_nq=ph_nq, ph_setup=ph_setup)


# ---------------------------------------------------------------------------
# Code-agnostic display
# ---------------------------------------------------------------------------

def _how_spin(params: Any, analysis: Any) -> Text:
    """Multi-line 'How' cell for the Spin row."""
    prov = params.spin_decision.provenance
    t = Text()
    if prov == "user_hint":
        t.append("user override", style="bold yellow")
        return t
    if prov == "ML":
        conf = analysis.magnetic_confidence
        conf_str = f" {conf:.0%}" if conf is not None else ""
        t.append(f"ML (mMACE{conf_str})", style="bold cyan")
        # Heuristic comparison: replicate _heuristic_treatment logic
        if not analysis.magnetic_elements:
            heur = "non_magnetic"
        elif analysis.soc_relevant:
            heur = "non_collinear_soc"
        else:
            heur = "collinear"
        if heur == params.spin_decision.treatment:
            t.append("\n✓ heuristic: agree", style="dim")
        else:
            t.append(f"\n⚠ heuristic: {heur}", style="yellow")
        return t
    # heuristic
    t.append("heuristic", style="dim")
    mag = analysis.magnetic_elements
    if mag:
        els = ", ".join(mag[:3]) + ("…" if len(mag) > 3 else "")
        t.append(f"\n({els})", style="dim")
    else:
        t.append("\n(no mag. elements)", style="dim")
    return t


def _how_kmesh(params: Any, analysis: Any, intent: Any, k_distance_ml: float | None) -> Text:
    """Multi-line 'How' cell for the K-mesh row."""
    from goldilocks_core.advise.protocol import select_protocol

    prov = params.kpoints_decision.provenance
    t = Text()
    if prov == "user_hint":
        t.append("user override", style="bold yellow")
        return t
    proto, _ = select_protocol(analysis, intent)
    if prov == "ML" and k_distance_ml is not None:
        t.append("ML (CGCNN+QRF)", style="bold cyan")
        t.append(f"\npredicted {k_distance_ml:.4f} Å⁻¹", style="dim")
        diff = abs(k_distance_ml - proto.k_distance)
        if diff > 0.02:
            t.append(f"\n⚠ heuristic {proto.k_distance:.2f} Å⁻¹", style="yellow")
        else:
            t.append(f"\n✓ heuristic {proto.k_distance:.2f} Å⁻¹ ≈", style="dim")
        return t
    # heuristic
    t.append("heuristic", style="dim")
    t.append(f"\n({intent.accuracy} → {proto.k_distance:.2f} Å⁻¹)", style="dim")
    return t


def _how_occupation(params: Any, analysis: Any) -> Text:
    """Multi-line 'How' cell for the Occupation scheme row."""
    prov = params.smearing_decision.provenance
    t = Text()
    if prov == "user_hint":
        t.append("user override", style="bold yellow")
        return t
    src = getattr(analysis, "metallicity_source", "heuristic")
    if src in ("ml", "both"):
        conf = getattr(analysis, "metallicity_confidence", None)
        conf_str = f" {conf:.0%}" if conf is not None else ""
        t.append(f"ML{conf_str}", style="bold cyan")
        t.append(f"\n({analysis.metallicity})", style="dim")
        return t
    t.append("heuristic", style="dim")
    t.append(f"\n({analysis.metallicity})", style="dim")
    return t


def _how_rel() -> Text:
    """'How' cell for Relativistic treatment — always heuristic."""
    t = Text()
    t.append("heuristic", style="dim")
    t.append("\n(Z > 30 rule)", style="dim")
    return t


def _display_agnostic(
    console: Console,
    analysis: Any,
    intent: Any,
    params: Any,
    k_distance_ml: float | None = None,
) -> None:
    """Display physics-level recommendations, independent of any DFT code."""
    formula = intent.structure.composition.reduced_formula
    console.print()
    console.rule(
        f"[bold]goldilocks[/bold]  ·  [bold blue]{formula}[/bold blue]"
        f"  ·  {intent.task}  ·  {intent.accuracy}  ·  code-agnostic",
        style="blue",
    )

    _TREATMENT_LABEL = {
        "non_magnetic":      "Non-magnetic",
        "collinear":         "Spin-polarised (collinear)",
        "non_collinear":     "Non-collinear spin",
        "non_collinear_soc": "Non-collinear + SOC",
    }
    _REL_LABEL = {"SR": "Scalar-relativistic", "FR": "Fully-relativistic"}

    # ── Section 1: computed recommendations from the advise pipeline ────────
    console.print()
    console.print("  [bold]Parameter Recommendations[/bold]")
    t = Table(box=box.SIMPLE_HEAD, padding=(0, 2), show_edge=False)
    t.add_column("Physics",        style="bold", min_width=20)
    t.add_column("Recommendation", min_width=38)
    t.add_column("How",            min_width=26)

    # Spin
    treatment = params.spin_decision.treatment
    spin_val = Text(_TREATMENT_LABEL.get(treatment, treatment))
    if params.starting_magnetization:
        mag_pairs = "  ".join(
            f"{el} {v:+.1f} μB" for el, v in params.starting_magnetization.items()
        )
        spin_val.append(f"  ({mag_pairs})", style="dim")
    t.add_row("Spin", spin_val, _how_spin(params, analysis))

    # Relativistic treatment (SR vs FR) — purely from structure, code-agnostic
    if analysis.soc_relevant:
        soc_val = Text("Fully-relativistic (SOC)")
        soc_val.append("  (heavy elements, Z > 30)", style="dim")
    else:
        soc_val = Text("Scalar-relativistic")
        soc_val.append("  (no heavy elements)", style="dim")
    t.add_row("Relativistic treatment", soc_val, _how_rel())

    # Occupation scheme
    if params.smearing_decision.use_smearing:
        sm_val = Text("Smearing required")
        sm_val.append(
            f"  ({params.smearing}, {params.smearing_decision.width_ev:.3f} eV)",
            style="dim",
        )
    else:
        sm_val = Text("Fixed occupations")
        sm_val.append("  (gapped / insulating)", style="dim")
    t.add_row("Occupation scheme", sm_val, _how_occupation(params, analysis))

    # K-mesh
    g = params.kpoints_grid
    kmesh_val = Text(f"Grid {g[0]}×{g[1]}×{g[2]}")
    kmesh_val.append("  (Monkhorst-Pack)", style="dim")
    t.add_row("K-mesh", kmesh_val, _how_kmesh(params, analysis, intent, k_distance_ml))

    console.print(t)

    # ── Section 2: advisory flags from structure analysis ───────────────────
    console.print()
    console.print("  [bold]Flags to Check[/bold]"
                  "  [dim](goldilocks has not computed these — verify manually)[/dim]")
    ft = Table(box=None, padding=(0, 2), show_edge=False, show_header=False)
    ft.add_column("flag",  style="bold", min_width=20)
    ft.add_column("note",  min_width=50)

    def _flag(label: str, note: str, level: str = "info") -> None:
        icon = {"warn": "[yellow]⚠[/yellow]", "ok": "[dim]—[/dim]", "info": "[cyan]·[/cyan]"}[level]
        ft.add_row(f"{icon}  {label}", Text(note, style="dim" if level == "ok" else ""))

    # Dipole correction
    if analysis.is_slab and analysis.is_polar:
        _flag("Dipole correction", "likely needed — polar slab with net dipole", "warn")
    elif analysis.is_slab:
        _flag("Dipole correction", "check — slab geometry detected", "info")
    else:
        _flag("Dipole correction", "not applicable (bulk)", "ok")

    # DFT+U
    if analysis.has_f_electrons:
        _flag("DFT+U / Hubbard", "strongly recommended — f-electron elements present", "warn")
    elif analysis.has_d_electrons and analysis.metallicity not in {"metallic", "likely_metallic"}:
        _flag("DFT+U / Hubbard",
              f"consider — d-electron elements: {', '.join(analysis.magnetic_elements)}", "info")
    elif analysis.has_d_electrons:
        _flag("DFT+U / Hubbard", "not needed — itinerant d-electrons (metallic)", "ok")
    else:
        _flag("DFT+U / Hubbard", "not needed", "ok")

    # van der Waals — reflect actual pipeline decision
    vdw = params.vdw_decision
    if vdw.use_vdw:
        _flag(
            "van der Waals",
            f"{vdw.method} applied ({params.vdw_corr})",
            "info",
        )
    else:
        _flag("van der Waals", "not applied (3D bulk) — set hint use_vdw=True if needed", "ok")

    console.print(ft)

    # ── Rationale ────────────────────────────────────────────────────────────
    console.print()
    console.print("  [bold]Rationale[/bold]")
    rt = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
    rt.add_column("key", style="dim", min_width=20)
    rt.add_column("rationale", overflow="fold", max_width=66)
    rt.add_row("Spin",              escape(params.spin_decision.rationale))
    rt.add_row("Occupation scheme", escape(params.smearing_decision.rationale))
    rt.add_row("K-mesh",            escape(params.kpoints_decision.rationale))
    console.print(rt)
    console.print()


def _display_ph_setup(console: Console, ph_setup: Any) -> None:
    """Print the phonon setup checklist panel."""
    from rich.panel import Panel

    qadv = ph_setup.q_grid
    nq   = qadv.nq
    kg   = ph_setup.phonon_kgrid

    lines: list[str] = [
        "  [bold]Q-grid[/bold]  [dim](heuristic)[/dim]\n"
        f"  Recommended  [bold cyan]{nq[0]} {nq[1]} {nq[2]}[/bold cyan]"
        f"  [dim]IFC range ≈ {qadv.target_range_aa:.0f} Å[/dim]\n"
        f"  [dim]{qadv.rationale}[/dim]\n",

        "  [bold]Auto-applied to goldilocks.in (SCF for phonon)[/bold]\n"
        f"  [green]✓[/green]  conv_thr = {ph_setup.scf_conv_thr:.0e}"
        "  [dim](tighter than standard SCF)[/dim]\n"
        f"  [green]✓[/green]  k-grid  {kg[0]} {kg[1]} {kg[2]}"
        f"  [dim](commensurate with q-grid)[/dim]\n",

        "  [bold]Auto-applied to ph.in[/bold]\n"
        f"  [green]✓[/green]  tr2_ph = {ph_setup.tr2_ph:.0e}\n"
        + (
            "  [green]✓[/green]  epsil = .true."
            "  [dim](polar insulator → Born charges + ε∞ for LO-TO splitting)[/dim]\n"
            if ph_setup.needs_epsil else
            "  [dim]–[/dim]  epsil = .false.  [dim](non-polar)[/dim]\n"
        ),

        "  [bold]Advisory (not in generated files)[/bold]\n"
        "  [dim]·[/dim]  Relax structure first: "
        "force < 1e-4 Ry/Bohr, stress < 0.1 kbar\n"
        "  [dim]·[/dim]  matdyn.x: asr = 'crystal'\n"
        "  [dim]·[/dim]  Verify q-grid convergence: compare phonon dispersion "
        "at successive grids\n",
    ]

    if ph_setup.warnings:
        warn_lines = "\n".join(
            f"  [yellow]⚠[/yellow]  {w}" for w in ph_setup.warnings
        )
        lines.append(f"  [bold]Warnings[/bold]\n{warn_lines}\n")

    console.print()
    console.print(
        Panel(
            "\n".join(lines),
            title="[bold]Phonon Setup[/bold]  ·  heuristic",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print()


def _generate_qe(
    console: Console,
    params: Any,
    structure: Any,
    intent: Any,
    output_dir: str,
    ph_nq: tuple[int, int, int] | None = None,
    ph_setup: Any = None,
) -> None:
    """Write gl-pw-{task}.in (and gl-ph.in for ph tasks) + copy pseudopotentials."""
    from goldilocks_core.generate.qe import write_ph_inputs, write_qe_inputs

    is_ph = intent.task == "ph"
    run_dir = _next_run_dir(output_dir)

    with console.status("  Writing files…", spinner="dots"):
        try:
            # For ph tasks, generate an SCF input with phonon-appropriate settings
            scf_intent = intent
            if is_ph:
                from goldilocks_core.intent import CalculationIntent

                scf_intent = CalculationIntent(
                    structure=intent.structure,
                    code=intent.code,
                    task="scf",
                    xc=intent.xc,
                    pseudo_family=intent.pseudo_family,
                    accuracy=intent.accuracy,
                    hints=intent.hints,
                )
            kgrid_override = ph_setup.phonon_kgrid if (is_ph and ph_setup) else None
            conv_thr       = ph_setup.scf_conv_thr  if (is_ph and ph_setup) else None
            result = write_qe_inputs(
                params, structure, scf_intent, output_dir=run_dir,
                kgrid_override=kgrid_override, conv_thr=conv_thr,
            )
            ph_result = write_ph_inputs(
                output_dir=run_dir,
                nq=ph_nq,
                epsil=ph_setup.needs_epsil if ph_setup else False,
                tr2_ph=ph_setup.tr2_ph if ph_setup else 1e-14,
            ) if is_ph else None
        except Exception as exc:
            console.print(f"  [red]Error generating files:[/red] {exc}")
            return

    input_path = result["input_file"]
    pseudo_path = result["pseudo_dir"]
    missing: list[str] = result.get("missing_pp") or []  # type: ignore[assignment]

    console.print()
    console.print(f"  [bold]Output:[/bold] {run_dir}")
    console.print(f"  [green]✓[/green] [bold]{input_path.name}[/bold]")
    if ph_result:
        console.print(f"  [green]✓[/green] [bold]{ph_result['ph_file'].name}[/bold]")

    if pseudo_path.exists():
        for f in sorted(pseudo_path.iterdir()):
            console.print(f"  [green]✓[/green] pseudo/{f.name}")

    if missing:
        console.print(
            f"\n  [yellow]⚠[/yellow] pp files not found for: "
            f"{', '.join(missing)} — add them manually to {pseudo_path}"
        )

    run_name = run_dir.name
    console.print()
    if is_ph:
        console.print(
            f"  [dim]Run with:[/dim]  pw.x -in {run_name}/gl-pw-scf.in > {run_name}/gl-pw-scf.out",
            highlight=False,
        )
        console.print(
            f"             then:  ph.x -in {run_name}/gl-ph.in > {run_name}/gl-ph.out",
            highlight=False,
        )
    else:
        console.print(
            f"  [dim]Run with:[/dim]  pw.x -in {run_name}/gl-pw-{intent.task}.in"
            f" > {run_name}/gl-pw-{intent.task}.out",
            highlight=False,
        )


def _next_run_dir(base: str) -> Any:
    """Return the next unused run_NNN Path under *base*, creating it."""
    from pathlib import Path

    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        candidate = base_path / f"run_{n:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        n += 1


