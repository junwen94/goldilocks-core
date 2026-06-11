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

_TASKS = ["scf", "relax", "vc-relax", "nscf", "bands", "md", "vc-md"]
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
    for i, t in enumerate(_TASKS, 1):
        suffix = "  [dim](default)[/dim]" if t == "scf" else ""
        console.print(f"    [cyan]{i})[/cyan] {t}{suffix}")
    t_choice = Prompt.ask(
        "  Select  [dim](press Enter for default)[/dim]",
        choices=[str(i) for i in range(1, len(_TASKS) + 1)],
        default="1",
        show_default=False,
    )
    task = _TASKS[int(t_choice) - 1]

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
        console.print("  Enter key=value pairs, blank line to finish:")
        while True:
            h = Prompt.ask("  hint", default="")
            if not h:
                break
            k, _, v = h.partition("=")
            if k.strip():
                hints[k.strip()] = _coerce(v.strip())

    # --- build intent and run pipeline -----------------------------------
    from goldilocks_core.advise.pipeline import build_qe_parameter_set
    from goldilocks_core.intent import CalculationIntent

    _task = cast(
        Literal["scf", "nscf", "bands", "relax", "md", "vc-relax", "vc-md"],
        task,
    )
    _accuracy = cast(Literal["fast", "balanced", "accurate"], accuracy)

    intent = CalculationIntent(
        structure=ctx.structure,
        code="qe",
        task=_task,
        xc="pbesol",
        pseudo_family="PseudoDojo/0.4/PBEsol/SR/standard/upf",
        accuracy=_accuracy,
        hints=hints,
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

        params = build_qe_parameter_set(
            ctx.analysis, intent, k_distance_ml=k_distance_ml
        )

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
        _generate_qe(console, params, ctx.structure, intent, raw_dir)


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
    elif analysis.has_d_electrons:
        _flag("DFT+U / Hubbard",
              f"consider — d-electron elements: {', '.join(analysis.magnetic_elements)}", "info")
    else:
        _flag("DFT+U / Hubbard", "not needed", "ok")

    # van der Waals
    if analysis.dimensionality in ("2d", "1d") or analysis.has_vacuum:
        _flag("van der Waals", "consider — low-dimensional / layered system", "info")
    else:
        _flag("van der Waals", "not needed (3D bulk)", "ok")

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


def _generate_qe(
    console: Console,
    params: Any,
    structure: Any,
    intent: Any,
    output_dir: str,
) -> None:
    """Write goldilocks.in + copy pseudopotentials, then report to console."""
    from goldilocks_core.generate.qe import write_qe_inputs

    with console.status("  Writing files…", spinner="dots"):
        try:
            result = write_qe_inputs(params, structure, intent, output_dir=output_dir)
        except Exception as exc:
            console.print(f"  [red]Error generating files:[/red] {exc}")
            return

    input_path = result["input_file"]
    pseudo_path = result["pseudo_dir"]
    missing: list[str] = result.get("missing_pp", [])

    console.print()
    console.print(f"  [green]✓[/green] [bold]{input_path}[/bold]")

    # List copied pp files
    if pseudo_path.exists():
        pp_files = sorted(pseudo_path.iterdir())
        for f in pp_files:
            console.print(f"  [green]✓[/green] {f}")

    if missing:
        console.print(
            f"\n  [yellow]⚠[/yellow] pp files not found for: "
            f"{', '.join(missing)} — add them manually to {pseudo_path}"
        )

    console.print()
    console.print(
        "  [dim]Run with:[/dim]  pw.x -in goldilocks.in > goldilocks.out",
        highlight=False,
    )


def _coerce(value: str) -> Any:
    for try_type in (int, float):
        try:
            return try_type(value)
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value
