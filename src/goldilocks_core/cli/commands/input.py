"""``gl input`` — recommend QE parameters for a structure file."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

_RY_TO_EV: float = 13.605693122994

_PROV_STYLE: dict[str, str] = {
    "heuristic": "dim",
    "ML":        "bold cyan",
    "MLIP":      "bold magenta",
    "user_hint": "bold yellow",
}

_VALID_TASKS = {
    "scf", "nscf", "bands", "relax", "md", "vc-relax", "vc-md", "ph",
}

_VALID_ACCURACY = {"fast", "balanced", "accurate"}

_VALID_CODES = {"qe"}
_VALID_XC = {"pbesol"}

_DEFAULT_PSEUDO_SR = "PseudoDojo/0.4/PBEsol/SR/standard/upf"
_DEFAULT_PSEUDO_FR = "PseudoDojo/0.4/PBEsol/FR/standard/upf"


def run(
    structure: Annotated[
        Path,
        typer.Option("--structure", "-s", help="Structure file (CIF, POSCAR, XSF, …)"),
    ],
    task: Annotated[
        str,
        typer.Option("--task", "-t", help="Calculation task: scf | relax | vc-relax | nscf | bands | md | vc-md"),
    ] = "scf",
    accuracy: Annotated[
        str,
        typer.Option("--accuracy", "-a", help="Accuracy tier: fast | balanced | accurate"),
    ] = "balanced",
    code: Annotated[
        str,
        typer.Option("--code", "-c", help="DFT code [qe]"),
    ] = "qe",
    xc: Annotated[
        str,
        typer.Option("--xc", help="Exchange-correlation functional [pbesol]"),
    ] = "pbesol",
    pseudo: Annotated[
        Optional[str],
        typer.Option(
            "--pseudo", "--pp",
            help="Pseudo family.  Default: PseudoDojo SR (auto-upgrades to FR for SOC structures).  "
                 "Example: --pseudo PseudoDojo/0.4/PBEsol/FR/standard/upf",
        ),
    ] = None,
    hints: Annotated[
        Optional[list[str]],
        typer.Option(
            "--hints", "-H",
            help="Parameter overrides as key=value (repeatable).  "
                 "Example: -H spin_treatment=collinear -H ecutwfc_ev=680",
        ),
    ] = None,
    explain: Annotated[
        bool,
        typer.Option("--explain", "-e", help="Show full rationale for every decision"),
    ] = False,
    output: Annotated[
        Optional[str],
        typer.Option("--output", "-o", help="Generate QE input files in this directory"),
    ] = None,
) -> None:
    """Recommend QE input parameters for STRUCTURE."""
    console = Console()

    # --- validate args early ---
    if task not in _VALID_TASKS:
        console.print(
            f"[red]Error:[/red] unknown task {task!r}. "
            f"Valid: {', '.join(sorted(_VALID_TASKS))}"
        )
        raise typer.Exit(1)
    if task == "ph" and output is None:
        console.print(
            "[yellow]Note:[/yellow] ph task — pass --output <dir> to also generate "
            "gl-pw-scf.in and gl-ph.in files."
        )
    if accuracy not in _VALID_ACCURACY:
        console.print(
            f"[red]Error:[/red] unknown accuracy {accuracy!r}. "
            f"Valid: fast, balanced, accurate"
        )
        raise typer.Exit(1)
    if code not in _VALID_CODES:
        console.print(
            f"[red]Error:[/red] unsupported code {code!r}. "
            f"Currently supported: {', '.join(sorted(_VALID_CODES))}"
        )
        raise typer.Exit(1)
    if xc not in _VALID_XC:
        console.print(
            f"[red]Error:[/red] unsupported XC functional {xc!r}. "
            f"Currently supported: {', '.join(sorted(_VALID_XC))}"
        )
        raise typer.Exit(1)

    pseudo_family = pseudo if pseudo is not None else _DEFAULT_PSEUDO_SR

    # --- parse hints ---
    parsed_hints: dict[str, Any] = {}
    if hints:
        for h in hints:
            k, _, v = h.partition("=")
            parsed_hints[k.strip()] = _coerce(v.strip())

    # --- load structure ---
    try:
        from goldilocks_core.io.structures import load_structure
        structure_obj = load_structure(structure)
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] file not found: {structure}")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Error:[/red] could not parse structure — {exc}")
        raise typer.Exit(1)

    # --- analyse ---
    from goldilocks_core.analyse.structure import analyze_structure
    with console.status("Analysing structure…", spinner="dots"):
        analysis = analyze_structure(structure_obj)

    # --- build intent ---
    from typing import Literal, cast

    from goldilocks_core.intent import CalculationIntent
    _task = cast(
        Literal["scf", "nscf", "bands", "relax", "md", "vc-relax", "vc-md", "ph"],
        task,
    )
    _accuracy = cast(Literal["fast", "balanced", "accurate"], accuracy)
    intent = CalculationIntent(
        structure=structure_obj,
        code=code,
        task=_task,
        xc=xc,
        pseudo_family=pseudo_family,
        accuracy=_accuracy,
        hints=parsed_hints,
    )

    # --- advise ---
    from goldilocks_core.advise.pipeline import build_qe_parameter_set
    try:
        with console.status("Running advise pipeline…", spinner="dots"):
            params = build_qe_parameter_set(analysis, intent)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    _display(console, structure, analysis, intent, params, explain)

    # ── Optional file generation ──────────────────────────────────────────────
    if output is not None:
        _generate(console, params, structure_obj, intent, analysis, output)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _generate(console: Console, params: Any, structure: Any, intent: Any, analysis: Any, output: str) -> None:
    """Write input files into a new run_NNN/ sub-directory."""
    from goldilocks_core.generate.qe import write_ph_inputs, write_qe_inputs

    base = Path(output)
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        run_dir = base / f"run_{n:03d}"
        if not run_dir.exists():
            run_dir.mkdir(parents=True)
            break
        n += 1

    is_ph = intent.task == "ph"
    with console.status("  Writing files…", spinner="dots"):
        try:
            scf_intent = intent
            ph_setup = None
            if is_ph:
                from typing import Literal as _Lit
                from typing import cast as _cast

                from goldilocks_core.advise.phonon import advise_ph_setup
                from goldilocks_core.intent import CalculationIntent

                scf_intent = CalculationIntent(
                    structure=intent.structure, code=intent.code, task="scf",
                    xc=intent.xc, pseudo_family=intent.pseudo_family,
                    accuracy=intent.accuracy, hints=intent.hints,
                )
                ph_setup = advise_ph_setup(
                    intent.structure, analysis,
                    _cast(_Lit["fast", "balanced", "accurate"], intent.accuracy),
                    params.kpoints_grid,
                )
            result = write_qe_inputs(
                params, structure, scf_intent, output_dir=run_dir,
                kgrid_override=ph_setup.phonon_kgrid if ph_setup else None,
                conv_thr=ph_setup.scf_conv_thr if ph_setup else None,
            )
            ph_result = write_ph_inputs(
                output_dir=run_dir,
                nq=ph_setup.q_grid.nq if ph_setup else None,
                epsil=ph_setup.needs_epsil if ph_setup else False,
                tr2_ph=ph_setup.tr2_ph if ph_setup else 1e-14,
            ) if is_ph else None
        except Exception as exc:
            console.print(f"  [red]Error generating files:[/red] {exc}")
            return

    console.print()
    console.print(f"  [bold]Output:[/bold] {run_dir}")
    console.print(f"  [green]✓[/green] {result['input_file'].name}")
    if ph_result:
        console.print(f"  [green]✓[/green] {ph_result['ph_file'].name}")
    missing: list[str] = result.get("missing_pp") or []  # type: ignore[assignment]
    if missing:
        console.print(
            f"  [yellow]⚠[/yellow] pp not found: {', '.join(missing)}"
        )
    console.print()


def _prov(provenance: str) -> Text:
    return Text(provenance, style=_PROV_STYLE.get(provenance, ""))


def _coerce(value: str) -> Any:
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def _display(console, structure_path, analysis, intent, params, explain: bool) -> None:
    formula = intent.structure.composition.reduced_formula

    # ── header ──
    console.print()
    console.rule(
        f"[bold]goldilocks[/bold]  ·  [bold blue]{formula}[/bold blue]"
        f"  ·  {intent.task}  ·  {intent.accuracy}  ·  QE",
        style="blue",
    )

    # ── structure analysis ──
    console.print()
    console.print("  [bold]Structure Analysis[/bold]")
    sa = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
    sa.add_column("key", style="dim", min_width=18)
    sa.add_column("value")

    sa.add_row("Formula", formula)
    sa.add_row(
        "Space group",
        f"{analysis.space_group_symbol} ({analysis.space_group_number}), {analysis.crystal_system}",
    )
    sa.add_row("Sites", f"{analysis.n_atoms} atoms, {analysis.n_species} species")
    sa.add_row(
        "Metallicity",
        Text.assemble(
            analysis.metallicity,
            "  ",
            _prov(analysis.metallicity_source),
        ),
    )
    sa.add_row(
        "Magnetic el.",
        ", ".join(analysis.magnetic_elements) if analysis.magnetic_elements else "—",
    )
    sa.add_row(
        "Heavy el. (SOC)",
        ", ".join(analysis.heavy_elements) if analysis.heavy_elements else "—",
    )
    sa.add_row("SOC relevant", "[bold]yes[/bold]" if analysis.soc_relevant else "no")
    console.print(sa)

    for w in analysis.warnings:
        console.print(f"    [yellow]⚠[/yellow]  {w}")

    # ── parameters ──
    console.print()
    console.print("  [bold]Recommended QE Parameters[/bold]")
    pt = Table(box=box.SIMPLE_HEAD, padding=(0, 2), show_edge=False)
    pt.add_column("Parameter", style="bold", min_width=14)
    pt.add_column("Value", min_width=32)
    pt.add_column("Provenance", min_width=12)

    # spin
    spin_val = Text(params.spin_decision.treatment)
    spin_detail = f"  nspin={params.nspin}"
    if params.noncolin:
        spin_detail += ", noncolin"
    if params.lspinorb:
        spin_detail += ", lspinorb"
    spin_val.append(spin_detail, style="dim")
    pt.add_row("Spin", spin_val, _prov(params.spin_decision.provenance))

    # starting magnetisation (when set)
    if params.starting_magnetization:
        mag_pairs = "  ".join(
            f"{el}={v:.1f}" for el, v in params.starting_magnetization.items()
        )
        mag_text = Text(f"{mag_pairs}  ", style="dim")
        mag_text.append("μB", style="dim italic")
        pt.add_row("  Starting mag.", mag_text, Text(""))
        if params.angle1 is not None:
            ang_pairs = "  ".join(
                f"{el}={v:.0f}°" for el, v in params.angle1.items()
            )
            pt.add_row(
                "  Angles (θ,φ)",
                Text(f"angle1: {ang_pairs}  angle2: all 0° (FM ∥ z)", style="dim"),
                Text(""),
            )

    # smearing
    if params.smearing_decision.use_smearing:
        smear_val = Text(f"{params.smearing}  ")
        smear_val.append(
            f"{params.smearing_decision.width_ev:.4f} eV", style="dim"
        )
    else:
        smear_val = Text("fixed occupations")
    pt.add_row("Smearing", smear_val, _prov(params.smearing_decision.provenance))

    # k-mesh
    g = params.kpoints_grid
    kmesh_val = Text(f"{g[0]} × {g[1]} × {g[2]}")
    pt.add_row("K-mesh", kmesh_val, _prov(params.kpoints_decision.provenance))

    # cutoffs (Ry — what QE uses; eV in dim for reference)
    wfc_ry, rho_ry = params.ecutwfc, params.ecutrho
    cutoff_val = Text(f"{wfc_ry:.0f} / {rho_ry:.0f} Ry")
    cutoff_val.append(
        f"  ({wfc_ry * _RY_TO_EV:.0f} / {rho_ry * _RY_TO_EV:.0f} eV, wfc/rho)",
        style="dim",
    )
    pt.add_row("Cutoffs", cutoff_val, _prov(params.cutoff_decision.provenance))

    # vdW correction
    vdw = params.vdw_decision
    if vdw.use_vdw:
        vdw_val = Text(f"{vdw.method}")
        vdw_val.append(f"  ({params.vdw_corr})", style="dim")
    else:
        vdw_val = Text("none", style="dim")
    pt.add_row("vdW correction", vdw_val, _prov(vdw.provenance))

    # pseudos
    for ps in params.pseudos:
        fam_parts = ps.family.split("/")
        # PseudoDojo/0.4/PBEsol/FR/standard/upf → PseudoDojo FR 0.4
        fam_short = (
            f"{fam_parts[0]} {fam_parts[3]} {fam_parts[1]}"
            if len(fam_parts) >= 4
            else ps.family
        )
        if ps.filename:
            ps_val = Text(ps.filename)
            ps_val.append(f"  {fam_short}", style="dim")
        else:
            ps_val = Text("deferred ", style="dim italic")
            ps_val.append(f"({fam_short})", style="dim")
        pt.add_row(f"Pseudo {ps.element}", ps_val, _prov(ps.provenance))

    console.print(pt)

    # ── rationale ──
    if explain:
        console.print()
        console.print("  [bold]Rationale[/bold]")
        rt = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
        rt.add_column("key", style="dim", min_width=10)
        rt.add_column("rationale", overflow="fold", max_width=80)
        rt.add_row("spin",     escape(params.spin_decision.rationale))
        rt.add_row("smearing", escape(params.smearing_decision.rationale))
        rt.add_row("kpoints",  escape(params.kpoints_decision.rationale))
        rt.add_row("cutoffs",  escape(params.cutoff_decision.rationale))
        console.print(rt)
    else:
        console.print(
            "  [dim]Pass --explain / -e to see the full rationale for each decision.[/dim]"
        )

    console.print()
