"""Wizard step — Structure analysis and standalone database search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from goldilocks_core.cli.wizard._context import WizardContext


def run(console: Console) -> WizardContext | None:
    """Load a local structure file, analyse it, and display results.

    Returns a WizardContext with structure + analysis populated.
    Task, accuracy, and hints are left at defaults — Input Kit collects them.
    Returns None if the user aborts.
    """
    console.print()

    # --- structure path --------------------------------------------------
    structure = None
    path: Path | None = None
    while True:
        raw = Prompt.ask("  [bold]Structure file[/bold] (CIF, POSCAR, XSF, …)")
        p = Path(raw).expanduser()
        if not p.exists():
            console.print(f"  [red]File not found:[/red] {p}")
            continue
        try:
            from goldilocks_core.io.structures import load_structure

            with console.status("  Loading…", spinner="dots"):
                structure = load_structure(p)
            path = p
            break
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]Could not parse structure:[/red] {exc}")

    assert structure is not None and path is not None  # type-narrowing

    # --- analyse ---------------------------------------------------------
    from goldilocks_core.analyse.structure import analyze_structure
    from goldilocks_core.ml.loader import try_load_magnetic_classifier

    mag_classifier = try_load_magnetic_classifier()

    with console.status(
        "  Analysing"
        + (" (magnetic ML active)…" if mag_classifier is not None else "…"),
        spinner="dots",
    ):
        analysis = analyze_structure(structure, magnetic_classifier=mag_classifier)

    _display_analysis(console, analysis, structure)

    return WizardContext(
        structure_path=path,
        structure=structure,
        analysis=analysis,
    )


def run_search(console: Console) -> None:
    """Standalone database search — menu option 3."""
    console.print()
    _search_database(console)


# ---------------------------------------------------------------------------
# Database search
# ---------------------------------------------------------------------------

def _search_database(console: Console) -> None:
    """Ask the user for a formula or structure file, query databases, show results."""
    from goldilocks_core.io.db_search import normalise_formula, search_databases

    console.print()
    console.print("  [bold]Search by[/bold]")
    console.print("    [cyan]1)[/cyan] Formula  (e.g. Fe, Fe2O3)")
    console.print("    [cyan]2)[/cyan] Structure file  (formula extracted automatically)")
    mode = Prompt.ask("  Select", choices=["1", "2"], default="1")

    if mode == "1":
        raw_formula = Prompt.ask("  Formula").strip()
        if not raw_formula:
            return
        formula = normalise_formula(raw_formula)
    else:
        raw = Prompt.ask("  Structure file path").strip()
        p = Path(raw).expanduser()
        if not p.exists():
            console.print(f"  [red]File not found:[/red] {p}")
            return
        try:
            from goldilocks_core.io.structures import load_structure
            with console.status("  Reading structure…", spinner="dots"):
                s = load_structure(p)
            formula = normalise_formula(s.composition.reduced_formula)
            console.print(f"  Extracted formula: [bold]{formula}[/bold]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]Could not read structure:[/red] {exc}")
            return

    with console.status(
        f"  Searching databases for [bold]{formula}[/bold]…", spinner="dots"
    ):
        results, errors = search_databases(formula)

    for source, err in errors.items():
        console.print(f"    [yellow]⚠[/yellow]  {source}: {err}")

    if not results:
        console.print(f"  No entries found for [bold]{formula}[/bold].")
        return

    _display_results(console, formula, results)


def _display_results(
    console: Console, formula: str, results: list[Any]
) -> None:
    console.print()
    console.rule(
        f"[bold]Database results[/bold]  ·  [bold blue]{formula}[/bold blue]"
        f"  ·  {len(results)} entries",
        style="blue",
    )

    t = Table(box=box.SIMPLE_HEAD, padding=(0, 2), show_edge=False)
    t.add_column("Source",      min_width=16)
    t.add_column("Formula",     min_width=10)
    t.add_column("Space group", min_width=12)
    t.add_column("Entry ID",    min_width=14)
    t.add_column("URL")

    for r in results:
        url = r.url or ""
        t.add_row(
            r.source,
            r.formula or "—",
            r.spacegroup or "—",
            r.entry_id or "—",
            Text(url, style=f"link {url}") if url else Text("—", style="dim"),
        )

    console.print(t)


# ---------------------------------------------------------------------------
# Analysis display
# ---------------------------------------------------------------------------

def _display_analysis(console: Console, analysis: Any, structure: Any) -> None:
    formula = structure.composition.reduced_formula
    console.print()
    console.rule(
        f"[bold]Structure Analysis[/bold]  ·  [bold blue]{formula}[/bold blue]",
        style="blue",
    )

    t = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
    t.add_column("key", style="dim", min_width=18)
    t.add_column("value")

    t.add_row("Formula", formula)
    t.add_row(
        "Space group",
        f"{analysis.space_group_symbol} ({analysis.space_group_number}),"
        f" {analysis.crystal_system}  ·  {analysis.point_group}",
    )
    t.add_row(
        "Symmetry",
        ("centrosymmetric" if analysis.has_inversion_symmetry else "non-centrosymmetric")
        + ("  [dim](polar)[/dim]" if analysis.is_polar else ""),
    )
    t.add_row("Sites", f"{analysis.n_atoms} atoms, {analysis.n_species} species")
    t.add_row(
        "Periodicity",
        f"{analysis.dimensionality}  ·  {analysis.system_type}"
        + ("  [dim](vacuum)[/dim]" if analysis.has_vacuum else ""),
    )
    t.add_row(
        "Metallicity",
        Text.assemble(
            analysis.metallicity,
            "  ",
            Text(analysis.metallicity_source, style="dim"),
        ),
    )
    mag_src = f"  [dim]({analysis.magnetic_source})[/dim]" if analysis.magnetic_source else ""
    t.add_row(
        "Magnetic el.",
        (", ".join(analysis.magnetic_elements) + mag_src)
        if analysis.magnetic_elements else "—",
    )
    t.add_row(
        "Heavy el. (SOC)",
        ", ".join(analysis.heavy_elements) if analysis.heavy_elements else "—",
    )
    t.add_row("SOC relevant", "[bold]yes[/bold]" if analysis.soc_relevant else "no")
    if analysis.has_d_electrons or analysis.has_f_electrons:
        flags = []
        if analysis.has_d_electrons:
            flags.append("d-electrons")
        if analysis.has_f_electrons:
            flags.append("f-electrons  [yellow](DFT+U recommended)[/yellow]")
        t.add_row("Correlated el.", "  ".join(flags))
    if analysis.has_partial_occupancy:
        sites = ", ".join(str(i) for i in analysis.disordered_sites[:6])
        t.add_row(
            "Disorder",
            f"[yellow]partial occupancy[/yellow]  [dim]sites: {sites}[/dim]",
        )

    console.print(t)

    for w in analysis.warnings:
        console.print(f"    [yellow]⚠[/yellow]  {w}")
