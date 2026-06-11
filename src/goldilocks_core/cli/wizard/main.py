"""goldilocks interactive wizard — top-level menu loop."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

_VALID_CHOICES = {"0", "1", "2", "a", "b"}


def wizard() -> None:
    """Run the interactive wizard menu loop."""
    console = Console()
    console.print()

    while True:
        _print_menu(console)
        raw = Prompt.ask(
            "  Select  [dim](press Enter for default: 0 = quit)[/dim]",
            default="0",
            show_default=False,
        )
        choice = raw.strip().lower()

        if choice not in _VALID_CHOICES:
            console.print(f"  [red]Invalid choice:[/red] {choice!r}")
            console.print()
            continue

        if choice == "0":
            console.print()
            break

        elif choice == "a":
            from goldilocks_core.cli.wizard.pre_analysis import run as pre_run

            pre_run(console)

        elif choice == "b":
            from goldilocks_core.cli.wizard.pre_analysis import run_search

            run_search(console)

        elif choice == "1":
            from goldilocks_core.cli.wizard.input_kit import run as kit_run

            kit_run(console, None, mode="agnostic")

        elif choice == "2":
            from goldilocks_core.cli.wizard.input_kit import run as kit_run

            kit_run(console, None, mode="qe")

        Prompt.ask(
            "\n  [dim]Press Enter to return to menu[/dim]",
            default="",
            show_default=False,
        )
        console.print()


def _print_menu(console: Console) -> None:
    lines = (
        "  ─── [bold]Pre-Analysis Studio[/bold] ───────────────────────────────\n"
        "   [bold cyan]a)[/bold cyan]  Analyse Structure   —"
        " symmetry, magnetism, SOC, dimensionality, etc.\n"
        "   [bold cyan]b)[/bold cyan]  Search Database     —"
        " MP · MC · NOMAD · JARVIS, etc.\n"
        "\n"
        "  ─── [bold]Inputs Kit[/bold] ────────────────────────────────────────\n"
        "   [bold cyan]1)[/bold cyan]  Code-agnostic       —"
        " spin · k-mesh · SOC · dipole · DFT+U, etc.\n"
        "   [bold cyan]2)[/bold cyan]  QE Inputs           —"
        " tailored for Quantum ESPRESSO\n"
        "\n"
        "  ─── [dim]HPC Playground[/dim] ──────────────────────────────────────\n"
        "   [dim]3)  HPC Scripts         — PBS · SLURM submission scripts, etc.  (coming soon)[/dim]\n"
        "   [dim]4)  AiiDA Workflow      — job management & provenance           (coming soon)[/dim]\n"
        "\n"
        "  ─── [dim]Post-Analysis Lab[/dim] ─────────────────────────────────────\n"
        "   [dim]5)  Parse & Validate    — SCF · relax output + consistency check, etc.  (coming soon)[/dim]\n"
        "   [dim]6)  Visualise           — band structure · DOS · charge density, etc.    (coming soon)[/dim]\n"
        "\n"
        "   [bold]0)[/bold]  Quit"
    )
    console.print(
        Panel(
            lines,
            title="[bold]goldilocks[/bold]  ·  AI-powered DFT tools",
            border_style="blue",
            padding=(1, 2),
        )
    )
    console.print()
