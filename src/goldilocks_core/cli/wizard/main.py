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
        "  ─── [bold]Pre-Analysis[/bold] ──────────────────────────────────────\n"
        "   [bold cyan]a)[/bold cyan]  Analyse Structure   —"
        " symmetry, magnetism, SOC, dimensionality\n"
        "   [bold cyan]b)[/bold cyan]  Database Search     —"
        " search MP · MC · NOMAD · JARVIS\n"
        "\n"
        "  ─── [bold]Input Kit[/bold] ──────────────────────────────────────────\n"
        "   [bold cyan]1)[/bold cyan]  Code-agnostic       —"
        " spin · k-mesh · SOC · dipole · DFT+U\n"
        "   [bold cyan]2)[/bold cyan]  QE Inputs           —"
        " Quantum ESPRESSO parameters\n"
        "\n"
        "  ─── [dim]Submit Playground[/dim]   [dim]  HPC scripts & AiiDA  (coming soon)[/dim]\n"
        "  ─── [dim]Results Lab[/dim]         [dim]  parse, check & visualise  (coming soon)[/dim]\n"
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
