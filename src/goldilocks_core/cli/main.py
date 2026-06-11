"""goldilocks CLI entry point."""

from __future__ import annotations

import typer

from goldilocks_core.cli.commands.input import run as _input_run

app = typer.Typer(
    name="gl",
    help="goldilocks — AI-powered DFT parameter assistant.",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    """goldilocks — AI-powered DFT parameter assistant.

    Run without arguments to enter the interactive wizard.
    """
    if ctx.invoked_subcommand is None:
        from goldilocks_core.cli.wizard.main import wizard

        wizard()


@app.command("wizard")
def _wizard_cmd() -> None:
    """Enter the interactive wizard (same as running gl with no arguments)."""
    from goldilocks_core.cli.wizard.main import wizard

    wizard()


app.command("input", help="Recommend QE input parameters for a structure file.")(_input_run)


def main() -> None:
    app()
