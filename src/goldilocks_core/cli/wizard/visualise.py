"""Visualise QE output — wizard module (DOS, bands)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.prompt import Prompt

_SUPPORTED_CODES = {"qe"}
_ALL_CODES = [
    ("qe",   "Quantum ESPRESSO",  True),
    ("vasp", "VASP",              False),
    ("cp2k", "CP2K",              False),
]


def _ask_code(console: Console) -> str:
    console.print("  DFT code:")
    for key, label, active in _ALL_CODES:
        if active:
            console.print(f"    [bold cyan]{key})[/bold cyan]  {label}")
        else:
            console.print(f"    [dim]{key})  {label}  (coming soon)[/dim]")
    return Prompt.ask("  Choose", default="qe")


def run(console: Console) -> None:
    """Interactive visualisation of DFT output files."""
    console.print()
    console.rule("[bold]Visualise[/bold]", style="blue")
    console.print()

    code = _ask_code(console)
    if code != "qe":
        console.print(f"  [yellow]Visualiser for {code.upper()} not yet available.[/yellow]")
        return
    console.print()

    dir_str = Prompt.ask("  Output directory", default=".")
    out_dir = Path(dir_str).expanduser().resolve()
    if not out_dir.exists():
        console.print(f"  [red]Error:[/red] {out_dir} does not exist.")
        return

    console.print()
    console.print("  What to plot:")
    console.print("    [bold cyan]1)[/bold cyan]  DOS  (density of states)")
    console.print("    [bold cyan]2)[/bold cyan]  Band structure")
    console.print("    [bold cyan]3)[/bold cyan]  Both")
    choice = Prompt.ask("  Choose", choices=["1", "2", "3"], default="1", show_choices=False)
    console.print()

    if choice in ("1", "3"):
        _plot_dos(console, out_dir)
    if choice in ("2", "3"):
        _plot_bands(console, out_dir)


# ---------------------------------------------------------------------------
# DOS
# ---------------------------------------------------------------------------

def _plot_dos(console: Console, out_dir: Path) -> None:
    """Find and plot a QE DOS file."""
    dos_file = _find_dos_file(out_dir)
    if dos_file is None:
        console.print("  [yellow]⚠[/yellow]  No DOS file found.")
        console.print("  [dim]Expected: *.dos, pwscf.dos, or prefix.dos[/dim]")
        console.print()
        return

    console.print(f"  [dim]Using:[/dim] {dos_file.name}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("  [red]Error:[/red] matplotlib not installed.")
        return

    try:
        data = _parse_dos_file(dos_file)
    except Exception as exc:
        console.print(f"  [red]Error parsing DOS:[/red] {exc}")
        return

    energies = data[:, 0]
    dos      = data[:, 1]

    # find Fermi energy from header comment
    fermi_ev = _read_dos_fermi(dos_file)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(energies, dos, lw=1.0, color="steelblue")
    ax.fill_between(energies, dos, alpha=0.3, color="steelblue")
    if fermi_ev is not None:
        ax.axvline(fermi_ev, color="red", lw=1.0, ls="--", label=f"Eᶠ = {fermi_ev:.3f} eV")
        ax.legend(fontsize=8)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DOS (states/eV)")
    ax.set_title("Density of States")
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    png = out_dir / "dos.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    console.print(f"  [green]✓[/green] Saved: [bold]{png}[/bold]")
    console.print()


def _find_dos_file(out_dir: Path) -> Path | None:
    for pattern in ("*.dos", "pwscf.dos", "prefix.dos"):
        hits = list(out_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def _parse_dos_file(path: Path) -> "Any":
    import numpy as np
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                rows.append([float(p) for p in parts[:2]])
            except ValueError:
                pass
    if not rows:
        raise ValueError("No data rows found in DOS file")
    return np.array(rows)


def _read_dos_fermi(path: Path) -> float | None:
    import re
    first = path.read_text(errors="replace").splitlines()[:5]
    for line in first:
        m = re.search(r"EFermi\s*=\s*([-\d.]+)", line, re.IGNORECASE)
        if m:
            return float(m.group(1))
        m = re.search(r"Fermi energy\s*=\s*([-\d.]+)", line, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------

def _plot_bands(console: Console, out_dir: Path) -> None:
    """Find and plot a QE bands.dat.gnu or bands.x output file."""
    bands_file = _find_bands_file(out_dir)
    if bands_file is None:
        console.print("  [yellow]⚠[/yellow]  No bands file found.")
        console.print("  [dim]Expected: *.dat.gnu or *.bands.dat[/dim]")
        console.print()
        return

    console.print(f"  [dim]Using:[/dim] {bands_file.name}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("  [red]Error:[/red] matplotlib not installed.")
        return

    try:
        kpts, eigs = _parse_bands_gnu(bands_file)
    except Exception as exc:
        console.print(f"  [red]Error parsing bands:[/red] {exc}")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    for band in eigs.T:
        ax.plot(kpts, band, lw=0.8, color="steelblue")
    ax.axhline(0, color="red", lw=0.8, ls="--", label="Eᶠ = 0")
    ax.set_xlabel("k-path")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Band Structure")
    ax.set_xlim(kpts[0], kpts[-1])
    ax.legend(fontsize=8)
    fig.tight_layout()

    png = out_dir / "bands.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    console.print(f"  [green]✓[/green] Saved: [bold]{png}[/bold]")
    console.print()


def _find_bands_file(out_dir: Path) -> Path | None:
    for pattern in ("*.dat.gnu", "*.bands.dat", "bands.dat.gnu"):
        hits = list(out_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def _parse_bands_gnu(path: Path) -> "tuple[Any, Any]":
    """Parse a QE bands.x *.dat.gnu file.

    Format: blocks separated by blank lines, each block is one band:
      kpath_coord  energy(eV)
    """
    import numpy as np

    text = path.read_text(errors="replace")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    bands: list[list[float]] = []
    kpts_ref: list[float] = []

    for block in blocks:
        rows = []
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                rows.append((float(parts[0]), float(parts[1])))
        if rows:
            if not kpts_ref:
                kpts_ref = [r[0] for r in rows]
            bands.append([r[1] for r in rows])

    if not bands:
        raise ValueError("No band data found")

    return np.array(kpts_ref), np.array(bands).T


