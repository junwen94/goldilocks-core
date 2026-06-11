"""Parse & Validate QE output files — wizard module."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from rich import box
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text


@dataclass
class _PWResult:
    path: Path
    calc_type: str = "scf"           # scf / relax / vc-relax / bands / …
    converged: bool | None = None
    total_energy_ry: float | None = None
    fermi_energy_ev: float | None = None
    total_mag: float | None = None
    n_iter: int | None = None
    max_force_ry_au: float | None = None
    warnings: list[str] = field(default_factory=list)


def run(console: Console) -> None:
    """Parse DFT output files in a directory and display a summary."""
    console.print()
    console.rule("[bold]Parse & Validate[/bold]", style="blue")
    console.print()

    code = _ask_code(console)
    if code != "qe":
        console.print(f"  [yellow]Parser for {code.upper()} not yet available.[/yellow]")
        return
    console.print()

    dir_str = Prompt.ask("  Output directory", default=".")
    out_dir = Path(dir_str).expanduser().resolve()
    if not out_dir.exists():
        console.print(f"  [red]Error:[/red] {out_dir} does not exist.")
        return

    # find .out files (heuristic: QE pw.x output usually contains "Program PWSCF")
    candidates = list(out_dir.glob("*.out")) + list(out_dir.glob("**/*.out"))
    pw_files = [f for f in candidates if _looks_like_pw_output(f)]

    if not pw_files:
        console.print("  [yellow]No pw.x output files found.[/yellow]")
        console.print("  [dim]Looking for *.out files containing QE PWSCF header.[/dim]")
        return

    console.print(f"  [dim]Found {len(pw_files)} output file(s).[/dim]")
    console.print()

    results = [_parse_pw_output(f) for f in pw_files]
    _display_results(console, results)

    # consistency check against manifest if present
    manifest_path = out_dir / "goldilocks_manifest.json"
    if manifest_path.exists():
        _check_manifest(console, manifest_path, results)


# ---------------------------------------------------------------------------
# Code selection
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_RY_TO_EV = 13.605693122994

def _looks_like_pw_output(path: Path) -> bool:
    try:
        head = path.read_text(errors="replace")[:2000]
        return "Program PWSCF" in head or "PWSCF" in head
    except OSError:
        return False


def _parse_pw_output(path: Path) -> _PWResult:
    res = _PWResult(path=path)
    try:
        text = path.read_text(errors="replace")
    except OSError:
        res.warnings.append("Could not read file.")
        return res

    # calc type
    m = re.search(r"calculation\s*=\s*'(\S+)'", text)
    if m:
        res.calc_type = m.group(1).strip("'")

    # convergence (SCF)
    if re.search(r"convergence has been achieved in\s+(\d+)\s+iteration", text):
        res.converged = True
        m = re.search(r"convergence has been achieved in\s+(\d+)\s+iteration", text)
        if m:
            res.n_iter = int(m.group(1))
    elif re.search(r"convergence NOT achieved", text, re.IGNORECASE):
        res.converged = False
        res.warnings.append("SCF did not converge!")
    # relax convergence
    if re.search(r"End of (BFGS|Damped dynamics) Geometry Optimization", text, re.IGNORECASE):
        res.converged = True

    # total energy — the converged value has "!" prefix
    m = re.search(r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry", text)
    if m:
        res.total_energy_ry = float(m.group(1))
    else:
        # fallback: last occurrence without !
        matches = re.findall(r"total energy\s+=\s+([-\d.]+)\s+Ry", text)
        if matches:
            res.total_energy_ry = float(matches[-1])

    # Fermi energy
    m = re.search(r"the Fermi energy is\s+([-\d.]+)\s+ev", text, re.IGNORECASE)
    if m:
        res.fermi_energy_ev = float(m.group(1))

    # total magnetization
    m = re.search(r"total magnetization\s+=\s+([-\d.]+)\s+Bohr", text, re.IGNORECASE)
    if m:
        res.total_mag = float(m.group(1))

    # max force (relax)
    m = re.search(r"Total force\s+=\s+([\d.]+)", text)
    if m:
        res.max_force_ry_au = float(m.group(1))

    return res


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _display_results(console: Console, results: list[_PWResult]) -> None:
    t = Table(box=box.SIMPLE_HEAD, padding=(0, 2), show_edge=False)
    t.add_column("File", style="bold", min_width=20)
    t.add_column("Task", min_width=8)
    t.add_column("Converged", min_width=10)
    t.add_column("Energy (eV)", min_width=14)
    t.add_column("Fermi (eV)", min_width=10)
    t.add_column("Mag (μB)", min_width=9)
    t.add_column("N iter", min_width=7)

    for r in results:
        conv_text = (
            Text("yes", style="green") if r.converged is True
            else Text("NO", style="bold red") if r.converged is False
            else Text("—", style="dim")
        )
        energy_str = (
            f"{r.total_energy_ry * _RY_TO_EV:.4f}" if r.total_energy_ry is not None else "—"
        )
        fermi_str = f"{r.fermi_energy_ev:.4f}" if r.fermi_energy_ev is not None else "—"
        mag_str   = f"{r.total_mag:.2f}" if r.total_mag is not None else "—"
        iter_str  = str(r.n_iter) if r.n_iter is not None else "—"

        t.add_row(
            r.path.name,
            r.calc_type,
            conv_text,
            energy_str,
            fermi_str,
            mag_str,
            iter_str,
        )

    console.print(t)

    for r in results:
        for w in r.warnings:
            console.print(f"  [yellow]⚠[/yellow]  {r.path.name}: {w}")

    if any(r.max_force_ry_au is not None for r in results):
        console.print()
        for r in results:
            if r.max_force_ry_au is not None:
                console.print(
                    f"  [dim]Max force ({r.path.name}):[/dim] "
                    f"{r.max_force_ry_au:.6f} Ry/au"
                )

    console.print()


def _check_manifest(console: Console, manifest_path: Path, results: list[_PWResult]) -> None:
    import json

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return

    analysis = manifest.get("structure_analysis", {})
    metallicity = analysis.get("metallicity", "unknown")
    console.print("  [bold]Consistency check[/bold]  [dim](vs goldilocks_manifest.json)[/dim]")
    console.print()

    for r in results:
        if r.total_energy_ry is None:
            continue
        issues: list[str] = []

        # metallic but no Fermi energy → suspicious
        if metallicity in ("metallic", "likely_metallic") and r.fermi_energy_ev is None:
            issues.append("metallicity=metallic but Fermi energy not found in output")

        # non-magnetic recommendation but large magnetization
        spin_treatment = manifest.get("parameter_set", {}).get("spin_treatment", "")
        if spin_treatment == "non_magnetic" and r.total_mag is not None and abs(r.total_mag) > 0.5:
            issues.append(
                f"non_magnetic recommended but |total_mag|={abs(r.total_mag):.2f} μB — "
                "consider re-running with collinear spin"
            )

        if issues:
            for issue in issues:
                console.print(f"  [yellow]⚠[/yellow]  {r.path.name}: {issue}")
        else:
            console.print(f"  [green]✓[/green]  {r.path.name}: no consistency issues")

    console.print()
