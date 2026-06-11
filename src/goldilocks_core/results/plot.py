"""Band structure and DOS visualisation using pymatgen plotters.

All functions save PNG files and return the output path.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def plot_bands(
    data_file_xml: str | Path,
    output: str | Path | None = None,
    zero_to_fermi: bool = True,
    ylim: tuple[float, float] = (-6.0, 6.0),
) -> Path:
    """Generate a band structure PNG from a QE pw.x XML output file.

    The XML file is ``<outdir>/<prefix>.save/data-file-schema.xml``.

    Args:
        data_file_xml: Path to ``data-file-schema.xml``.
        output: Output PNG path.  Defaults to ``bands.png`` alongside the XML.
        zero_to_fermi: Shift eigenvalues so E_F = 0.
        ylim: Energy window in eV (relative to E_F when zero_to_fermi is True).

    Returns:
        Path to the saved PNG.
    """
    from pymatgen.electronic_structure.plotter import BSPlotter
    from pymatgen.io.espresso.pwxml import PwXmlDocument

    xml_path = Path(data_file_xml)
    out_path = Path(output) if output else xml_path.parent / "bands.png"

    doc = PwXmlDocument.from_file(xml_path)
    bs = doc.get_band_structure(kpoints_filename=None, efermi=None, line_mode=True)

    plotter = BSPlotter(bs)
    ax = plotter.get_plot(zero_to_efermi=zero_to_fermi, ylim=ylim)
    fig = ax.get_figure()
    assert fig is not None
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path


def plot_dos(
    dos_file: str | Path,
    output: str | Path | None = None,
    xlim: tuple[float, float] = (-6.0, 6.0),
    sigma: float = 0.05,
) -> Path:
    """Generate a total DOS PNG from a QE pw.x XML output file.

    Args:
        dos_file: Path to ``data-file-schema.xml`` (same as bands — DOS is
            extracted from the same XML).
        output: Output PNG path.  Defaults to ``dos.png`` alongside the XML.
        xlim: Energy window in eV relative to E_F.
        sigma: Gaussian smearing width in eV applied for display.

    Returns:
        Path to the saved PNG.
    """
    from pymatgen.electronic_structure.plotter import DosPlotter
    from pymatgen.io.espresso.pwxml import PwXmlDocument

    xml_path = Path(dos_file)
    out_path = Path(output) if output else xml_path.parent / "dos.png"

    doc = PwXmlDocument.from_file(xml_path)
    cdos = doc.complete_dos

    plotter = DosPlotter(sigma=sigma)
    plotter.add_dos("Total DOS", cdos)
    ax = plotter.get_plot(xlim=xlim)
    import matplotlib.figure as _mfig
    fig = ax.get_figure()
    assert isinstance(fig, _mfig.Figure)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path
