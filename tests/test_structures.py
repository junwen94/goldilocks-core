import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.helpers.structures import load_structure


def test_load_structure_returns_structure_input() -> None:
    """Return the input unchanged when it is already a Structure."""
    structure = Structure(
        lattice=Lattice.cubic(3.5),
        species=["Si"],
        coords=[[0.0, 0.0, 0.0]],
    )

    loaded = load_structure(structure)

    assert loaded is structure


def test_load_structure_raises_for_missing_file() -> None:
    """Raise FileNotFoundError when the structure file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_structure("missing_structure.cif")


def test_load_structure_raises_for_unsupported_xyz(tmp_path) -> None:
    """Raise ValueError for unsupported XYZ structure files."""
    xyz_file = tmp_path / "test.xyz"
    xyz_file.write_text("1\ncomment\nH 0.0 0.0 0.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported structure file format"):
        load_structure(xyz_file)
