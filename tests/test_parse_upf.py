from pathlib import Path

import pytest

from goldilocks_core.pseudo.parse_upf import parse_upf_metadata


def test_parse_upf_metadata_parses_real_pslibrary_file() -> None:
    """Parse metadata from a real PSLibrary UPF file."""
    pseudo_path = Path(
        "local_data/pseudopotentials/pslibrary/Hg.pbe-n-rrkjus_psl.1.0.0.UPF"
    )

    metadata = parse_upf_metadata(pseudo_path)

    assert metadata.element == "Hg"
    assert metadata.filename == "Hg.pbe-n-rrkjus_psl.1.0.0.UPF"
    assert metadata.pseudo_type == "USPP"
    assert metadata.functional == "PBE"
    assert metadata.relativistic == "scalar"
    assert metadata.z_valence == 12.0


def test_parse_upf_metadata_parses_real_gbrv_text_header() -> None:
    """Parse metadata from a real GBRV text-style PP_HEADER."""
    pseudo_path = Path(
        "local_data/pseudopotentials/GBRV/all_pbe_UPF_v1.5/li_pbe_v1.4.uspp.F.UPF"
    )

    metadata = parse_upf_metadata(pseudo_path)

    assert metadata.element == "Li"
    assert metadata.filename == "li_pbe_v1.4.uspp.F.UPF"
    assert metadata.pseudo_type == "USPP"
    assert metadata.functional == "PBE"
    assert metadata.relativistic == "non-relativistic"
    assert metadata.z_valence == 3.0


def test_parse_upf_metadata_raises_for_missing_file(tmp_path: Path) -> None:
    """Raise an error when the UPF file does not exist."""
    pseudo_path = tmp_path / "missing.UPF"

    with pytest.raises(FileNotFoundError):
        parse_upf_metadata(pseudo_path)


def test_parse_upf_metadata_parses_real_pslibrary_pbesol_file() -> None:
    """Parse metadata from a real PSLibrary PBESOL UPF file."""
    pseudo_path = Path(
        "local_data/pseudopotentials/pslibrary/Al.pbesol-n-kjpaw_psl.1.0.0.UPF"
    )

    metadata = parse_upf_metadata(pseudo_path)

    assert metadata.element == "Al"
    assert metadata.filename == "Al.pbesol-n-kjpaw_psl.1.0.0.UPF"
    assert metadata.pseudo_type == "PAW"
    assert metadata.functional == "PBESOL"
    assert metadata.relativistic == "scalar"
    assert metadata.z_valence == 3.0


def test_parse_upf_metadata_prefers_header_pseudo_type_over_filename_hint() -> None:
    """Prefer PP_HEADER pseudo_type over filename naming conventions."""
    pseudo_path = Path(
        "local_data/pseudopotentials/pslibrary/B.pbe-n-kjpaw_psl.0.1.UPF"
    )

    metadata = parse_upf_metadata(pseudo_path)

    assert metadata.element == "B"
    assert metadata.pseudo_type == "USPP"
    assert metadata.functional == "PBE"
    assert metadata.relativistic == "scalar"
