"""Shared, plain (non-fixture) test helpers.

Not named ``conftest.py``: more than one ``conftest.py`` exists in this
tree (``tests/conftest.py``, ``tests/server/conftest.py``), and pytest's
own rootless import mode gives every file literally named ``conftest.py``
the same bare top-level module name -- a test module doing
``from conftest import X`` resolves against whichever ``conftest.py``
Python's import cache happens to have already bound to that name, which
is order-dependent once more than one exists (confirmed the hard way,
v2 epic 9, #9: importable in isolation, ``ImportError`` once the full
suite collects ``tests/server/`` too). Fixtures still belong in
``conftest.py`` (pytest's own fixture-discovery is directory-scoped, not
Python's plain import system, so that part is unaffected); plain helper
*functions* a test calls directly belong here instead, under a name
unique across the whole tree.

``run_cli``/``default_profile_installed``/``REAL_ASSET_ROOT`` moved
here from ``tests/integration/test_cli_v2.py`` (v2 epic 9, #9) once a
second file (the five-scenario acceptance suite) needed to spawn the
real CLI subprocess too -- the ``real_assets`` fixture that wraps
``default_profile_installed`` stays in ``tests/conftest.py`` for the
same fixtures-are-directory-scoped reason.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pymatgen.core import Lattice, Structure

from goldilocks_core.analysis.geometry import GeometryFacts
from goldilocks_core.assets.records import AssetFile, AssetSpec
from goldilocks_core.assets.runtime import statuses
from goldilocks_core.assets.store import AssetStore, asset_root

if TYPE_CHECKING:
    from goldilocks_core.analysis.geometry import Dimensionality, LatticeAxis
    from goldilocks_core.assets.pseudopotentials.registry import PseudoTable

REAL_ASSET_ROOT = asset_root()
"""The real, developer-machine asset root (not the per-test isolated
one ``conftest.py``'s ``isolated_default_asset_root`` autouse fixture
sets up) -- shared by every test that needs a genuinely-installed
pseudopotential table, not a synthetic fixture."""


def default_profile_installed() -> bool:
    store = AssetStore(REAL_ASSET_ROOT)
    return all(state == "installed" for _, _, state in statuses("default", store=store))


def run_cli(
    *arguments: str, cwd: Path | None = None, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    """Spawn the real ``goldilocks`` CLI entry point as a subprocess --
    shared by every integration test that exercises argument parsing
    end to end (v2 epic 9, #9), not just the internal core API."""
    return subprocess.run(
        [sys.executable, "-m", "goldilocks_core.cli.core", *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def strict_json(document: str | bytes) -> Any:
    """``json.loads`` that rejects the non-standard ``Infinity``,
    ``-Infinity`` and ``NaN`` tokens Python's ``json.dumps`` writes by
    default (#137) -- shared by every test that checks a JSON document a
    transport or the bundle writes."""

    def reject(constant: str) -> None:
        raise ValueError(f"{constant} is not valid JSON")

    return json.loads(document, parse_constant=reject)


def fixture_upf(element: str, z_valence: float) -> bytes:
    """A minimal SSSP-shaped UPF file for ``element``: only the
    ``PP_HEADER`` attributes the importer reads and the projector counts
    every installed UPF carries, which the job's memory estimate reads
    (#148)."""
    return (
        b'<UPF version="2.0.1">\n'
        + (
            f'<PP_HEADER element="{element}" pseudo_type="NC" functional="PBEsol" '
            f'relativistic="scalar" z_valence="{z_valence}" number_of_wfc="2" '
            'number_of_proj="4" l_max="1"/>\n'
        ).encode()
        + b"</UPF>\n"
    )


SSSP_FIXTURE_UPF = fixture_upf("Si", 4.0)


def sssp_fixture_archive(path: Path, members: dict[str, bytes]) -> None:
    """Write a ``.tar.gz`` containing ``members`` -- shared by every test
    that builds a fully-offline (``file://``, no network) SSSP-shaped
    pseudopotential asset."""
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


def sssp_fixture_table_spec(
    tmp_path: Path, z_valences: dict[str, float] | None = None
) -> tuple[AssetSpec, PseudoTable]:
    """Build one real, fully-offline SSSP-shaped pseudopotential asset
    (file:// sources, no network) -- shared between
    ``tests/unit/test_service_v2.py`` and ``tests/unit/test_service_dos.py``
    (v2 epic 9, #9), matching the pattern already used by
    ``tests/unit/test_pseudo_importers.py``'s ``install_sssp_fixture``.

    ``z_valences`` maps every element the table covers to its UPF
    ``z_valence``. The default is Si alone, whose file is
    ``SSSP_FIXTURE_UPF`` as this module holds it at call time, so a test
    can monkeypatch it to vary the header."""
    from goldilocks_core.assets.pseudopotentials.registry import PseudoTable

    upf_by_element = (
        {
            element: fixture_upf(element, z_valence)
            for element, z_valence in z_valences.items()
        }
        if z_valences
        else {"Si": SSSP_FIXTURE_UPF}
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    upfs = tmp_path / "table.tar.gz"
    sidecar = tmp_path / "table.json"
    licence = tmp_path / "LICENSE.txt"
    sssp_fixture_archive(
        upfs,
        {f"nested/{element}.upf": upf for element, upf in upf_by_element.items()},
    )
    sidecar.write_text(
        json.dumps(
            {
                element: {
                    "filename": f"{element}.upf",
                    "md5": hashlib.md5(upf).hexdigest(),
                    "cutoff_wfc": 30.0,
                    "cutoff_rho": 120.0,
                    "pseudopotential": f"{element} fixture",
                }
                for element, upf in upf_by_element.items()
            }
        )
    )
    licence.write_text("SSSP fixture licence\n")
    spec = AssetSpec(
        "pseudopotentials/sssp-fixture",
        "1",
        (
            AssetFile("pseudopotentials", "source/table.tar.gz", upfs.as_uri()),
            AssetFile("metadata", "source/table.json", sidecar.as_uri()),
            AssetFile("licence", "source/LICENSE.txt", licence.as_uri()),
        ),
    )
    registry_table = PseudoTable(
        id="sssp-fixture",
        provider="sssp",
        upstream_table="fixture",
        version="1",
        functional="PBEsol",
        relativistic="scalar",
        accuracy="efficiency",
        licence="fixture licence",
        citation="Synthetic SSSP fixture, cite me.",
        elements=tuple(sorted(upf_by_element)),
        asset=spec,
        default=True,
    )
    return spec, registry_table


_MC2D_60196AFB_PHOSPHORENE_CIF = """\
data_P
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   3.30546570
_cell_length_b   4.61434267
_cell_length_c   22.18331037
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   1
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  P  P0  1  0.50000000  0.91158132  0.45243739  1
  P  P1  1  0.50000000  0.08841868  0.54756261  1
  P  P2  1  1.00000000  0.58841862  0.45243739  1
  P  P3  1  1.00000000  0.41158138  0.54756261  1
"""


def phosphorene() -> Structure:
    """Phosphorene monolayer, MC2D mc2d-60196afb: 4 P atoms, P-P bonds of
    2.22-2.26 A, vacuum along c (22.2 A cell)."""
    return Structure.from_str(_MC2D_60196AFB_PHOSPHORENE_CIF, fmt="cif")


_MC3D_6516_CARBON_PAIR_CIF = """\
data_0

loop_
  _atom_site_label
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  _atom_site_type_symbol
         C1  0.33333333333333337  0.6666666666666667  0.24999999999854772  C
         C2  0.6666666666666667  0.33333333333333337  0.7500000000014405  C
_cell_angle_alpha                       90.0
_cell_angle_beta                        90.0
_cell_angle_gamma                       119.99999999987348
_cell_length_a                          10.559999907643
_cell_length_b                          10.55999990768339
_cell_length_c                          17.179999849807
loop_
  _symmetry_equiv_pos_as_xyz
         'x, y, z'
_symmetry_int_tables_number             1
_symmetry_space_group_name_H-M          'P 1'
"""


def carbon_pair_in_a_box() -> Structure:
    """MC3D mc3d-6516, an isolated-atom artifact: two C atoms 10.5 A apart
    in a 10.56 x 10.56 x 17.18 A hexagonal cell, so no atom has a neighbour
    within 10 A."""
    return Structure.from_str(_MC3D_6516_CARBON_PAIR_CIF, fmt="cif")


def sodium_chloride() -> Structure:
    """Rock salt, Fm-3m, a = 5.64 A: Na-Cl 2.82 A."""
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def carbon_dioxide() -> Structure:
    """Dry ice, Pa-3, a = 5.62 A: C=O 1.15 A inside each molecule, 3.1 A
    between molecules."""
    return Structure.from_spacegroup(
        "Pa-3",
        Lattice.cubic(5.62),
        ["C", "O"],
        [[0, 0, 0], [0.1185, 0.1185, 0.1185]],
    )


_MC3D_4573_WATER_ICE_CIF = """\
data_0

loop_
  _atom_site_label
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  _atom_site_type_symbol
         H1  0.16030715231323672  0.8396928476867637  0.24999999999239195  H
         H2  0.8396928476867634  0.16030715231323656  0.7499999999923843  H
         H3  0.005979649911938667  0.00597964991193467  9.715845373787842e-35  H
         H4  0.9940203500646564  0.9940203500646564  0.5000000000000077  H
         H5  0.9674206607569137  0.42584589732319206  0.9837891779390359  H
         H6  0.5741541026767801  0.03257933924305845  0.5162108220609611  H
         H7  0.032579339228914254  0.5741541026909709  0.4837891779390389  H
         H8  0.4258458973090057  0.9674206607710624  0.01621082206096412  H
         O1  0.1612316690525443  0.8410138827825016  0.06589485616831037  O
         O2  0.1589861172174753  0.8387683309474325  0.43410514383168813  O
         O3  0.8387683309332875  0.158986117231637  0.5658948561683119  O
         O4  0.8410138827683481  0.16123166906669772  0.9341051438316957  O
_cell_angle_alpha                       90.0
_cell_angle_beta                        90.0
_cell_angle_gamma                       117.95814914432724
_cell_length_a                          4.126434642797939
_cell_length_b                          4.126434642797939
_cell_length_c                          6.5883867784937
loop_
  _symmetry_equiv_pos_as_xyz
         'x, y, z'
_symmetry_int_tables_number             1
_symmetry_space_group_name_H-M          'P 1'
"""


def water_ice() -> Structure:
    """H2O ice, MC3D mc3d-4573: 4 hydrogen-bonded molecules in a 12-atom
    cell (shortest O-O 2.43 A), so no atom-free slab is wider than 1.21 A."""
    return Structure.from_str(_MC3D_4573_WATER_ICE_CIF, fmt="cif")


def geometry_facts(
    dimensionality: Dimensionality, vacuum_axes: tuple[LatticeAxis, ...] = ()
) -> GeometryFacts:
    """``GeometryFacts`` for advisor and check tests: 20 A of vacuum along
    each axis in ``vacuum_axes`` and a 2 A gap along the others."""
    return GeometryFacts(
        dimensionality=dimensionality,
        low_dimensional=dimensionality != "3d",
        vacuum_axes=vacuum_axes,
        vacuum_widths_angstrom=tuple(
            20.0 if axis in vacuum_axes else 2.0 for axis in ("a", "b", "c")
        ),
    )
