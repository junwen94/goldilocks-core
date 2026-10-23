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
import math
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from pymatgen.core import Lattice, Structure

from goldilocks_core.advisors.magnetic_config import (
    MagneticConfigFacts,
    MagneticConfigHumanInput,
    OrderingRecord,
    magnetic_cell,
    magnetic_config,
)
from goldilocks_core.analysis.geometry import GeometryFacts, SlabGeometry
from goldilocks_core.assets.records import AssetFile, AssetSpec
from goldilocks_core.assets.runtime import statuses
from goldilocks_core.assets.store import AssetStore, asset_root
from goldilocks_core.resolution import Provenance, Resolved

if TYPE_CHECKING:
    from goldilocks_core.analysis.geometry import Dimensionality, LatticeAxis
    from goldilocks_core.analysis.is_magnetic import Magnetism
    from goldilocks_core.assets.pseudopotentials.registry import PseudoTable
    from goldilocks_core.resolution import FieldState
    from goldilocks_core.types import RelativisticTreatment

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


def magnetic_decision(
    structure: Structure,
    is_magnetic: FieldState[Magnetism],
    needs_soc: FieldState[bool] | None = None,
    z_valences: dict[str, float] | None = None,
    human: MagneticConfigHumanInput | None = None,
    *,
    soc_table: FieldState[PseudoTable] | None = None,
) -> FieldState[MagneticConfigFacts]:
    """``magnetic_cell`` then ``magnetic_config``, the two steps
    ``service/_magnetic.py`` runs around pseudopotential selection -- for
    advisor-level tests that need one structure's whole magnetic
    decision (#128)."""
    cell = magnetic_cell(structure, is_magnetic, human)
    return magnetic_config(cell, needs_soc, soc_table, z_valences, human)


def strict_json(document: str | bytes) -> Any:
    """``json.loads`` that rejects the non-standard ``Infinity``,
    ``-Infinity`` and ``NaN`` tokens Python's ``json.dumps`` writes by
    default (#137) -- shared by every test that checks a JSON document a
    transport or the bundle writes."""

    def reject(constant: str) -> None:
        raise ValueError(f"{constant} is not valid JSON")

    return json.loads(document, parse_constant=reject)


def fixture_upf(
    element: str, z_valence: float, relativistic: RelativisticTreatment = "scalar"
) -> bytes:
    """A minimal SSSP-shaped UPF file for ``element``: only the
    ``PP_HEADER`` attributes the importer reads and the projector counts
    every installed UPF carries, which the job's memory estimate reads
    (#148)."""
    return (
        b'<UPF version="2.0.1">\n'
        + (
            f'<PP_HEADER element="{element}" pseudo_type="NC" functional="PBEsol" '
            f'relativistic="{relativistic}" z_valence="{z_valence}" '
            'number_of_wfc="2" number_of_proj="4" l_max="1"/>\n'
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
    tmp_path: Path,
    z_valences: dict[str, float] | None = None,
    relativistic: RelativisticTreatment = "scalar",
) -> tuple[AssetSpec, PseudoTable]:
    """Build one real, fully-offline SSSP-shaped pseudopotential asset
    (file:// sources, no network) -- shared between
    ``tests/unit/test_service_v2.py`` and ``tests/unit/test_service_dos.py``
    (v2 epic 9, #9), matching the pattern already used by
    ``tests/unit/test_pseudo_importers.py``'s ``install_sssp_fixture``.

    ``z_valences`` maps every element the table covers to its UPF
    ``z_valence``. The default is Si alone, whose file is
    ``SSSP_FIXTURE_UPF`` as this module holds it at call time, so a test
    can monkeypatch it to vary the header. ``relativistic="full"`` builds
    the fully relativistic table a spin-orbit run selects (it needs
    ``z_valences``), named ``sssp-fixture-fr`` so it installs next to the
    scalar one."""
    from goldilocks_core.assets.pseudopotentials.registry import PseudoTable

    table_id = "sssp-fixture" if relativistic == "scalar" else "sssp-fixture-fr"
    upf_by_element = (
        {
            element: fixture_upf(element, z_valence, relativistic)
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
        f"pseudopotentials/{table_id}",
        "1",
        (
            AssetFile("pseudopotentials", "source/table.tar.gz", upfs.as_uri()),
            AssetFile("metadata", "source/table.json", sidecar.as_uri()),
            AssetFile("licence", "source/LICENSE.txt", licence.as_uri()),
        ),
    )
    registry_table = PseudoTable(
        id=table_id,
        provider="sssp",
        upstream_table="fixture",
        version="1",
        functional="PBEsol",
        relativistic=relativistic,
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


def silicon_dimer_in_a_box() -> Structure:
    """An Si2 dimer (bond 2.25 A, along [111]) in a 12 A cubic box: 10.7 A
    of vacuum along every lattice vector. Si is the one element of
    ``sssp_fixture_table_spec``'s default table, so the dimer runs through
    ``generate()`` without installed pseudopotentials."""
    offset = 2.25 / (12.0 * 3**0.5)
    return Structure(
        Lattice.cubic(12.0), ["Si", "Si"], [[0, 0, 0], [offset, offset, offset]]
    )


def sodium_chloride() -> Structure:
    """Rock salt, Fm-3m, a = 5.64 A: Na-Cl 2.82 A."""
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def graphite() -> Structure:
    """Bernal graphite, C on the 2b and 2c sites of P6_3/mmc, a = 2.46 A,
    c = 6.71 A: C-C 1.42 A in the sheets, 3.36 A between them."""
    return Structure.from_spacegroup(
        "P6_3/mmc",
        Lattice.hexagonal(2.46, 6.71),
        ["C", "C"],
        [[0, 0, 0.25], [1 / 3, 2 / 3, 0.25]],
    )


def molybdenum_disulfide() -> Structure:
    """2H-MoS2, P6_3/mmc, a = 3.16 A, c = 12.29 A: S-Mo-S sheets stacked
    across a van der Waals gap."""
    return Structure.from_spacegroup(
        "P6_3/mmc",
        Lattice.hexagonal(3.16, 12.29),
        ["Mo", "S"],
        [[1 / 3, 2 / 3, 0.25], [1 / 3, 2 / 3, 0.621]],
    )


def rhombohedral_bismuth_selenide() -> Structure:
    """Bi2Se3 (R-3m, hexagonal a = 4.14 A, c = 28.64 A) in its 5-atom
    rhombohedral primitive cell, a = 9.84 A and alpha = 24.28 deg, the cell
    MC3D mc3d-75053 stores. pymatgen puts c along z, so a and b point out
    of the xy plane."""
    a, c = 4.14, 28.64
    length = math.sqrt(a**2 / 3 + c**2 / 9)
    alpha = math.degrees(math.acos((2 * c**2 - 3 * a**2) / (2 * c**2 + 6 * a**2)))
    bismuth, selenium = 0.4008, 0.2117
    return Structure(
        Lattice.rhombohedral(length, alpha),
        ["Bi", "Bi", "Se", "Se", "Se"],
        [[bismuth] * 3, [-bismuth] * 3, [0, 0, 0], [selenium] * 3, [-selenium] * 3],
    )


def graphene_monolayer(c: float = 20.0) -> Structure:
    """A graphene sheet (C-C 1.42 A) in the xy plane, in a cell ``c`` A high
    along z (20 A by default), all of it vacuum."""
    return Structure(
        Lattice.hexagonal(2.46, c), ["C", "C"], [[0, 0, 0.5], [1 / 3, 2 / 3, 0.5]]
    )


def molybdenum_disulfide_monolayer(c: float = 20.0) -> Structure:
    """One S-Mo-S sheet of ``molybdenum_disulfide`` (a = 3.16 A, S planes
    3.17 A apart) in the xy plane, in a cell ``c`` A high along z (20 A by
    default)."""
    half_thickness = 1.585 / c
    return Structure(
        Lattice.hexagonal(3.16, c),
        ["Mo", "S", "S"],
        [
            [1 / 3, 2 / 3, 0.5],
            [2 / 3, 1 / 3, 0.5 + half_thickness],
            [2 / 3, 1 / 3, 0.5 - half_thickness],
        ],
    )


def janus_molybdenum_sulfide_selenide(c: float = 20.0) -> Structure:
    """A Janus MoSSe sheet (a = 3.22 A) in the xy plane: the S plane 1.55 A
    above Mo and the Se plane 1.65 A below, in a cell ``c`` A high along z
    (20 A by default). No symmetry operation maps c to -c (P3m1), so the
    sheet carries a dipole along its normal."""
    return Structure(
        Lattice.hexagonal(3.22, c),
        ["Mo", "S", "Se"],
        [
            [1 / 3, 2 / 3, 0.5],
            [2 / 3, 1 / 3, 0.5 + 1.55 / c],
            [2 / 3, 1 / 3, 0.5 - 1.65 / c],
        ],
    )


def rotated_about_x(structure: Structure, degrees: float) -> Structure:
    """``structure`` turned rigidly about the Cartesian x axis."""
    angle = np.radians(degrees)
    rotation = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )
    return Structure(
        Lattice(structure.lattice.matrix @ rotation.T),
        structure.species,
        structure.frac_coords,
    )


def with_vacuum_along(structure: Structure, axis: LatticeAxis) -> Structure:
    """``structure``, whose vacuum lies along c, with its lattice vectors and
    fractional coordinates permuted cyclically so that the vacuum vector
    becomes ``axis``. Atoms keep their Cartesian positions and site order."""
    order = np.roll([0, 1, 2], {"a": 1, "b": 2, "c": 0}[axis])
    return Structure(
        Lattice(structure.lattice.matrix[order]),
        structure.species,
        structure.frac_coords[:, order],
    )


def graphene_with_vacuum_along_x() -> Structure:
    """The graphene sheet of ``graphene_monolayer`` in the yz plane: the
    vacuum vector is a = (20, 0, 0) A, and b and c span the sheet."""
    return Structure(
        Lattice([[20.0, 0, 0], [0, 2.46, 0], [0, -1.23, 1.23 * math.sqrt(3)]]),
        ["C", "C"],
        [[0.5, 0, 0], [0.5, 1 / 3, 2 / 3]],
    )


def graphene_with_c_tilted() -> Structure:
    """``graphene_monolayer`` with c leaning 3 A along x: a and b stay in the
    xy plane and the vacuum along c is still 20 A."""
    sheet = graphene_monolayer()
    a, b, _ = sheet.lattice.matrix
    return Structure(Lattice([a, b, [3.0, 0, 20.0]]), sheet.species, sheet.frac_coords)


def graphene_with_vacuum_along_a() -> Structure:
    """``graphene_monolayer`` with its lattice vectors permuted cyclically:
    the vacuum vector a = (0, 0, 20) A comes first, and b and c span the
    sheet in the xy plane."""
    return Structure(
        Lattice([[0, 0, 20.0], [2.46, 0, 0], [-1.23, 1.23 * math.sqrt(3), 0]]),
        ["C", "C"],
        [[0.5, 0, 0], [0.5, 1 / 3, 2 / 3]],
    )


def carbon_chain() -> Structure:
    """A carbon chain along c (C-C 1.28 A) with 12 A between chains along a
    and b: a wire, periodic along c."""
    return Structure(
        Lattice.tetragonal(12.0, 2.56), ["C", "C"], [[0.5, 0.5, 0], [0.5, 0.5, 0.5]]
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


_MC3D_67775_SILICON_CIF = """\
data_0

loop_
  _atom_site_label
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  _atom_site_type_symbol
         Si1  0.6666666666716649  0.3333333333433297  0.06292314380102376  Si
         Si2  0.33333333332833526  0.6666666666566704  0.5629231437931663  Si
         Si3  0.6666666666716649  0.3333333333433297  0.4370768562068337  Si
         Si4  0.33333333332833526  0.6666666666566704  0.9370768561989763  Si
_cell_angle_alpha                       90.0
_cell_angle_beta                        90.0
_cell_angle_gamma                       120.00000000021976
_cell_length_a                          3.8504103266
_cell_length_b                          3.85041032657442
_cell_length_c                          6.3634082249
loop_
  _symmetry_equiv_pos_as_xyz
         'x, y, z'
_symmetry_int_tables_number             1
_symmetry_space_group_name_H-M          'P 1'
"""


def mc3d_67775_silicon() -> Structure:
    """Hexagonal Si, MC3D 67775, as goldilocks-data's SCF campaign ran it.
    Its a and b differ in the last few bits of the cell file
    (3.8504103266 against 3.85041032657442)."""
    return Structure.from_str(_MC3D_67775_SILICON_CIF, fmt="cif")


_MC3D_42058_SILICON_CIF = """\
data_0

loop_
  _atom_site_label
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  _atom_site_type_symbol
         Si1       0.5       0.5       0.5       Si
         Si2       0.7500000000092051  0.7500000000092051  0.7500000000092051  Si
_cell_angle_alpha                       59.99999999999999
_cell_angle_beta                        59.99999999999999
_cell_angle_gamma                       59.99999999999999
_cell_length_a                          3.8408466174999094
_cell_length_b                          3.8408466174999094
_cell_length_c                          3.8408466174999094
loop_
  _symmetry_equiv_pos_as_xyz
         'x, y, z'
_symmetry_int_tables_number             1
_symmetry_space_group_name_H-M          'P 1'
"""


def mc3d_42058_silicon() -> Structure:
    """Diamond Si, MC3D mc3d-42058: the 2-atom fcc primitive cell, which
    pymatgen writes with c along z and a in the xz plane. pw.x 7.5 found 4
    of its 48 symmetry operations in that frame and ran 282 k-points of a
    10x10x10 mesh (#140)."""
    return Structure.from_str(_MC3D_42058_SILICON_CIF, fmt="cif")


_MP_44_ALPHA_URANIUM_CIF = """\
data_U
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   2.77075206
_cell_length_b   3.23613745
_cell_length_c   4.85010636
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   115.34673871
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
  U  U0  1  0.09626689  0.19253377  0.75000000  1
  U  U1  1  0.90373311  0.80746523  0.25000000  1
"""


def alpha_uranium() -> Structure:
    """alpha-U (Cmcm), Materials Project mp-44, as MP serves it: the reduced
    primitive cell of the base-centred lattice, 2 atoms, gamma 115.35 deg.
    Its point group mixes a and b, whose reciprocal lengths differ (2.509
    against 2.148 A^-1)."""
    return Structure.from_str(_MP_44_ALPHA_URANIUM_CIF, fmt="cif")


_MP_1865_URANIUM_NITRIDE_CIF = """\
data_UN
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   3.44882753
_cell_length_b   3.44882701
_cell_length_c   3.44882695
_cell_angle_alpha   60.00001067
_cell_angle_beta   60.00001562
_cell_angle_gamma   60.00000542
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
  U  U0  1  0.50000000  0.50000000  0.50000000  1
  N  N1  1  0.00000000  0.00000000  0.00000000  1
"""


def uranium_nitride() -> Structure:
    """UN (rock salt, Fm-3m), Materials Project mp-1865, as MP serves it:
    the fcc primitive cell, whose three reciprocal lengths, equal by
    symmetry, differ in the seventh significant figure from CIF noise
    (up to 4.5e-7 A^-1)."""
    return Structure.from_str(_MP_1865_URANIUM_NITRIDE_CIF, fmt="cif")


def hcp_gadolinium() -> Structure:
    """hcp Gd (P6_3/mmc), Materials Project mp-155: 2 atoms, a = 3.642 A,
    c = 5.902 A."""
    return Structure.from_spacegroup(
        "P6_3/mmc", Lattice.hexagonal(3.642, 5.902), ["Gd"], [[1 / 3, 2 / 3, 0.25]]
    )


def fcc_lead() -> Structure:
    """fcc Pb (Fm-3m), MC3D mc3d-48868, in its 4-atom conventional cell
    (a = 4.919 A)."""
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(4.919), ["Pb"], [[0.0, 0.0, 0.0]]
    )


def geometry_facts(
    dimensionality: Dimensionality,
    vacuum_axes: tuple[LatticeAxis, ...] = (),
    *,
    ab_in_xy: bool = True,
) -> GeometryFacts:
    """``GeometryFacts`` for advisor and check tests: 20 A of vacuum along
    each axis in ``vacuum_axes`` and a 2 A gap along the others. With one
    vacuum axis the cell is a slab, in the xy frame when that axis is c and
    ``ab_in_xy``, with c along z, room for QE's 2D Coulomb truncation in that
    frame, and its vacuum centred on the cell origin, filling 80% of the
    cell."""
    slab = None
    if len(vacuum_axes) == 1:
        (axis,) = vacuum_axes
        xy_frame = axis == "c" and ab_in_xy
        slab = SlabGeometry(
            vacuum_axis=axis,
            xy_frame=xy_frame,
            fits_2d_truncation=xy_frame,
            vacuum_center=0.0,
            vacuum_fraction=0.8,
        )
    return GeometryFacts(
        dimensionality=dimensionality,
        low_dimensional=dimensionality != "3d",
        vacuum_axes=vacuum_axes,
        vacuum_widths_angstrom=tuple(
            20.0 if axis in vacuum_axes else 2.0 for axis in ("a", "b", "c")
        ),
        slab=slab,
    )


def written_cell(structure: Structure) -> Resolved[MagneticConfigFacts]:
    """A resolved non-magnetic ``MagneticConfigFacts`` whose ``cell`` is
    ``structure``: the cell the writer renders, which the written-cell rules
    of ``checks.check_all`` read."""
    facts = MagneticConfigFacts(
        cell=structure,
        ordering=OrderingRecord(requested="fm", applied="fm", compensated=False),
        spin_polarized=False,
        magnetic_elements=(),
        starting_magnetization=None,
        tot_magnetization=None,
        spin_orbit_enabled=False,
        angle1=None,
        angle2=None,
    )
    return Resolved(facts, Provenance(source="heuristic"))
