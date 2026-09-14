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
from typing import TYPE_CHECKING

from goldilocks_core.assets.records import AssetFile, AssetSpec
from goldilocks_core.assets.runtime import statuses
from goldilocks_core.assets.store import AssetStore, asset_root

if TYPE_CHECKING:
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
    *arguments: str, cwd: Path | None = None
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
    )


SSSP_FIXTURE_UPF = (
    b'<UPF version="2.0.1">\n'
    b'<PP_HEADER element="Si" pseudo_type="NC" functional="PBEsol" '
    b'relativistic="scalar" z_valence="4.0"/>\n'
    b"</UPF>\n"
)


def sssp_fixture_archive(path: Path, members: dict[str, bytes]) -> None:
    """Write a ``.tar.gz`` containing ``members`` -- shared by every test
    that builds a fully-offline (``file://``, no network) SSSP-shaped
    pseudopotential asset."""
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


def sssp_fixture_table_spec(tmp_path: Path) -> tuple[AssetSpec, PseudoTable]:
    """Build one real, fully-offline SSSP-shaped pseudopotential asset
    (file:// sources, no network) -- shared between
    ``tests/unit/test_service_v2.py`` and ``tests/unit/test_service_dos.py``
    (v2 epic 9, #9), matching the pattern already used by
    ``tests/unit/test_pseudo_importers.py``'s ``install_sssp_fixture``."""
    from goldilocks_core.assets.pseudopotentials.registry import PseudoTable

    tmp_path.mkdir(parents=True, exist_ok=True)
    upfs = tmp_path / "table.tar.gz"
    sidecar = tmp_path / "table.json"
    licence = tmp_path / "LICENSE.txt"
    sssp_fixture_archive(upfs, {"nested/Si.upf": SSSP_FIXTURE_UPF})
    sidecar.write_text(
        json.dumps(
            {
                "Si": {
                    "filename": "Si.upf",
                    "md5": hashlib.md5(SSSP_FIXTURE_UPF).hexdigest(),
                    "cutoff_wfc": 30.0,
                    "cutoff_rho": 120.0,
                    "pseudopotential": "Si fixture",
                }
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
        elements=("Si",),
        asset=spec,
        default=True,
    )
    return spec, registry_table
