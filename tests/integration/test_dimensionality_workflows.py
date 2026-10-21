"""Vacuum-driven decisions through the real service pipeline (#126).

``advise()``/``check()``/``generate()`` and their relax variants on Si
cells, with the synthetic SSSP Si table from ``tests/support.py``, so these
run without installed pseudopotentials.
"""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure
from support import sssp_fixture_table_spec

from goldilocks_core.assets.pseudopotentials.importers import sssp_preparer
from goldilocks_core.assets.store import AssetStore
from goldilocks_core.inputs.hpc import load_hpc_profile
from goldilocks_core.service import (
    advise,
    advise_relax,
    check,
    check_relax,
    generate,
    generate_relax,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def store(tmp_path, monkeypatch) -> AssetStore:
    spec, registry_table = sssp_fixture_table_spec(tmp_path / "sources")
    monkeypatch.setattr(
        "goldilocks_core.service._pseudo.load_tables",
        lambda *_args, **_kwargs: {registry_table.id: registry_table},
    )
    store = AssetStore(tmp_path / "store")
    store.install(spec, sssp_preparer(registry_table))
    return store


def silicon_dimer_in_a_box() -> Structure:
    """An Si2 dimer (bond 2.25 A, along [111]) in a 12 A cubic box: 10.7 A
    of vacuum along every lattice vector."""
    offset = 2.25 / (12.0 * 3**0.5)
    return Structure(
        Lattice.cubic(12.0), ["Si", "Si"], [[0, 0, 0], [offset, offset, offset]]
    )


def diamond_silicon() -> Structure:
    return Structure.from_spacegroup(
        "Fd-3m", Lattice.cubic(5.431), ["Si"], [[0.0, 0.0, 0.0]]
    )


def test_an_isolated_molecule_gets_martyna_tuckerman_and_no_stress(
    store: AssetStore,
) -> None:
    advice = advise(
        silicon_dimer_in_a_box(), hpc=load_hpc_profile("scarf"), store=store
    )
    report = check(advice)
    assert report.ok, report.blocking

    content = generate(advice, report)[0].files["scf.in"]

    assert advice.system.boundary.value.assume_isolated == "martyna-tuckerman"
    assert "assume_isolated  = 'martyna-tuckerman'" in content
    assert "tstress" not in content


def test_vc_relax_of_an_isolated_molecule_is_refused(store: AssetStore) -> None:
    """Under MT, pw.x computes no stress, so a vc-relax would report the
    input cell as converged (#126)."""
    advice = advise_relax(
        silicon_dimer_in_a_box(),
        calculation="vc-relax",
        hpc=load_hpc_profile("scarf"),
        store=store,
    )

    report = check_relax(advice)

    assert len(report.blocking) == 1
    assert "assume_isolated='martyna-tuckerman'" in report.blocking[0]
    assert "task='relax'" in report.blocking[0]


def test_vc_relax_of_a_bulk_crystal_computes_the_stress(store: AssetStore) -> None:
    advice = advise_relax(
        diamond_silicon(),
        calculation="vc-relax",
        hpc=load_hpc_profile("scarf"),
        store=store,
    )
    report = check_relax(advice)
    assert report.ok, report.blocking

    content = generate_relax(advice, report)[0].files["vc-relax.in"]

    assert "tstress          = .true." in content
    assert "assume_isolated" not in content
