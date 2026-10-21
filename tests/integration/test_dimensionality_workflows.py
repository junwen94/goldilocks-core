"""Dimensionality-driven decisions through the real service pipeline.

``advise()``/``check()``/``generate()`` and their relax variants on Si
cells, with the synthetic SSSP Si table from ``tests/support.py``, so these
run without installed pseudopotentials (#126). The dispersion tests (#141)
read only ``advise()``/``check()``, which need no pseudopotentials either.
"""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure
from support import graphite, sssp_fixture_table_spec

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
from goldilocks_core.set_overrides import build_overrides

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


def test_lda_graphite_gets_no_d3_dispersion_and_a_warning() -> None:
    """#141: QE's DFT-D3 has no D3(BJ) parameters for LDA, so graphite keeps
    its default D3 only under a functional that has them."""
    hpc = load_hpc_profile("scarf")
    default = advise(graphite(), hpc=hpc)
    lda = advise(graphite(), hpc=hpc, overrides=build_overrides({"functional": "LDA"}))

    assert default.system.vdw.value.use_vdw is True
    assert lda.analysis.geometry.value.dimensionality == "2d"
    assert lda.system.vdw.value.use_vdw is False
    codes = {warning["code"] for warning in lda.warnings()}
    assert "vdw.d3bj_unavailable_for_functional" in codes
    assert not any("use_vdw" in reason for reason in check(lda).blocking)


def test_d3_dispersion_with_lda_is_refused_before_generation() -> None:
    """#141: pw.x would stop at startup with 'functional name unknown'."""
    advice = advise(
        graphite(),
        hpc=load_hpc_profile("scarf"),
        overrides=build_overrides({"functional": "LDA", "use_vdw": True}),
    )

    report = check(advice)

    assert advice.system.vdw.source == "human"
    dispersion = [reason for reason in report.blocking if "use_vdw=false" in reason]
    assert len(dispersion) == 1
    assert "'LDA'" in dispersion[0]
