"""The band task's pw.x steps (#118) on a synthetic SSSP-shaped Si table.

``advise_steps`` advises the scf and the pw.x bands step of diamond Si
(MC3D mc3d-42058) together, under the band task's protocols: the bands step
runs every point of the written cell's band path, three times the occupied
bands, the scf step's occupations, and its own share of the one job both
steps run in. Its input repeats scf.in apart from the step's own values and
the K_POINTS card.
"""

from __future__ import annotations

import pytest
from support import mc3d_42058_silicon, sssp_fixture_table_spec

from goldilocks_core.advisors.convergence import ConvergenceHumanInput
from goldilocks_core.advisors.k_path import KPathHumanInput
from goldilocks_core.advisors.nbnd import NbndHumanInput
from goldilocks_core.assets.pseudopotentials.importers import sssp_preparer
from goldilocks_core.assets.store import AssetStore
from goldilocks_core.generation.quantum_espresso.bands import k_points_crystal_b
from goldilocks_core.inputs.hpc import load_hpc_profile
from goldilocks_core.resolution import Blocked
from goldilocks_core.service import (
    Advice,
    KPathOverrides,
    KpointsOverrides,
    PathStepAdvice,
    RunOverrides,
    StepOverrides,
    SystemOverrides,
    check,
    generate,
)
from goldilocks_core.service._pipeline import advise_steps
from goldilocks_core.service._step_protocol import BAND_SCF_PROTOCOL, BANDS_PROTOCOL
from goldilocks_core.steps import ExitStatusCheck, default_shared_context


@pytest.fixture
def store(tmp_path, monkeypatch) -> AssetStore:
    spec, registry_table = sssp_fixture_table_spec(tmp_path / "sources")
    monkeypatch.setattr(
        "goldilocks_core.service._pseudo.load_tables",
        lambda *a, **kw: {registry_table.id: registry_table},
    )
    installed = AssetStore(tmp_path / "store")
    installed.install(spec, sssp_preparer(registry_table))
    return installed


def _advice(
    store: AssetStore,
    bands: StepOverrides | None = None,
    overrides: RunOverrides | None = None,
) -> dict[str, Advice]:
    """Diamond Si's scf and bands steps, ``bands`` the bands step's own
    overrides and ``overrides`` the task's."""
    return advise_steps(
        mc3d_42058_silicon(),
        code="quantum_espresso",
        hpc=load_hpc_profile("scarf"),
        overrides=overrides or RunOverrides(),
        steps={"scf": StepOverrides(), "bands": bands or StepOverrides()},
        store=store,
        fetch_missing=False,
        protocols={"scf": BAND_SCF_PROTOCOL, "bands": BANDS_PROTOCOL},
    )


def _cards(content: str) -> dict[str, str]:
    """Each card of a pw.x input by its header line, namelists left out."""
    blocks = (block.strip("\n") for block in content.split("\n\n"))
    return {
        block.splitlines()[0]: block
        for block in blocks
        if block and not block.startswith("&")
    }


def test_the_bands_step_runs_every_point_of_the_written_cell_s_path(store) -> None:
    """pw.x runs a bands list as given, unreduced by symmetry (QE 7.3
    ``PW/src/setup.f90:579-585``), so the step's k-points are the path's
    points. Its occupations are the scf step's, its convergence the scf's
    rule on the same cell, and its bands three times the 4 occupied ones
    (aiida-quantumespresso's bands.yaml), against the scf's 8."""
    advice = _advice(store)
    scf, bands = advice["scf"].step.kpoints, advice["bands"].step.kpoints

    assert isinstance(bands, PathStepAdvice)
    path = bands.k_path.value
    assert path.lattice_type == "FCC"
    assert path.cell is advice["bands"].system.cell.value.structure
    assert bands.n_irr_k.value == path.n_kpoints
    assert bands.occupations.value == scf.occupations.value
    assert bands.occupations.provenance.derived_from == ("occupations",)
    assert (scf.nbnd.value.nbnd, bands.nbnd.value.nbnd) == (8, 12)
    assert bands.convergence.value == scf.convergence.value
    assert set(advice["bands"].step.records()) == {
        "occupations",
        "k_path",
        "nbnd",
        "convergence",
        "job",
        "parallelisation",
    }


def test_the_bands_step_reads_its_own_overrides(store) -> None:
    """A bands step's k_path_distance spaces its path, and its nbnd and
    conv_thr are its own: the scf step keeps its defaults."""
    default = _advice(store)["bands"].step.kpoints.k_path.value
    advice = _advice(
        store,
        StepOverrides(
            kpoints=KpointsOverrides(
                nbnd=NbndHumanInput(nbnd=20),
                convergence=ConvergenceHumanInput(conv_thr=1e-11),
            ),
            k_path=KPathOverrides(k_path=KPathHumanInput(k_path_distance=0.05)),
        ),
    )
    scf, bands = advice["scf"].step.kpoints, advice["bands"].step.kpoints

    path = bands.k_path
    assert path.value.k_path_distance == 0.05
    assert path.provenance.field_sources == {"k_path_distance": "human"}
    assert path.value.n_kpoints < default.n_kpoints
    assert [branch.vertices for branch in path.value.branches] == [
        branch.vertices for branch in default.branches
    ]
    assert (bands.nbnd.value.nbnd, bands.nbnd.source) == (20, "human")
    assert (scf.nbnd.value.nbnd, scf.nbnd.source) == (8, "heuristic")
    assert (bands.convergence.value.conv_thr, scf.convergence.value.conv_thr) == (
        1e-11,
        4e-10,
    )


def test_one_job_holds_both_pw_x_steps(store) -> None:
    """The steps run one after the other in one allocation (#169): the job
    is sized over both and splits 95% of its walltime between them by
    workload -- k-point/spin pairs x nbnd x 3 SCF passes for the scf, x 1
    for the bands step, which runs no SCF loop (npw is the same for both
    and cancels). The bands step's pools share out its path's points."""
    advice = _advice(store)
    scf, bands = advice["scf"].step, advice["bands"].step

    job = scf.resources.job.value
    assert bands.resources.job.value is job
    shares = job.max_seconds_by_step
    assert shares.keys() == {"scf", "bands"}
    assert (scf.resources.max_seconds, bands.resources.max_seconds) == (
        shares["scf"],
        shares["bands"],
    )
    budget = int(job.walltime_h * 3600 * 0.95)
    workload = {
        name: (
            step.resources.parallelisation.value.kpoint_spin_pairs
            * step.kpoints.nbnd.value.nbnd
            * passes
        )
        for name, step, passes in (("scf", scf, 3), ("bands", bands, 1))
    }
    for name, share in shares.items():
        assert abs(share - budget * workload[name] / sum(workload.values())) < 1
    assert bands.resources.parallelisation.value.kpoint_spin_pairs == (
        bands.kpoints.k_path.value.n_kpoints
    )


def test_without_pseudopotentials_the_bands_step_degrades(store) -> None:
    """Advice degrades instead of raising: without the table there is no
    electron count, so the bands step's band count and the job are blocked
    by the missing table, while its path, which reads only the cell,
    resolves. The gate names the table once."""
    advice = _advice(
        store,
        overrides=RunOverrides(system=SystemOverrides(pseudo_table_id="nowhere")),
    )
    bands = advice["bands"].step

    assert bands.kpoints.k_path.ok
    assert isinstance(bands.kpoints.nbnd, Blocked)
    assert isinstance(bands.resources.job, Blocked)
    (finding,) = check(advice["bands"], purpose="bands").blocking
    assert finding.code == "advice.blocked"
    assert "nowhere" in finding.message


def test_bands_in_repeats_scf_in_but_for_the_k_points_card(store) -> None:
    """``generate(purpose='bands')`` writes bands.in from the same system as
    scf.in: every card but K_POINTS is the scf's, and its K_POINTS crystal_b
    card holds the vertices of the step's path record. Both steps report
    through the status file of the same save."""
    advice = _advice(store)
    ctx = default_shared_context()
    scf_report = check(advice["scf"])
    bands_report = check(advice["bands"], purpose="bands")
    assert scf_report.ok, scf_report.blocking
    assert bands_report.ok, bands_report.blocking

    (scf,) = generate(advice["scf"], scf_report, ctx=ctx)
    (bands,) = generate(advice["bands"], bands_report, ctx=ctx, purpose="bands")

    scf_cards = _cards(scf.files["scf.in"])
    bands_cards = _cards(bands.files["bands.in"])
    assert scf_cards.pop("K_POINTS automatic")
    assert bands_cards.pop("K_POINTS crystal_b") == k_points_crystal_b(
        advice["bands"].step.kpoints.k_path.value
    ).strip("\n")
    assert bands_cards == scf_cards
    assert list(bands_cards) == [
        "ATOMIC_SPECIES",
        "CELL_PARAMETERS angstrom",
        "ATOMIC_POSITIONS crystal",
    ]
    lines = bands.files["bands.in"].splitlines()
    assert "   calculation      = 'bands'" in lines
    assert "   nbnd             = 12" in lines
    assert (bands.name, bands.stdout) == ("bands", "bands.out")
    assert isinstance(bands.completion, ExitStatusCheck)
    assert bands.completion == scf.completion
