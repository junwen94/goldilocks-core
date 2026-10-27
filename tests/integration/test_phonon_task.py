"""The phonon tasks (#215), step by step on a synthetic SSSP-shaped table.

The first step's k-mesh: under the phonon tasks' k rule, ``advise_steps``
raises the mesh the tiers decide against the ph step's q-mesh, on the
simulated cell, and the k-points pw.x runs and the job's pools follow the
raised mesh. No model is installed, so the heuristic tier decides.
"""

from __future__ import annotations

import pytest
from pymatgen.core import Structure
from support import fcc_lead, sodium_chloride, sssp_fixture_table_spec

from goldilocks_core.advisors.k_sampling import PhononKProtocol
from goldilocks_core.advisors.q_sampling import QSamplingHumanInput, q_sampling
from goldilocks_core.assets.pseudopotentials.importers import sssp_preparer
from goldilocks_core.assets.store import AssetStore
from goldilocks_core.inputs.hpc import load_hpc_profile
from goldilocks_core.service import Advice, RunOverrides
from goldilocks_core.service._pipeline import advise_steps
from goldilocks_core.service._step_protocol import StepProtocol
from goldilocks_core.set_overrides import build_overrides

_Z_VALENCES = {"Cl": 7.0, "Na": 9.0, "Pb": 14.0}


@pytest.fixture
def store(tmp_path, monkeypatch) -> AssetStore:
    """The scalar table, covering every element here."""
    spec, table = sssp_fixture_table_spec(tmp_path / "scalar", _Z_VALENCES)
    monkeypatch.setattr(
        "goldilocks_core.service._pseudo.load_tables",
        lambda *a, **kw: {table.id: table},
    )
    installed = AssetStore(tmp_path / "store")
    installed.install(spec, sssp_preparer(table))
    return installed


def _scf(
    structure: Structure,
    store: AssetStore,
    phonon: PhononKProtocol | None,
    overrides: RunOverrides | None = None,
) -> Advice:
    """The advice for one first step of ``structure``, under the phonon k
    rule ``phonon`` (``None`` for none)."""
    overrides = overrides or RunOverrides()
    return advise_steps(
        structure,
        code="quantum_espresso",
        hpc=load_hpc_profile("scarf"),
        overrides=overrides,
        steps=overrides.for_steps(("scf",)),
        store=store,
        fetch_missing=False,
        protocols={"scf": StepProtocol(phonon_k=phonon)},
    )["scf"]


def test_the_raised_k_mesh_is_the_one_pw_x_runs_and_the_job_splits(store) -> None:
    """fcc Pb is a metal: the heuristic 0.15 1/A gives 9x9x9, which the
    smearing run raises to 10x10x10, the next multiple of the 5x5x5 q-mesh.
    pw.x runs 56 irreducible k-points instead of 35, and the job's 64 tasks
    split into 32 pools instead of 8."""
    plain = _scf(fcc_lead(), store, None).step
    raised = _scf(fcc_lead(), store, PhononKProtocol()).step

    k_sampling = raised.kpoints.k_sampling
    assert plain.kpoints.k_sampling.value.mesh == (9, 9, 9)
    assert k_sampling.value.mesh == (10, 10, 10)
    assert raised.kpoints.occupations.value.occupations == "smearing"
    assert k_sampling.provenance.source_note.startswith(
        "Raised for DFPT from 9x9x9 to 10x10x10: each k-axis to the next "
        "multiple of its q-axis (q-mesh 5x5x5)"
    )
    assert [warning.code for warning in k_sampling.value.warnings] == [
        "phonon.k_mesh_raised"
    ]
    pairs = [
        (step.kpoints.n_irr_k.value, step.resources.parallelisation.value)
        for step in (plain, raised)
    ]
    assert [(n_irr_k, layout.kpoint_spin_pairs) for n_irr_k, layout in pairs] == [
        (35, 35),
        (56, 56),
    ]
    assert [layout.npool for _, layout in pairs] == [8, 32]


def test_a_request_q_mesh_sets_the_mesh_the_k_mesh_meets(store) -> None:
    """Rock-salt NaCl as a confirmed non-metal runs on fixed occupations at
    the heuristic 0.30 1/A, 4x4x4. A request q_distance of 0.2 1/A gives the
    q record 6x6x6 on the same cell, and the k-mesh is raised to it."""
    overrides = build_overrides({"is_metal": "non_metal"}, task="scf_single_point")
    q = QSamplingHumanInput(q_distance=0.2)

    plain = _scf(sodium_chloride(), store, None, overrides)
    advice = _scf(sodium_chloride(), store, PhononKProtocol(q), overrides)

    kpoints = advice.step.kpoints
    assert kpoints.occupations.value.occupations == "fixed"
    assert plain.step.kpoints.k_sampling.value.mesh == (4, 4, 4)
    assert q_sampling(advice.system.cell, q).value.mesh == (6, 6, 6)
    assert kpoints.k_sampling.value.mesh == (6, 6, 6)
    assert kpoints.k_sampling.value.k_distance is None
    assert (plain.step.kpoints.n_irr_k.value, kpoints.n_irr_k.value) == (10, 20)
