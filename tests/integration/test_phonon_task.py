"""The phonon tasks (#215), step by step on a synthetic SSSP-shaped table.

The first step's k-mesh: under the phonon tasks' k rule, ``advise_steps``
raises the mesh the tiers decide against the ph step's q-mesh, on the
simulated cell, and the k-points pw.x runs and the job's pools follow the
raised mesh. No model is installed, so the heuristic tier decides.

The job: under a job timer no step gets a max_seconds share, and the first
step's input carries none, while the dos and band tasks still split the
walltime between their pw.x steps.

The phonon steps: ``phonon_steps_advice`` advises the ph, q2r and phonon_dos
steps, a relaxation first step's relax record and the first step's FFT grid
on the advice of a first step under ``phonon_step_protocol``. The overrides
are routed for the scf, relax or vc-relax task, so the phonon steps keep
their defaults. The first step's input carries the grid ph.x accepts and
two-body D3, and the ph.x, q2r.x and matdyn.x inputs follow it on its
context.
"""

from __future__ import annotations

from typing import Literal

import pytest
from pymatgen.core import Structure
from support import (
    alpha_uranium,
    fcc_lead,
    mc3d_42058_silicon,
    sodium_chloride,
    sssp_fixture_table_spec,
)

from goldilocks_core.advisors.job_resources import JobTimerProtocol
from goldilocks_core.advisors.k_sampling import PhononKProtocol
from goldilocks_core.advisors.phonon import AIIDA_PROTOCOLS_CITATION
from goldilocks_core.advisors.q_sampling import QSamplingHumanInput, q_sampling
from goldilocks_core.assets.pseudopotentials.importers import sssp_preparer
from goldilocks_core.assets.store import AssetStore
from goldilocks_core.generation.quantum_espresso.phonon import write_qe_phonon
from goldilocks_core.inputs.hpc import load_hpc_profile
from goldilocks_core.resolution import Warning
from goldilocks_core.service import (
    Advice,
    RunOverrides,
    advise_band,
    advise_dos,
    check,
    generate,
)
from goldilocks_core.service._phonon_steps import (
    PhononStepsAdvice,
    phonon_step_settings,
    phonon_steps_advice,
)
from goldilocks_core.service._pipeline import advise_steps
from goldilocks_core.service._step_protocol import (
    PHONON_JOB_TIMER,
    RELAX_PHONON_PROTOCOL,
    StepProtocol,
    phonon_step_protocol,
)
from goldilocks_core.set_overrides import build_overrides
from goldilocks_core.steps import SharedContext

_Z_VALENCES = {"Cl": 7.0, "Na": 9.0, "Pb": 14.0, "Si": 4.0, "U": 14.0}

_FIRST_STEP_TASKS = {
    "scf": "scf_single_point",
    "relax": "relax",
    "vc-relax": "vc-relax",
}
"""The task whose routing gives each first step its overrides: the request
keys of the system and the first step route, and the phonon steps keep their
defaults."""


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
    protocol: StepProtocol,
    overrides: RunOverrides | None = None,
) -> Advice:
    """The advice for one first step of ``structure``, under ``protocol``."""
    overrides = overrides or RunOverrides()
    return advise_steps(
        structure,
        code="quantum_espresso",
        hpc=load_hpc_profile("scarf"),
        overrides=overrides,
        steps=overrides.for_steps(("scf",)),
        store=store,
        fetch_missing=False,
        protocols={"scf": protocol},
    )["scf"]


def test_the_raised_k_mesh_is_the_one_pw_x_runs_and_the_job_splits(store) -> None:
    """fcc Pb is a metal: the heuristic 0.15 1/A gives 9x9x9, which the
    smearing run raises to 10x10x10, the next multiple of the 5x5x5 q-mesh.
    pw.x runs 56 irreducible k-points instead of 35, and the job's 64 tasks
    split into 32 pools instead of 8."""
    plain = _scf(fcc_lead(), store, StepProtocol()).step
    raised = _scf(fcc_lead(), store, StepProtocol(phonon_k=PhononKProtocol())).step

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

    plain = _scf(sodium_chloride(), store, StepProtocol(), overrides)
    advice = _scf(
        sodium_chloride(), store, StepProtocol(phonon_k=PhononKProtocol(q)), overrides
    )

    kpoints = advice.step.kpoints
    assert kpoints.occupations.value.occupations == "fixed"
    assert plain.step.kpoints.k_sampling.value.mesh == (4, 4, 4)
    assert q_sampling(advice.system.cell, q).value.mesh == (6, 6, 6)
    assert kpoints.k_sampling.value.mesh == (6, 6, 6)
    assert kpoints.k_sampling.value.k_distance is None
    assert (plain.step.kpoints.n_irr_k.value, kpoints.n_irr_k.value) == (10, 20)


def test_a_job_timer_leaves_the_first_step_without_a_max_seconds_share(
    store,
) -> None:
    """Diamond Si under a job timer: the job holds no share and fires its
    timer at 95% of SCARF's 168 h, its record carries the protocol's note,
    and scf.in leaves max_seconds to pw.x's default."""
    timer = JobTimerProtocol(note="Test protocol: one timer for the whole job.")

    advice = _scf(mc3d_42058_silicon(), store, StepProtocol(job_timer=timer))

    resources = advice.step.resources
    assert resources.max_seconds is None
    assert resources.job.value.max_seconds_by_step == {}
    assert resources.job.value.timer_seconds == 574560
    assert resources.job.provenance.source_note == timer.note
    (scf,) = generate(advice, check(advice))
    assert "max_seconds" not in scf.files["scf.in"]


def test_the_dos_and_band_tasks_still_split_the_walltime(store) -> None:
    """Neither task's protocols declare a job timer, so each of its pw.x
    steps keeps its own max_seconds share of the one job."""
    hpc = load_hpc_profile("scarf")
    dos = advise_dos(mc3d_42058_silicon(), hpc=hpc, store=store)
    band = advise_band(mc3d_42058_silicon(), hpc=hpc, store=store)

    for steps in (
        {"scf": dos.scf, "nscf": dos.nscf},
        {"scf": band.scf, "bands": band.bands},
    ):
        job = steps["scf"].step.resources.job
        assert job.value.timer_seconds is None
        assert job.provenance.source_note is None
        shares = {
            purpose: advice.step.resources.max_seconds
            for purpose, advice in steps.items()
        }
        assert job.value.max_seconds_by_step == shares
        assert min(shares.values()) > 0


def _phonon(
    structure: Structure,
    store: AssetStore,
    first_step: Literal["scf", "relax", "vc-relax"],
    assignments: dict[str, object] | None = None,
) -> tuple[Advice, PhononStepsAdvice]:
    """The first step's advice under the phonon tasks' protocol for it, and
    the phonon steps advised on it, for a request of ``assignments``."""
    hpc = load_hpc_profile("scarf")
    overrides = build_overrides(assignments or {}, task=_FIRST_STEP_TASKS[first_step])
    steps = overrides.for_steps((first_step, "ph", "q2r", "phonon_dos"))
    protocol = phonon_step_protocol(first_step, steps["ph"].ph.q_sampling)
    first = advise_steps(
        structure,
        code="quantum_espresso",
        hpc=hpc,
        overrides=overrides,
        steps={first_step: steps[first_step]},
        store=store,
        fetch_missing=False,
        protocols={first_step: protocol},
    )[first_step]
    return first, phonon_steps_advice(first_step, first, steps, hpc, protocol.relax)


def _codes(warnings: tuple[Warning, ...]) -> list[str]:
    return [warning.code for warning in warnings]


def _keywords(text: str) -> dict[str, str]:
    """The ``key = value`` lines of a rendered input, values as written."""
    pairs = (line.split("=", 1) for line in text.splitlines() if "=" in line)
    return {key.strip(): value.strip() for key, value in pairs}


def test_the_phonon_steps_of_silicon(store) -> None:
    """Diamond Si: ph.x runs the 7x7x7 q-mesh of 0.3 1/A, 20 irreducible
    q-points and a work of 2401 scf-equivalents, with the balanced ph
    protocol; q2r.x and matdyn.x apply asr 'crystal', and the DOS mesh is
    21x21x21. The phonon record carries the phonon task's relax-first notice.
    pw.x's own FFT grid passes ph.x's test, so the first step writes none,
    and the job runs the phonon tasks' timer in place of any max_seconds
    share."""
    first, steps = _phonon(mc3d_42058_silicon(), store, "scf")

    q = steps.q_sampling.value
    assert (q.mesh, q.q_distance, q.n_irr_q, q.work, q.refusal) == (
        (7, 7, 7),
        0.3,
        20,
        2401,
        None,
    )
    phonon = steps.phonon
    assert (phonon.value.tr2_ph, phonon.value.alpha_mix, phonon.value.nmix_ph) == (
        1e-18,
        0.4,
        8,
    )
    assert phonon.source == "heuristic"
    assert first.step.kpoints.occupations.value.occupations == "smearing"
    assert _codes(phonon.value.warnings) == ["phonon.relax_first"]
    assert (steps.q2r.value.asr, steps.q2r.value.loto_2d) == ("crystal", False)
    assert steps.phonon_dos.value.mesh == (21, 21, 21)
    fft_grid = steps.fft_grid.value
    assert (fft_grid.compatible, fft_grid.written, fft_grid.fragile) == (
        True,
        None,
        False,
    )
    assert steps.relax is None
    resources = first.step.resources
    assert resources.max_seconds is None
    assert resources.job.value.max_seconds_by_step == {}
    assert resources.job.value.timer_seconds == 574560
    assert resources.job.provenance.source_note == PHONON_JOB_TIMER.note
    assert steps.first_step_records() == {"fft_grid": steps.fft_grid}
    assert steps.step_records() == {
        "ph": {"q_sampling": steps.q_sampling, "phonon": steps.phonon},
        "q2r": {"q2r": steps.q2r},
        "phonon_dos": {"phonon_dos": steps.phonon_dos},
    }


@pytest.mark.parametrize("first_step", ["scf", "relax"])
def test_the_first_step_writes_the_fft_grid_ph_x_accepts(store, first_step) -> None:
    """alpha-U given a Hubbard U on atomic projectors, the ones ph.x
    accepts: at the table's 120 Ry, pw.x's own dense grid, 20x24x32, is one
    ph.x refuses, so the scf of the phonon task and the fixed-cell
    relaxation of relax-phonon both write 24x24x32 as nr1-nr3, beside the
    HUBBARD card."""
    first, steps = _phonon(
        alpha_uranium(),
        store,
        first_step,
        {"hubbard_needs_correlation": True, "hubbard_projectors": "atomic"},
    )

    fft_grid = steps.fft_grid.value
    assert (fft_grid.pw_grid, fft_grid.compatible, fft_grid.written) == (
        (20, 24, 32),
        False,
        (24, 24, 32),
    )
    assert "phonon.hubbard_experimental" in _codes(steps.phonon.value.warnings)
    relax = None if steps.relax is None else steps.relax.value
    (step,) = generate(
        first,
        check(first, purpose=first_step, relax=steps.relax),
        purpose=first_step,
        relax=relax,
        fft_grid=fft_grid.written,
    )
    content = step.files[f"{first_step}.in"]
    keywords = _keywords(content)
    assert (keywords["nr1"], keywords["nr2"], keywords["nr3"]) == ("24", "24", "32")
    assert "HUBBARD (atomic)" in content


@pytest.mark.parametrize("first_step", ["relax", "vc-relax"])
def test_a_relaxation_first_step_relaxes_to_the_phonon_threshold(
    store, first_step
) -> None:
    """The relaxation that replaces the phonon scf runs to forc_conv_thr
    1e-4 Ry/Bohr, the phonon protocol's value, recorded as heuristic with the
    protocol's note and citation."""
    _, steps = _phonon(alpha_uranium(), store, first_step)

    relax = steps.relax
    assert relax.value.forc_conv_thr == 1e-4
    assert relax.provenance.field_sources["forc_conv_thr"] == "heuristic"
    assert relax.provenance.source_note == RELAX_PHONON_PROTOCOL.note
    assert relax.provenance.citations == (AIIDA_PROTOCOLS_CITATION,)
    assert relax.value.warnings == ()
    assert steps.first_step_records() == {
        "relax": steps.relax,
        "fft_grid": steps.fft_grid,
    }


def test_a_vc_relax_first_step_writes_no_fft_grid(store) -> None:
    """A vc-relax's final scf rebuilds pw.x's own grid for the relaxed cell,
    so on alpha-U, whose own grid ph.x refuses and whose b and a axes make it
    fragile, vc-relax.in carries no nr1-nr3."""
    first, steps = _phonon(alpha_uranium(), store, "vc-relax")

    fft_grid = steps.fft_grid.value
    assert (fft_grid.compatible, fft_grid.written, fft_grid.fragile_axes) == (
        False,
        None,
        (("b", "a"),),
    )
    (vc_relax,) = generate(
        first,
        check(first, purpose="vc-relax", relax=steps.relax),
        purpose="vc-relax",
        relax=steps.relax.value,
        fft_grid=fft_grid.written,
    )
    assert "nr1" not in vc_relax.files["vc-relax.in"]


def test_dft_d3_turns_the_first_step_s_three_body_terms_off(store) -> None:
    """With use_vdw=true the phonon record carries the two-body notice, since
    d3hess.x builds the D3 Hessian from two-body terms only, and the first
    step writes dftd3_threebody = .false. beside its DFT-D3 keywords."""
    first, steps = _phonon(mc3d_42058_silicon(), store, "scf", {"use_vdw": True})

    assert first.system.vdw.value.use_vdw
    assert _codes(steps.phonon.value.warnings) == [
        "phonon.relax_first",
        "phonon.d3_two_body",
    ]
    (scf,) = generate(first, check(first), dftd3_threebody=False)
    keywords = _keywords(scf.files["scf.in"])
    assert (keywords["vdw_corr"], keywords["dftd3_threebody"]) == (
        "'grimme-d3'",
        ".false.",
    )


def test_the_phonon_inputs_follow_the_first_step_on_its_context(store) -> None:
    """ph.x, q2r.x and matdyn.x run after the scf on one context: ph.x reads
    its save under the same prefix and outdir with its pools, and q2r.x and
    matdyn.x read the files named for that prefix."""
    first, steps = _phonon(mc3d_42058_silicon(), store, "scf")
    ctx = SharedContext(prefix="si", outdir="./si-out", pseudo_dir="./pseudo")

    (scf,) = generate(first, check(first), ctx=ctx)
    ph, q2r, dos = write_qe_phonon(
        *phonon_step_settings(steps, first), ctx, d3hess=False
    )

    assert [step.name for step in (scf, ph, q2r, dos)] == [
        "scf",
        "ph",
        "q2r",
        "phonon_dos",
    ]
    scf_keywords = _keywords(scf.files["scf.in"])
    ph_keywords = _keywords(ph.files["ph.in"])
    assert (scf_keywords["prefix"], scf_keywords["outdir"]) == ("'si'", "'./si-out'")
    assert (ph_keywords["prefix"], ph_keywords["outdir"]) == ("'si'", "'./si-out'")
    assert (ph_keywords["nq1"], ph_keywords["nq2"], ph_keywords["nq3"]) == (
        "7",
        "7",
        "7",
    )
    assert (ph_keywords["tr2_ph"], ph_keywords["fildyn"]) == ("1e-18", "'si.dyn'")
    npool = first.step.resources.parallelisation.value.npool
    assert ph.args[:2] == ("-npool", str(npool))
    assert _keywords(q2r.files["q2r.in"]) == {
        "fildyn": "'si.dyn'",
        "flfrc": "'si.fc'",
        "zasr": "'crystal'",
    }
    dos_keywords = _keywords(dos.files["phonon_dos.in"])
    assert (dos_keywords["flfrc"], dos_keywords["asr"], dos_keywords["fldos"]) == (
        "'si.fc'",
        "'crystal'",
        "'si.phdos'",
    )
    assert (dos_keywords["nk1"], dos_keywords["nk2"], dos_keywords["nk3"]) == (
        "21",
        "21",
        "21",
    )
