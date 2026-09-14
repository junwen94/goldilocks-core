"""Tests for goldilocks_core.service (v2 epic 8, #8).

Named ``test_service_v2.py``, not ``test_service.py``: that name is
already taken by v1's ``runtime.service.Service`` tests, which this
module does not replace (v2 epic 9 deletes the v1 tree, not this one).
"""

from __future__ import annotations

import json
import shutil

import pytest
from pymatgen.core import Lattice, Structure
from support import (
    SSSP_FIXTURE_UPF as _UPF,
    sssp_fixture_table_spec as _sssp_table_spec,
)

from goldilocks_core.advisors.magnetic_config import MagneticConfigHumanInput
from goldilocks_core.assets.pseudopotentials.importers import sssp_preparer
from goldilocks_core.assets.store import AssetStore
from goldilocks_core.bundle import DirectoryOutput, is_complete, publish
from goldilocks_core.inputs.hpc import Hardware, HpcProfile, Partition
from goldilocks_core.resolution import Blocked, Unavailable
from goldilocks_core.service import (
    AdviceIncomplete,
    RunOverrides,
    SystemOverrides,
    advise,
    check,
    generate,
    render_submission,
    to_bundle_input,
)
from goldilocks_core.steps import default_shared_context


@pytest.fixture
def silicon() -> Structure:
    return Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])


@pytest.fixture
def hpc() -> HpcProfile:
    hardware = Hardware(
        cores_per_node=64, mem_per_node_gb=250, max_nodes=100, max_walltime_h=168
    )
    return HpcProfile(
        name="test-cluster",
        scheduler="slurm",
        launcher="srun",
        modules={"quantum_espresso": ("qe/7.2",)},
        has_scalapack={"quantum_espresso": False},
        partitions={"scarf": Partition(name="scarf", hardware=hardware, default=True)},
    )


@pytest.fixture
def installed_table(tmp_path, monkeypatch):
    spec, registry_table = _sssp_table_spec(tmp_path / "sources")
    store = AssetStore(tmp_path / "store")
    store.install(spec, sssp_preparer(registry_table))
    monkeypatch.setattr(
        "goldilocks_core.service._pseudo.load_tables",
        lambda *a, **kw: {registry_table.id: registry_table},
    )
    return store, registry_table


class TestAdviseEndToEnd:
    """The real, non-hand-built-fixture pipeline: a genuine AssetStore
    install, not a FieldState constructed by hand -- this is the test
    every prior epic's own docstring flagged as missing until this one."""

    def test_full_pipeline_produces_a_complete_bundle(
        self, silicon, hpc, installed_table, tmp_path
    ) -> None:
        store, _table = installed_table

        advice = advise(silicon, hpc=hpc, store=store)
        report = check(advice)
        assert report.ok, report.blocking

        steps = generate(advice, report)
        assert len(steps) == 1
        assert steps[0].name == "scf"
        assert "scf.in" in steps[0].files

        ctx = default_shared_context()
        script = render_submission(advice, hpc, "quantum_espresso", ctx, steps)
        assert "#SBATCH" in script
        assert "pw.x" in script

        bundle_input = to_bundle_input(advice, steps, script, ctx)
        assert any(a["path"] == "submit.sh" for a in bundle_input.artifacts)
        assert any(a["path"] == "pseudo/Si.upf" for a in bundle_input.artifacts)
        assert bundle_input.citations == ("Synthetic SSSP fixture, cite me.",)

        destination = tmp_path / "out"
        publish(bundle_input, DirectoryOutput(destination))

        assert is_complete(destination)
        manifest = json.loads((destination / "goldilocks.json").read_text())
        assert manifest["records"]["functional"]["status"] == "resolved"
        assert manifest["records"]["functional"]["value"] == "PBEsol"
        assert manifest["records"]["pseudopotentials"]["status"] == "resolved"
        assert (destination / "pseudo" / "Si.upf").read_bytes() == _UPF
        assert (destination / "scf.in").exists()
        assert (destination / "submit.sh").exists()

    def test_warnings_flattens_every_record_s_structured_warnings(
        self, silicon, hpc, installed_table
    ) -> None:
        store, _table = installed_table

        advice = advise(silicon, hpc=hpc, store=store)

        warnings = advice.warnings()
        assert warnings  # scarf's own missing-walltime warning always fires
        assert all(
            warning.keys() == {"code", "level", "category", "message"}
            for warning in warnings
        )
        assert any(warning["code"] == "job.walltime_defaulted" for warning in warnings)

    def test_human_overrides_flow_through_to_generated_input(
        self, silicon, hpc, installed_table
    ) -> None:
        from goldilocks_core.advisors.functional import FunctionalHumanInput

        store, _ = installed_table
        overrides = RunOverrides(
            system=SystemOverrides(functional=FunctionalHumanInput(functional="PBE"))
        )

        advice = advise(silicon, hpc=hpc, store=store, overrides=overrides)

        assert advice.system.functional.ok
        assert advice.system.functional.value == "PBE"
        assert advice.system.functional.source == "human"
        # PBE pseudopotentials aren't in the fixture table (PBEsol only),
        # so downstream table selection correctly can't find a match --
        # proving the override actually reached pseudo_requirements().
        assert not advice.system.pseudo.table.ok

    def test_human_override_of_an_analysis_fact_flows_through(
        self, silicon, hpc, installed_table
    ) -> None:
        """`capabilities()` advertises the four analysis facts as
        overridable (v2 epic 8, #8) -- confirm `advise()`'s own end of
        that promise, not just that `set_overrides.build_overrides` can
        construct the ``RunOverrides`` (``test_set_overrides.py`` covers
        that half)."""
        from goldilocks_core.analysis.is_metal import IsMetalHumanInput
        from goldilocks_core.service._analysis import AnalysisOverrides

        store, _ = installed_table
        overrides = RunOverrides(
            analysis=AnalysisOverrides(is_metal=IsMetalHumanInput(is_metal=True))
        )

        advice = advise(silicon, hpc=hpc, store=store, overrides=overrides)

        assert advice.analysis.is_metal.ok
        assert advice.analysis.is_metal.value == "metal"
        assert advice.analysis.is_metal.source == "human"


class TestAdviseDegradation:
    """No asset store or network involved -- these exercise Blocked/
    Unavailable propagation through advise() using an explicit,
    guaranteed-missing table id, per the tri-state "never fake a value"
    contract."""

    def test_unknown_table_id_degrades_the_whole_pseudopotential_chain(
        self, silicon, hpc
    ) -> None:
        overrides = RunOverrides(
            system=SystemOverrides(pseudo_table_id="does-not-exist")
        )

        advice = advise(silicon, hpc=hpc, overrides=overrides)

        assert isinstance(advice.system.pseudo.table, Unavailable)
        assert isinstance(advice.system.pseudo.metadata, Blocked)
        assert isinstance(advice.system.pseudo.relativistic, Blocked)
        assert isinstance(advice.system.magnetic, Blocked)
        assert isinstance(advice.system.cutoffs, Blocked)
        assert isinstance(advice.system.electron_count, Blocked)
        assert isinstance(advice.step.kpoints.nbnd, Blocked)
        assert isinstance(advice.step.resources.job, Blocked)
        assert isinstance(advice.step.resources.parallelisation, Blocked)
        # Unrelated facts/advisors are unaffected. (is_metal is correctly
        # Unavailable for silicon regardless -- a semiconductor, not a
        # metal -- so it is not asserted here; see is_metal.py's own
        # docstring on why composition alone never confirms "non_metal".)
        assert advice.analysis.composition.ok
        assert advice.system.functional.ok
        assert advice.step.kpoints.k_sampling.ok

    def test_soc_on_a_lanthanide_blocks_the_whole_relativistic_chain(
        self, hpc
    ) -> None:
        """v2 epic 9 (#9)'s "SOC on Ce" acceptance scenario: SSSP is the
        only table lanthanides are allowed to use (``requires_sssp``), but
        no SSSP table is fully relativistic -- so requesting spin-orbit
        coupling on a lanthanide can never be satisfied by any table in
        the registry, real or synthetic, no asset store needed to prove
        it (table *selection* fails before any file is ever touched)."""
        cerium = Structure(Lattice.cubic(5.16), ["Ce"], [[0.0, 0.0, 0.0]])
        overrides = RunOverrides(
            system=SystemOverrides(
                magnetic=MagneticConfigHumanInput(spin_orbit_coupling=True)
            )
        )

        advice = advise(cerium, hpc=hpc, overrides=overrides)

        assert isinstance(advice.system.pseudo.table, Unavailable)
        assert "Ce" in advice.system.pseudo.table.reason
        assert isinstance(advice.system.pseudo.metadata, Blocked)
        assert isinstance(advice.system.pseudo.relativistic, Blocked)
        assert isinstance(advice.system.magnetic, Blocked)
        assert isinstance(advice.system.cutoffs, Blocked)
        # Genuinely unrelated decisions -- never fed pseudo/magnetic data
        # at all -- still resolve normally.
        assert advice.system.functional.ok
        assert advice.step.kpoints.k_sampling.ok
        assert advice.step.kpoints.occupations.ok
        assert advice.system.boundary.ok

    def test_blocked_root_cause_is_reachable_from_a_downstream_field(
        self, silicon, hpc
    ) -> None:
        overrides = RunOverrides(
            system=SystemOverrides(pseudo_table_id="does-not-exist")
        )

        advice = advise(silicon, hpc=hpc, overrides=overrides)

        assert "does-not-exist" in advice.step.resources.job.root_cause() or (
            "does-not-exist" in advice.system.pseudo.table.reason
        )

    def test_check_reports_blocking_and_generate_refuses(self, silicon, hpc) -> None:
        overrides = RunOverrides(
            system=SystemOverrides(pseudo_table_id="does-not-exist")
        )
        advice = advise(silicon, hpc=hpc, overrides=overrides)

        report = check(advice)

        assert not report.ok
        assert report.blocking

        with pytest.raises(AdviceIncomplete) as excinfo:
            generate(advice, report)
        assert excinfo.value.report is report

    def test_advice_incomplete_message_deduplicates_repeated_root_causes(
        self, silicon, hpc
    ) -> None:
        """report.blocking has one entry per blocked field, not per
        distinct cause -- many fields share the one pseudo_table_id
        failure here. The exception message must say it once."""
        overrides = RunOverrides(
            system=SystemOverrides(pseudo_table_id="does-not-exist")
        )
        advice = advise(silicon, hpc=hpc, overrides=overrides)
        report = check(advice)
        assert len(report.blocking) > 1, "test assumes multiple fields share one cause"

        with pytest.raises(AdviceIncomplete) as excinfo:
            generate(advice, report)

        assert str(excinfo.value).count("unknown pseudopotential table") == 1

    def test_fetch_missing_false_degrades_without_raising(
        self, silicon, hpc, tmp_path, monkeypatch
    ) -> None:
        _spec, registry_table = _sssp_table_spec(tmp_path / "sources")
        monkeypatch.setattr(
            "goldilocks_core.service._pseudo.load_tables",
            lambda *a, **kw: {registry_table.id: registry_table},
        )
        empty_store = AssetStore(tmp_path / "empty-store")

        advice = advise(silicon, hpc=hpc, store=empty_store, fetch_missing=False)

        assert isinstance(advice.system.pseudo.metadata, Unavailable)
        assert "not installed" in advice.system.pseudo.metadata.reason
        assert isinstance(advice.system.cutoffs, Blocked)
        # A table *was* selected (only the asset isn't downloaded yet) --
        # ``relativistic`` is already confirmed, and magnetic still falls
        # back to its pre-pseudo provisional guess rather than blocking:
        # this is an ordinary preview-without-download run, not a chain
        # failure (contrast the SOC-on-a-lanthanide test above, where no
        # table can ever be selected at all).
        assert advice.system.pseudo.relativistic.ok
        assert advice.system.magnetic.ok

    def test_fetch_missing_true_installs_then_succeeds(
        self, silicon, hpc, tmp_path, monkeypatch
    ) -> None:
        spec, registry_table = _sssp_table_spec(tmp_path / "sources")
        monkeypatch.setattr(
            "goldilocks_core.service._pseudo.load_tables",
            lambda *a, **kw: {registry_table.id: registry_table},
        )
        empty_store = AssetStore(tmp_path / "empty-store")
        prepare = sssp_preparer(registry_table)
        monkeypatch.setattr(
            "goldilocks_core.service._pseudo.install_assets",
            lambda name, *, store: (store.install(spec, prepare),),
        )

        advice = advise(silicon, hpc=hpc, store=empty_store, fetch_missing=True)

        assert advice.system.pseudo.metadata.ok
        assert advice.system.cutoffs.ok


@pytest.mark.skipif(
    shutil.which("enum.x") is None and shutil.which("multienum.x") is None,
    reason="needs the enumlib executables (enum.x, makeStr.py) on PATH",
)
class TestAfmRelabeling:
    """v2 epic 9 (#9)'s AFM acceptance scenario, through the real
    ``advise()`` orchestrator -- not just ``magnetic_config()`` in
    isolation (that half is ``test_advisors_magnetic_config.py``'s job).
    No asset store needed: table *selection* (which is all AFM relabeling
    itself depends on) works off the bundled registry data alone."""

    def test_afm_ordering_produces_a_relabeled_structure_and_a_different_symmetry_eff(
        self, hpc
    ) -> None:
        rock_salt_feo = Structure(
            Lattice.cubic(4.3), ["Fe", "O"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        )
        overrides = RunOverrides(
            system=SystemOverrides(
                magnetic=MagneticConfigHumanInput(magnetic_ordering="afm")
            )
        )

        advice = advise(rock_salt_feo, hpc=hpc, overrides=overrides)

        assert advice.system.magnetic.ok
        relabeled = advice.system.magnetic.value.relabeled_structure
        assert len(set(relabeled.species)) > len(set(rock_salt_feo.species))

        assert advice.analysis.symmetry.ok
        assert advice.system.symmetry_eff.ok
        assert advice.system.symmetry_eff.value != advice.analysis.symmetry.value

        # k_sampling/n_irr_k consumed the relabeled (bigger) cell, not the
        # original one, and did not crash doing it.
        assert advice.step.kpoints.k_sampling.ok
        assert advice.step.kpoints.n_irr_k.ok

    def test_fm_default_keeps_symmetry_eff_identical_to_symmetry(self, hpc) -> None:
        rock_salt_feo = Structure(
            Lattice.cubic(4.3), ["Fe", "O"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        )

        advice = advise(rock_salt_feo, hpc=hpc)

        assert advice.system.magnetic.ok
        assert advice.system.magnetic.value.relabeled_structure == rock_salt_feo
        assert advice.system.symmetry_eff.value == advice.analysis.symmetry.value


def test_pseudo_requirements_reflects_first_pass_spin_orbit(
    silicon, hpc, installed_table
) -> None:
    """Regression guard for the two-pass magnetic_config wiring: the
    table search must use the *first*-pass spin_orbit_enabled, since the
    real per-element z_valence calibration only exists after a table is
    already chosen."""
    store, table = installed_table

    advice = advise(silicon, hpc=hpc, store=store)

    assert advice.system.magnetic.ok
    assert advice.system.pseudo.table.ok
    assert advice.system.pseudo.table.value.id == table.id
