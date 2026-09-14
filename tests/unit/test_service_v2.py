"""Tests for goldilocks_core.service (v2 epic 8, #8).

Named ``test_service_v2.py``, not ``test_service.py``: that name is
already taken by v1's ``runtime.service.Service`` tests, which this
module does not replace (v2 epic 9 deletes the v1 tree, not this one).
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from goldilocks_core.assets.pseudopotentials.importers import sssp_preparer
from goldilocks_core.assets.pseudopotentials.registry import PseudoTable
from goldilocks_core.assets.records import AssetFile, AssetSpec
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

_UPF = (
    b'<UPF version="2.0.1">\n'
    b'<PP_HEADER element="Si" pseudo_type="NC" functional="PBEsol" '
    b'relativistic="scalar" z_valence="4.0"/>\n'
    b"</UPF>\n"
)


def _archive(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


def _sssp_table_spec(tmp_path: Path) -> tuple[AssetSpec, PseudoTable]:
    """Build one real, fully-offline SSSP-shaped pseudopotential asset
    (file:// sources, no network), matching the pattern already used by
    ``tests/unit/test_pseudo_importers.py``'s ``install_sssp_fixture``."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    upfs = tmp_path / "table.tar.gz"
    sidecar = tmp_path / "table.json"
    licence = tmp_path / "LICENSE.txt"
    _archive(upfs, {"nested/Si.upf": _UPF})
    sidecar.write_text(
        json.dumps(
            {
                "Si": {
                    "filename": "Si.upf",
                    "md5": hashlib.md5(_UPF).hexdigest(),
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

    def test_warnings_flattens_every_record_s_prose_warnings(
        self, silicon, hpc, installed_table
    ) -> None:
        store, _table = installed_table

        advice = advise(silicon, hpc=hpc, store=store)

        warnings = advice.warnings()
        assert warnings  # scarf's own missing-walltime warning always fires
        assert all(isinstance(message, str) and ": " in message for message in warnings)

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
