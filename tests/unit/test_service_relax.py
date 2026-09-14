"""Tests for goldilocks_core.service._relax (v2 epic 10, #10)."""

from __future__ import annotations

from pymatgen.core import Lattice, Structure
from support import sssp_fixture_table_spec

from goldilocks_core.advisors.relax import RelaxHumanInput, VcRelaxOptions
from goldilocks_core.assets.pseudopotentials.importers import sssp_preparer
from goldilocks_core.assets.store import AssetStore
from goldilocks_core.inputs.hpc import Hardware, HpcProfile, Partition
from goldilocks_core.resolution import Blocked
from goldilocks_core.service import (
    RelaxOverrides,
    RunOverrides,
    StepOverrides,
    SystemOverrides,
    advise_relax,
    check_relax,
    generate_relax,
)
from goldilocks_core.set_overrides import build_overrides


def _hpc() -> HpcProfile:
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


def _silicon() -> Structure:
    return Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])


class TestAdviseRelaxDegradation:
    def test_relax_settings_resolve_with_heuristic_defaults(self) -> None:
        advice = advise_relax(_silicon(), calculation="relax", hpc=_hpc())

        assert advice.relax.ok
        assert advice.relax.value.ion_dynamics == "bfgs"
        assert advice.relax.value.nstep == 50

    def test_vc_relax_task_resolves_a_vc_relax_options(self) -> None:
        advice = advise_relax(_silicon(), calculation="vc-relax", hpc=_hpc())

        assert isinstance(advice.relax.value, VcRelaxOptions)

    def test_set_style_override_reaches_relax_settings(self) -> None:
        """Unlike DosHumanInput/DosLlmInput (v2 epic 9, #9, never wired
        to any transport), relax's own settings are real ``--set``/HTTP
        overrides -- capabilities.bindings() knows about them."""
        overrides = build_overrides({"nstep": 200, "ion_dynamics": "fire"})

        advice = advise_relax(
            _silicon(), calculation="relax", hpc=_hpc(), overrides=overrides
        )

        assert advice.relax.value.nstep == 200
        assert advice.relax.value.ion_dynamics == "fire"
        assert advice.relax.source == "human"

    def test_structural_override_reaches_relax_settings_too(self) -> None:
        overrides = RunOverrides(
            step=StepOverrides(
                relax=RelaxOverrides(relax=RelaxHumanInput(remove_rigid_rot=True))
            )
        )

        advice = advise_relax(
            _silicon(), calculation="relax", hpc=_hpc(), overrides=overrides
        )

        assert advice.relax.value.remove_rigid_rot is True

    def test_unknown_table_id_blocks_relax_and_check_relax_reports_it(self) -> None:
        overrides = RunOverrides(
            system=SystemOverrides(pseudo_table_id="does-not-exist")
        )

        advice = advise_relax(
            _silicon(), calculation="relax", hpc=_hpc(), overrides=overrides
        )

        assert isinstance(advice.advice.system.pseudo.metadata, Blocked)
        report = check_relax(advice)
        assert not report.ok
        assert any("does-not-exist" in reason for reason in report.blocking)

    def test_vc_relax_with_non_bfgs_ion_dynamics_is_blocked_end_to_end(self) -> None:
        overrides = build_overrides({"ion_dynamics": "damp"})

        advice = advise_relax(
            _silicon(), calculation="vc-relax", hpc=_hpc(), overrides=overrides
        )
        report = check_relax(advice)

        assert not report.ok
        assert any("ion_dynamics='bfgs'" in reason for reason in report.blocking)


class TestGenerateRelaxEndToEnd:
    def _store(self, tmp_path, monkeypatch) -> AssetStore:
        spec, registry_table = sssp_fixture_table_spec(tmp_path / "sources")
        monkeypatch.setattr(
            "goldilocks_core.service._pseudo.load_tables",
            lambda *a, **kw: {registry_table.id: registry_table},
        )
        store = AssetStore(tmp_path / "store")
        store.install(spec, sssp_preparer(registry_table))
        return store

    def test_generate_relax_produces_one_step(self, tmp_path, monkeypatch) -> None:
        store = self._store(tmp_path, monkeypatch)

        advice = advise_relax(_silicon(), calculation="relax", hpc=_hpc(), store=store)
        report = check_relax(advice)
        assert report.ok, report.blocking

        steps = generate_relax(advice, report)

        assert [step.name for step in steps] == ["relax"]
        assert steps[0].executable == "pw.x"
        content = steps[0].files["relax.in"]
        assert "calculation      = 'relax'" in content
        assert "&IONS" in content
        assert "&CELL" not in content

    def test_generate_vc_relax_produces_one_step_with_cell_namelist(
        self, tmp_path, monkeypatch
    ) -> None:
        store = self._store(tmp_path, monkeypatch)

        advice = advise_relax(
            _silicon(), calculation="vc-relax", hpc=_hpc(), store=store
        )
        report = check_relax(advice)
        assert report.ok, report.blocking

        steps = generate_relax(advice, report)

        assert [step.name for step in steps] == ["vc-relax"]
        content = steps[0].files["vc-relax.in"]
        assert "calculation      = 'vc-relax'" in content
        assert "&CELL" in content
