"""Tests for goldilocks_core.service._dos (v2 epic 9, #9)."""

from __future__ import annotations

from pymatgen.core import Lattice, Structure
from support import sssp_fixture_table_spec

from goldilocks_core.assets.pseudopotentials.importers import sssp_preparer
from goldilocks_core.assets.store import AssetStore
from goldilocks_core.inputs.hpc import Hardware, HpcProfile, Partition
from goldilocks_core.resolution import Blocked
from goldilocks_core.service import advise_dos, check_dos, generate_dos


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


class TestAdviseDosDegradation:
    """No asset store needed: table *selection* alone (real registry
    data, no network) is enough to exercise the nscf overrides and the
    dos advisor -- same style as test_service_v2.py's own
    ``TestAdviseDegradation``."""

    def test_nscf_step_gets_a_denser_mesh_and_tetrahedra_opt_occupations(self) -> None:
        advice = advise_dos(_silicon(), hpc=_hpc())

        assert advice.scf.step.kpoints.occupations.value.occupations == "smearing"
        assert advice.nscf.step.kpoints.occupations.value.occupations == (
            "tetrahedra_opt"
        )
        scf_k_distance = advice.scf.step.kpoints.k_sampling.value.k_distance
        nscf_k_distance = advice.nscf.step.kpoints.k_sampling.value.k_distance
        assert nscf_k_distance == 0.10
        assert nscf_k_distance < scf_k_distance

    def test_dos_settings_resolve_from_the_nscf_steps_own_occupations(self) -> None:
        advice = advise_dos(_silicon(), hpc=_hpc())

        assert advice.dos.ok
        # nscf used tetrahedra_opt (forced) -- no smearing width exists,
        # so broadening is correctly left unset, not invented.
        assert advice.dos.value.broadening is None
        assert advice.dos.value.delta_e == 0.01
        assert advice.dos.value.ngauss == 0

    def test_unknown_table_id_blocks_both_pw_steps_and_check_dos_reports_it(
        self,
    ) -> None:
        from goldilocks_core.service import RunOverrides, SystemOverrides

        overrides = RunOverrides(
            system=SystemOverrides(pseudo_table_id="does-not-exist")
        )

        advice = advise_dos(_silicon(), hpc=_hpc(), overrides=overrides)

        assert isinstance(advice.scf.system.pseudo.metadata, Blocked)
        assert isinstance(advice.nscf.system.pseudo.metadata, Blocked)
        report = check_dos(advice)
        assert not report.ok
        assert any("does-not-exist" in reason for reason in report.blocking)


class TestGenerateDosEndToEnd:
    def test_generate_dos_produces_three_steps_sharing_one_context(
        self, tmp_path, monkeypatch
    ) -> None:
        spec, registry_table = sssp_fixture_table_spec(tmp_path / "sources")
        monkeypatch.setattr(
            "goldilocks_core.service._pseudo.load_tables",
            lambda *a, **kw: {registry_table.id: registry_table},
        )
        store = AssetStore(tmp_path / "store")
        store.install(spec, sssp_preparer(registry_table))

        advice = advise_dos(_silicon(), hpc=_hpc(), store=store)
        report = check_dos(advice)
        assert report.ok, report.blocking

        steps = generate_dos(advice, report)

        assert [step.name for step in steps] == ["scf", "nscf", "dos"]
        assert [step.executable for step in steps] == ["pw.x", "pw.x", "dos.x"]
        scf_step, nscf_step, dos_step = steps
        assert "calculation      = 'scf'" in scf_step.files["scf.in"]
        assert "calculation      = 'nscf'" in nscf_step.files["nscf.in"]
        assert "occupations      = 'tetrahedra_opt'" in nscf_step.files["nscf.in"]
        assert "&DOS" in dos_step.files["dos.in"]
        assert "deltae           = 0.01" in dos_step.files["dos.in"]
        # All three steps share one prefix/outdir (SharedContext).
        for step in steps:
            content = next(iter(step.files.values()))
            assert "prefix           = 'pwscf'" in content
            assert "outdir           = './out'" in content
