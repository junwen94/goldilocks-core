"""Integration tests for the v2 CLI (v2 epic 8, #8), replacing
``test_cli_interface.py``: that file tested v1's ``compute``/
``capabilities``/preset/record-selection/model-legal-metadata surface,
none of which exists in the design doc's 6-command shape this rewrite
replaces it with (issue #8's own "v1 concepts this epic replaces"
section). Kept as a real subprocess-spawning integration suite, same
style as the file it replaces: ``goldilocks_core.cli.core:main`` is the
actual ``[project.scripts]`` entry point, so this is the only test
layer that exercises real argument parsing end to end.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure
from support import run_cli as _run_cli

from goldilocks_core.capabilities import capabilities
from goldilocks_core.examples.structures import structure


def test_help_lists_the_design_docs_commands() -> None:
    completed = _run_cli("--help")

    assert completed.returncode == 0, completed.stderr
    for command in (
        "run",
        "explain",
        "inspect",
        "settings",
        "capabilities",
        "models",
        "assets",
        "examples",
        "serve",
    ):
        assert command in completed.stdout


def test_serve_http_help_lists_static_root() -> None:
    """#59: restores the flag ``poe stage``/the README's local-dev
    instructions already assumed exists. Real end-to-end verification
    (a running server actually serving a built Workbench directory) was
    done manually, not here -- `serve http` blocks forever, so a
    subprocess integration test can only prove the flag parses, not
    that the server behaves correctly once it does."""
    completed = _run_cli("serve", "http", "--help")

    assert completed.returncode == 0, completed.stderr
    assert "--static-root" in completed.stdout


def test_inspect_reports_structure_in_json_and_human_forms() -> None:
    silicon = structure("Si.cif")

    human = _run_cli("inspect", str(silicon))
    assert human.returncode == 0, human.stderr
    assert "formula: Si" in human.stdout

    as_json = _run_cli("inspect", str(silicon), "--json")
    assert as_json.returncode == 0, as_json.stderr
    document = json.loads(as_json.stdout)
    assert document["structure"]["reduced_formula"] == "Si"


def test_inspect_missing_structure_is_an_operator_error_not_a_traceback() -> None:
    completed = _run_cli("inspect", "/no/such/structure.cif")

    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
    assert "not found" in completed.stderr


def test_magnetic_orderings_lists_the_fm_identity_by_default() -> None:
    """The fm candidate is always listed first, regardless of whatever
    else enumlib (if present on this machine's real PATH -- this
    subprocess inherits it, unlike an in-process monkeypatch) finds for
    bcc Fe -- #87's listing never picks a winner on its own."""
    iron = structure("Fe_bcc.cif")

    completed = _run_cli("magnetic-orderings", str(iron))

    assert completed.returncode == 0, completed.stderr
    assert "fm" in completed.stdout


def test_magnetic_orderings_json_reports_every_candidate() -> None:
    iron = structure("Fe_bcc.cif")

    completed = _run_cli("magnetic-orderings", str(iron), "--json")

    assert completed.returncode == 0, completed.stderr
    document = json.loads(completed.stdout)
    assert document["ranked"] is False
    assert document["candidates"][0]["label"] == "fm"
    assert document["candidates"][0]["formula"] == "Fe"
    assert document["candidates"][0]["energy_per_atom_ev"] is None


def test_magnetic_orderings_rank_degrades_without_a_configured_checkpoint(
    monkeypatch,
) -> None:
    """``--rank-with-mmace`` without ``GOLDILOCKS_MACE_BACKBONE`` configured
    (the real state of this test machine, and of CI) reports the
    candidates unranked with a warning, not a failure."""
    monkeypatch.delenv("GOLDILOCKS_MACE_BACKBONE", raising=False)
    iron = structure("Fe_bcc.cif")

    completed = _run_cli("magnetic-orderings", str(iron), "--rank-with-mmace", "--json")

    assert completed.returncode == 0, completed.stderr
    document = json.loads(completed.stdout)
    assert document["ranked"] is False
    # Not an exact warnings count: this subprocess inherits the real
    # machine's PATH, so whether AFM enumeration also warns
    # (magnetic.afm_ordering_unavailable) depends on whether enum.x
    # happens to be installed here -- only ranking degradation is this
    # test's own concern.
    codes = {warning["code"] for warning in document["warnings"]}
    assert "magnetic.ordering_ranking_unavailable" in codes


def test_settings_json_matches_the_capabilities_contract() -> None:
    completed = _run_cli("settings", "--json")

    assert completed.returncode == 0, completed.stderr
    settings = json.loads(completed.stdout)
    keys = {setting["key"] for setting in settings}
    assert "functional" in keys
    assert "ecutwfc_ry" in keys


def test_settings_human_output_lists_sources() -> None:
    completed = _run_cli("settings")

    assert completed.returncode == 0, completed.stderr
    assert "sources: human . heuristic" in completed.stdout


def test_capabilities_json_matches_the_real_capabilities_payload() -> None:
    """#62: the CLI must be a thin third entry point onto the same
    ``capabilities()`` HTTP's ``GET /capabilities`` and MCP's
    ``capabilities`` tool already call, not a re-derived subset."""
    completed = _run_cli("capabilities", "--json")

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == capabilities()


def test_capabilities_human_output_summarizes_every_section() -> None:
    completed = _run_cli("capabilities")

    assert completed.returncode == 0, completed.stderr
    for expected in (
        "core_version:",
        "code: quantum_espresso",
        "task: scf_single_point",
        "task: dos",
        "pseudopotential_table:",
        "hpc_profile:",
        "warnings:",
    ):
        assert expected in completed.stdout


def test_models_list_reports_every_registered_model() -> None:
    """v2 epic 11 (#11): models[] lists what registry.toml declares
    regardless of install status (an isolated, per-test asset root has
    nothing installed) -- "is it actually usable" lives in a fact's own
    ``approaches`` instead, not here."""
    completed = _run_cli("models", "--json", "list")

    assert completed.returncode == 0, completed.stderr
    ids = {model["id"] for model in json.loads(completed.stdout)}
    assert ids == {
        "models/qrf-kpoints",
        "models/metallicity-cgcnn",
        "models/is-metal-classifier",
        "models/is-magnetic-classifier",
    }


def test_assets_status_prints_the_asset_root() -> None:
    completed = _run_cli("assets", "status")

    assert completed.returncode == 0, completed.stderr
    assert "asset root:" in completed.stdout


def test_assets_status_json_carries_asset_root_and_a_list() -> None:
    completed = _run_cli("assets", "status", "--json")

    assert completed.returncode == 0, completed.stderr
    document = json.loads(completed.stdout)
    assert document["asset_root"]  # isolated per-test root; just non-empty
    assert isinstance(document["assets"], list)


def test_examples_path_prints_a_real_directory() -> None:
    completed = _run_cli("examples", "path")

    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()).is_dir()


def test_examples_path_json_carries_the_same_directory() -> None:
    completed = _run_cli("examples", "path", "--json")

    assert completed.returncode == 0, completed.stderr
    document = json.loads(completed.stdout)
    assert Path(document["path"]).is_dir()


def test_unknown_set_key_is_a_did_you_mean_operator_error() -> None:
    silicon = structure("Si.cif")

    completed = _run_cli("run", str(silicon), "--hpc", "scarf", "--set", "ecutwf_ry=30")

    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
    assert "did you mean: ecutwfc_ry" in completed.stderr


def test_unknown_hpc_profile_is_an_operator_error() -> None:
    silicon = structure("Si.cif")

    completed = _run_cli("explain", str(silicon), "--hpc", "does-not-exist")

    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr


def test_blocked_pipeline_reports_deduplicated_reasons_and_exits_nonzero() -> None:
    """No real assets needed: an unknown pseudo_table_id fails inside
    the bundled registry lookup, before ever touching the asset store."""
    silicon = structure("Si.cif")

    completed = _run_cli(
        "run",
        str(silicon),
        "--hpc",
        "scarf",
        "--set",
        "pseudo_table_id=does-not-exist",
    )

    assert completed.returncode == 2
    lines = [
        line for line in completed.stderr.splitlines() if line.startswith("blocked:")
    ]
    assert len(lines) == 1, completed.stderr
    assert "does-not-exist" in lines[0]


class TestRunAndExplainAgainstRealAssets:
    """Skips itself when the default pseudopotential profile isn't
    installed (same convention as test_docs_executable.py's
    real_assets fixture) -- these are the only tests in this file that
    need a real, resolvable pseudopotential."""

    def test_explain_prints_every_decision_with_its_source(
        self, real_assets: None
    ) -> None:
        silicon = structure("Si.cif")

        completed = _run_cli("explain", str(silicon), "--hpc", "scarf")

        assert completed.returncode == 0, completed.stderr
        assert "functional: 'PBEsol' (source=heuristic)" in completed.stdout
        # v2 epic 11 (#11): the real, installed CGCNN classifier resolves
        # Si confidently (elemental Si gives composition-only heuristics
        # nothing to exclude, so that tier alone would say unavailable --
        # ml is strictly more capable here, not a fallback).
        assert "is_metal: 'non_metal' (source=ml)" in completed.stdout

    def test_explain_json_carries_a_structured_top_level_warnings_array(
        self, real_assets: None
    ) -> None:
        """Matches HTTP/MCP's own ``/explain`` shape
        (``{"records", "warnings"}``) -- v2 epic 8's own "warnings array
        in every CLI/HTTP/MCP response" requirement."""
        silicon = structure("Si.cif")

        completed = _run_cli("explain", str(silicon), "--hpc", "scarf", "--json")

        assert completed.returncode == 0, completed.stderr
        document = json.loads(completed.stdout)
        assert document.keys() == {"records", "warnings"}
        assert document["warnings"]
        assert any(
            warning["code"] == "job.walltime_defaulted"
            for warning in document["warnings"]
        )
        assert all(
            warning.keys() == {"code", "level", "category", "message"}
            for warning in document["warnings"]
        )

    def test_run_without_out_previews_without_touching_disk(
        self, real_assets: None, tmp_path: Path
    ) -> None:
        silicon = structure("Si.cif")

        completed = _run_cli("run", str(silicon), "--hpc", "scarf", cwd=tmp_path)

        assert completed.returncode == 0, completed.stderr
        assert "memory-only preview" in completed.stdout
        assert "scf.in" in completed.stdout
        assert list(tmp_path.iterdir()) == []

    def test_explain_dos_task_reports_nscf_prefixed_records(
        self, real_assets: None
    ) -> None:
        """v2 epic 9 (#9, #28): --task dos used to be silently accepted
        and ignored by every delivery layer -- this confirms it now
        actually routes to advise_dos through the real CLI subprocess."""
        silicon = structure("Si.cif")

        completed = _run_cli(
            "explain", str(silicon), "--hpc", "scarf", "--task", "dos", "--json"
        )

        assert completed.returncode == 0, completed.stderr
        records = json.loads(completed.stdout)["records"]
        # v2 epic 11 (#11): the real, installed CGCNN classifier now
        # resolves Si confidently non-metal (source=ml), so the scf
        # step's own occupations correctly follows the non-metal branch
        # -- unlike nscf_occupations below, which dos's own advisor
        # always pins to tetrahedra_opt regardless of metallicity.
        assert records["occupations"]["value"]["occupations"] == "fixed"
        assert records["nscf_occupations"]["value"]["occupations"] == ("tetrahedra_opt")
        assert records["dos"]["value"]["delta_e"] == 0.01

    def test_run_dos_task_publishes_all_three_steps(
        self, real_assets: None, tmp_path: Path
    ) -> None:
        silicon = structure("Si.cif")
        destination = tmp_path / "out"

        completed = _run_cli(
            "run",
            str(silicon),
            "--hpc",
            "scarf",
            "--task",
            "dos",
            "-o",
            str(destination),
        )

        assert completed.returncode == 0, completed.stderr
        assert (destination / "scf.in").exists()
        assert (destination / "nscf.in").exists()
        assert (destination / "dos.in").exists()
        assert (destination / "submit.sh").exists()

    def test_explain_relax_task_reports_the_relax_record(
        self, real_assets: None
    ) -> None:
        """v2 epic 10 (#10): confirms --task relax actually routes to
        advise_relax through the real CLI subprocess, the same
        end-to-end check #28 added for --task dos."""
        silicon = structure("Si.cif")

        completed = _run_cli(
            "explain", str(silicon), "--hpc", "scarf", "--task", "relax", "--json"
        )

        assert completed.returncode == 0, completed.stderr
        records = json.loads(completed.stdout)["records"]
        assert records["relax"]["value"]["ion_dynamics"] == "bfgs"

    def test_run_relax_task_publishes_relax_in(
        self, real_assets: None, tmp_path: Path
    ) -> None:
        silicon = structure("Si.cif")
        destination = tmp_path / "out"

        completed = _run_cli(
            "run",
            str(silicon),
            "--hpc",
            "scarf",
            "--task",
            "relax",
            "-o",
            str(destination),
        )

        assert completed.returncode == 0, completed.stderr
        assert (destination / "relax.in").exists()
        content = (destination / "relax.in").read_text()
        assert "calculation      = 'relax'" in content

    def test_run_vc_relax_task_publishes_vc_relax_in(
        self, real_assets: None, tmp_path: Path
    ) -> None:
        silicon = structure("Si.cif")
        destination = tmp_path / "out"

        completed = _run_cli(
            "run",
            str(silicon),
            "--hpc",
            "scarf",
            "--task",
            "vc-relax",
            "-o",
            str(destination),
        )

        assert completed.returncode == 0, completed.stderr
        assert (destination / "vc-relax.in").exists()
        content = (destination / "vc-relax.in").read_text()
        assert "calculation      = 'vc-relax'" in content
        assert "&CELL" in content

    def test_set_nstep_reaches_the_generated_relax_input(
        self, real_assets: None, tmp_path: Path
    ) -> None:
        """Regression-shaped guard, same class of bug #34 fixed for
        other settings: an accepted --set must actually reach the
        rendered file, not just validate and get dropped."""
        silicon = structure("Si.cif")
        destination = tmp_path / "out"

        completed = _run_cli(
            "run",
            str(silicon),
            "--hpc",
            "scarf",
            "--task",
            "relax",
            "--set",
            "nstep=123",
            "-o",
            str(destination),
        )

        assert completed.returncode == 0, completed.stderr
        content = (destination / "relax.in").read_text()
        assert "nstep            = 123" in content

    def test_unknown_task_is_an_operator_error_not_a_silent_fallback(self) -> None:
        silicon = structure("Si.cif")

        completed = _run_cli("run", str(silicon), "--hpc", "scarf", "--task", "bands")

        assert completed.returncode == 2
        assert "Traceback" not in completed.stderr

    def test_run_with_out_publishes_a_complete_bundle(
        self, real_assets: None, tmp_path: Path
    ) -> None:
        silicon = structure("Si.cif")
        destination = tmp_path / "out"

        completed = _run_cli(
            "run", str(silicon), "--hpc", "scarf", "-o", str(destination)
        )

        assert completed.returncode == 0, completed.stderr
        assert (destination / ".complete").exists()
        assert (destination / "scf.in").exists()
        assert (destination / "submit.sh").exists()
        manifest = json.loads((destination / "goldilocks.json").read_text())
        assert manifest["records"]["functional"]["value"] == "PBEsol"
        assert any(
            warning["code"] == "job.walltime_defaulted"
            for warning in manifest["warnings"]
        )

    @pytest.mark.skipif(
        shutil.which("enum.x") is None and shutil.which("multienum.x") is None,
        reason="needs the enumlib executables (enum.x, makeStr.py) on PATH",
    )
    def test_run_afm_ordering_publishes_without_crashing(
        self, real_assets: None, tmp_path: Path
    ) -> None:
        """Regression for #27: an AFM-relabeled structure used to crash
        ``goldilocks run`` with an uncaught ``KeyError`` (write_qe_scf
        looked up QE species labels in a dict keyed by real elements) --
        this reproduces the exact failing case through the real CLI
        subprocess, not just the writer function in isolation."""
        rock_salt_feo = tmp_path / "FeO.cif"
        Structure(
            Lattice.cubic(4.3), ["Fe", "O"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        ).to(filename=str(rock_salt_feo))
        destination = tmp_path / "out"

        completed = _run_cli(
            "run",
            str(rock_salt_feo),
            "--hpc",
            "scarf",
            "--set",
            "magnetic_ordering=afm",
            "--set",
            "hubbard_needs_correlation=false",
            "-o",
            str(destination),
        )

        assert completed.returncode == 0, completed.stderr
        content = (destination / "scf.in").read_text()
        assert "ntyp             = 3" in content
        assert "  Fe1  " in content
        assert "  Fe2  " in content

    def test_run_default_magnetic_structure_publishes_without_crashing(
        self, real_assets: None, tmp_path: Path
    ) -> None:
        """Regression for #32: the bundled Fe_bcc.cif example (like every
        structure loaded through pymatgen's own CIF reader) carries
        per-site labels ('Fe0'/'Fe1') distinct from its element symbol
        ('Fe') even with zero --set flags and no AFM relabeling -- this
        used to crash goldilocks run with an uncaught KeyError on the
        plain default heuristic path, a regression #27 introduced while
        fixing the AFM-specific case."""
        destination = tmp_path / "out"

        completed = _run_cli(
            "run",
            str(structure("Fe_bcc.cif")),
            "--hpc",
            "scarf",
            "-o",
            str(destination),
        )

        assert completed.returncode == 0, completed.stderr
        content = (destination / "scf.in").read_text()
        assert "starting_magnetization(1)" in content
        assert "starting_magnetization(2)" in content

    def test_run_json_output_is_stable_and_sorted(self, real_assets: None) -> None:
        silicon = structure("Si.cif")

        completed = _run_cli("run", str(silicon), "--hpc", "scarf", "--json")

        assert completed.returncode == 0, completed.stderr
        document = json.loads(completed.stdout)
        assert "scf.in" in document["files"]
        assert any(
            warning["code"] == "job.walltime_defaulted"
            for warning in document["warnings"]
        )
        # P2: sort_keys=True -- re-serializing must reproduce the same text.
        assert json.dumps(
            document, indent=2, sort_keys=True
        ) == completed.stdout.rstrip("\n")
