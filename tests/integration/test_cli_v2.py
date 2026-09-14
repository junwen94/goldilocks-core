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

from goldilocks_core.examples.structures import structure


def test_help_lists_the_design_docs_commands() -> None:
    completed = _run_cli("--help")

    assert completed.returncode == 0, completed.stderr
    for command in (
        "run",
        "explain",
        "inspect",
        "settings",
        "models",
        "assets",
        "examples",
        "serve",
    ):
        assert command in completed.stdout


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


def test_models_list_is_honest_about_having_none_yet() -> None:
    completed = _run_cli("models", "list")

    assert completed.returncode == 0, completed.stderr
    assert "no ml models installed" in completed.stdout


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
        assert "is_metal: unavailable:" in completed.stdout

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
