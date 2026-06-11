"""Smoke tests for `gl input` non-interactive command."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from goldilocks_core.cli.main import app

_runner = CliRunner()
_FE_CIF = Path(__file__).parent / "fixtures" / "structures" / "Fe_bcc.cif"


@pytest.mark.skipif(not _FE_CIF.exists(), reason="Fe_bcc.cif not found in tests/fixtures/structures/")
class TestGlInputSmoke:
    """gl input invocations against Fe_bcc.cif — only check exit code and key output tokens."""

    def test_basic_scf_exits_zero(self) -> None:
        result = _runner.invoke(app, ["input", "-s", str(_FE_CIF)])
        assert result.exit_code == 0, result.output

    def test_output_contains_formula(self) -> None:
        result = _runner.invoke(app, ["input", "-s", str(_FE_CIF)])
        assert "Fe" in result.output

    def test_output_contains_kmesh_section(self) -> None:
        result = _runner.invoke(app, ["input", "-s", str(_FE_CIF)])
        assert "K-mesh" in result.output

    def test_output_contains_spin_section(self) -> None:
        result = _runner.invoke(app, ["input", "-s", str(_FE_CIF)])
        assert "Spin" in result.output

    def test_relax_task_exits_zero(self) -> None:
        result = _runner.invoke(app, ["input", "-s", str(_FE_CIF), "-t", "relax"])
        assert result.exit_code == 0, result.output

    def test_explain_flag_shows_rationale(self) -> None:
        result = _runner.invoke(app, ["input", "-s", str(_FE_CIF), "-e"])
        assert result.exit_code == 0, result.output
        assert "Rationale" in result.output

    def test_collinear_hint(self) -> None:
        result = _runner.invoke(
            app, ["input", "-s", str(_FE_CIF), "-H", "spin_treatment=collinear"]
        )
        assert result.exit_code == 0, result.output
        assert "collinear" in result.output

    def test_invalid_task_exits_nonzero(self) -> None:
        result = _runner.invoke(app, ["input", "-s", str(_FE_CIF), "-t", "not_a_task"])
        assert result.exit_code != 0

    def test_invalid_vdw_method_exits_nonzero(self) -> None:
        result = _runner.invoke(
            app, ["input", "-s", str(_FE_CIF), "-H", "use_vdw=true", "-H", "vdw_method=bad"]
        )
        assert result.exit_code != 0

    def test_generate_output_creates_input_file(self, tmp_path: Path) -> None:
        result = _runner.invoke(
            app, ["input", "-s", str(_FE_CIF), "--output", str(tmp_path)]
        )
        assert result.exit_code == 0, result.output
        run_dirs = list(tmp_path.glob("run_*"))
        assert len(run_dirs) == 1
        input_files = list(run_dirs[0].glob("*.in"))
        assert len(input_files) >= 1
