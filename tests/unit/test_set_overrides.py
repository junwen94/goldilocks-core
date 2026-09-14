"""Tests for goldilocks_core.set_overrides (v2 epic 8, #8)."""

from __future__ import annotations

import pytest

from goldilocks_core.set_overrides import (
    InvalidSetting,
    build_overrides,
    build_overrides_from_cli,
    coerce_cli_assignments,
    coerce_cli_value,
    parse_set_flags,
)


class TestParseSetFlags:
    def test_splits_key_equals_value(self) -> None:
        assert parse_set_flags(["functional=PBE"]) == {"functional": "PBE"}

    def test_multiple_flags(self) -> None:
        result = parse_set_flags(["functional=PBE", "ecutwfc_ry=30"])

        assert result == {"functional": "PBE", "ecutwfc_ry": "30"}

    def test_value_may_itself_contain_an_equals_sign(self) -> None:
        result = parse_set_flags(['u_by_element={"Fe": 5.3}'])

        assert result == {"u_by_element": '{"Fe": 5.3}'}

    def test_repeating_the_same_key_with_the_same_value_is_fine(self) -> None:
        result = parse_set_flags(["functional=PBE", "functional=PBE"])

        assert result == {"functional": "PBE"}

    def test_repeating_the_same_key_with_a_different_value_raises(self) -> None:
        with pytest.raises(InvalidSetting, match="given twice"):
            parse_set_flags(["functional=PBE", "functional=LDA"])

    def test_missing_equals_sign_raises(self) -> None:
        with pytest.raises(InvalidSetting, match="KEY=VALUE"):
            parse_set_flags(["functional"])

    def test_empty_key_raises(self) -> None:
        with pytest.raises(InvalidSetting, match="empty key"):
            parse_set_flags(["=PBE"])

    def test_steps_prefix_is_stripped_for_a_real_step(self) -> None:
        result = parse_set_flags(["steps.scf.k_distance=0.12"])

        assert result == {"k_distance": "0.12"}

    def test_steps_prefix_rejects_an_unknown_step(self) -> None:
        with pytest.raises(InvalidSetting, match="unknown step 'bands'"):
            parse_set_flags(["steps.bands.k_distance=0.12"])

    def test_steps_nscf_prefix_is_accepted_now_that_dos_has_a_real_nscf_step(
        self,
    ) -> None:
        """Regression for #33 (v2 epic 9, #9): nscf used to be rejected
        as an unknown step even though dos.x's task genuinely has one --
        still not scoped to just that step (see set_overrides.py's own
        docstring), but no longer an outright rejection."""
        result = parse_set_flags(["steps.nscf.k_distance=0.12"])

        assert result == {"k_distance": "0.12"}

    def test_steps_prefix_rejects_a_malformed_path(self) -> None:
        with pytest.raises(InvalidSetting, match="malformed"):
            parse_set_flags(["steps.scf=0.12"])


class TestCoerceCliValue:
    def _binding(self, key: str):
        from goldilocks_core.capabilities import bindings

        return bindings()[key]

    @pytest.mark.parametrize("raw", ["true", "True", "1", "yes", "on"])
    def test_boolean_true_spellings(self, raw: str) -> None:
        assert coerce_cli_value(self._binding("nosym"), "nosym", raw) is True

    @pytest.mark.parametrize("raw", ["false", "False", "0", "no", "off"])
    def test_boolean_false_spellings(self, raw: str) -> None:
        assert coerce_cli_value(self._binding("nosym"), "nosym", raw) is False

    def test_invalid_boolean_raises(self) -> None:
        with pytest.raises(InvalidSetting, match="not a boolean"):
            coerce_cli_value(self._binding("nosym"), "nosym", "maybe")

    def test_integer(self) -> None:
        assert coerce_cli_value(self._binding("nbnd"), "nbnd", "12") == 12

    def test_invalid_integer_raises(self) -> None:
        with pytest.raises(InvalidSetting, match="not an integer"):
            coerce_cli_value(self._binding("nbnd"), "nbnd", "12.5")

    def test_number(self) -> None:
        assert (
            coerce_cli_value(self._binding("ecutwfc_ry"), "ecutwfc_ry", "30.5") == 30.5
        )

    def test_invalid_number_raises(self) -> None:
        with pytest.raises(InvalidSetting, match="not a number"):
            coerce_cli_value(self._binding("ecutwfc_ry"), "ecutwfc_ry", "abc")

    def test_array_is_parsed_as_json(self) -> None:
        assert coerce_cli_value(self._binding("k_grid"), "k_grid", "[4, 4, 4]") == [
            4,
            4,
            4,
        ]

    def test_object_is_parsed_as_json(self) -> None:
        result = coerce_cli_value(
            self._binding("u_by_element"), "u_by_element", '{"Fe": 5.3}'
        )

        assert result == {"Fe": 5.3}

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(InvalidSetting, match="not valid JSON"):
            coerce_cli_value(self._binding("k_grid"), "k_grid", "not json")

    def test_string_passes_through(self) -> None:
        assert (
            coerce_cli_value(self._binding("functional"), "functional", "PBE") == "PBE"
        )


class TestCoerceCliAssignments:
    def test_unknown_key_raises_with_did_you_mean(self) -> None:
        with pytest.raises(InvalidSetting, match="did you mean: ecutwfc_ry"):
            coerce_cli_assignments({"ecutwf_ry": "30"})

    def test_unknown_key_with_no_close_match_has_a_generic_message(self) -> None:
        with pytest.raises(InvalidSetting, match="goldilocks settings"):
            coerce_cli_assignments({"totally_unrelated_nonsense_key": "1"})


class TestBuildOverrides:
    def test_system_level_setting_lands_in_the_right_human_input(self) -> None:
        overrides = build_overrides({"functional": "PBE"})

        assert overrides.system.functional.functional == "PBE"

    def test_plain_system_field_needs_no_human_input_wrapper(self) -> None:
        overrides = build_overrides({"pseudo_table_id": "sssp-fixture"})

        assert overrides.system.pseudo_table_id == "sssp-fixture"

    def test_two_fields_of_the_same_human_input_class_merge(self) -> None:
        overrides = build_overrides({"ecutwfc_ry": 30.0, "ecutrho_ry": 120.0})

        assert overrides.system.cutoffs.ecutwfc_ry == 30.0
        assert overrides.system.cutoffs.ecutrho_ry == 120.0

    def test_kpoints_setting_lands_under_step_kpoints(self) -> None:
        overrides = build_overrides({"k_distance": 0.12})

        assert overrides.step.kpoints.k_sampling.k_distance == 0.12

    def test_resources_setting_lands_under_step_resources(self) -> None:
        overrides = build_overrides({"partition": "scarf"})

        assert overrides.step.resources.job.partition == "scarf"

    def test_hubbard_needs_correlation_key_maps_back_to_the_real_field(self) -> None:
        overrides = build_overrides({"hubbard_needs_correlation": True})

        assert overrides.system.hubbard.needs_correlation is True

    def test_analysis_fact_settings_land_under_analysis(self) -> None:
        overrides = build_overrides(
            {
                "is_metal": True,
                "is_magnetic": False,
                "needs_soc": True,
                "needs_correlation": True,
            }
        )

        assert overrides.analysis.is_metal.is_metal is True
        assert overrides.analysis.is_magnetic.is_magnetic is False
        assert overrides.analysis.needs_soc.needs_soc is True
        assert overrides.analysis.needs_correlation.needs_correlation is True

    def test_needs_correlation_and_hubbard_needs_correlation_are_independent(
        self,
    ) -> None:
        """The two ``needs_correlation``-named fields (the general analysis
        fact and hubbard's own escalation override) share an inner Python
        field name but must not collide as `--set` keys -- see
        ``capabilities.py``'s own docstring on this exact collision."""
        overrides = build_overrides(
            {"needs_correlation": True, "hubbard_needs_correlation": False}
        )

        assert overrides.analysis.needs_correlation.needs_correlation is True
        assert overrides.system.hubbard.needs_correlation is False

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(InvalidSetting):
            build_overrides({"not_a_real_setting": 1})

    def test_enum_violation_raises_invalid_setting_not_a_raw_pydantic_error(
        self,
    ) -> None:
        with pytest.raises(InvalidSetting, match="occupations"):
            build_overrides({"occupations": "bogus"})

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("npool", 0),
            ("npool", -1),
            ("ndiag", 0),
            ("nodes", 0),
            ("ntasks", 0),
            ("walltime_h", -5.0),
            ("nbnd", -5),
            ("degauss", -0.5),
            ("smearing_type", "totally-bogus-value"),
        ],
    )
    def test_physically_invalid_override_raises_invalid_setting_not_a_crash(
        self, key: str, value: object
    ) -> None:
        """Regression for #35 (v2 epic 9, #9): each of these used to
        either be accepted silently or crash downstream with an
        unrelated Python exception (ZeroDivisionError, a negative
        math.isqrt argument, ...) instead of a clean InvalidSetting at
        the same boundary an unknown key or a bad enum member already
        gets."""
        with pytest.raises(InvalidSetting):
            build_overrides({key: value})

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("k_distance", 0.0),
            ("k_distance", -0.3),
            ("k_grid", [0, 4, 4]),
            ("k_grid", [-1, 4, 4]),
            ("shift", [5, 5, 5]),
            ("shift", [-1, 0, 1]),
        ],
    )
    def test_k_sampling_invalid_override_raises_invalid_setting_not_a_crash(
        self, key: str, value: object
    ) -> None:
        """Regression for #48: each of these used to be accepted
        silently (k_grid/shift, written verbatim into the K_POINTS
        card) or crash downstream with a raw ZeroDivisionError
        (k_distance<=0), instead of a clean InvalidSetting -- the one
        advisor #35's sweep never touched."""
        with pytest.raises(InvalidSetting):
            build_overrides({key: value})

    def test_empty_assignments_gives_all_defaults(self) -> None:
        overrides = build_overrides({})

        assert overrides.analysis.is_metal is None
        assert overrides.system.functional is None
        assert overrides.step.kpoints.occupations is None
        assert overrides.step.resources.job is None


class TestBuildOverridesFromCli:
    def test_full_pipeline_from_raw_set_flags(self) -> None:
        overrides = build_overrides_from_cli(
            ["functional=PBE", "ecutwfc_ry=30.5", "k_grid=[4,4,4]", "nosym=true"]
        )

        assert overrides.system.functional.functional == "PBE"
        assert overrides.system.cutoffs.ecutwfc_ry == 30.5
        assert overrides.step.kpoints.k_sampling.k_grid == (4, 4, 4)
        assert overrides.step.kpoints.n_irr_k.nosym is True

    def test_steps_scf_prefix_lands_the_same_as_flat(self) -> None:
        flat = build_overrides_from_cli(["k_distance=0.12"])
        scoped = build_overrides_from_cli(["steps.scf.k_distance=0.12"])

        assert flat.step.kpoints.k_sampling == scoped.step.kpoints.k_sampling

    def test_typo_surfaces_as_invalid_setting(self) -> None:
        with pytest.raises(InvalidSetting):
            build_overrides_from_cli(["ecutwf_ry=30"])
