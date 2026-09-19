"""Tests for goldilocks_core.capabilities (v2 epic 8, #8).

Named ``test_capabilities_v2.py``, not ``test_capabilities.py``: that
name is already taken by v1's ``runtime.capabilities`` tests, which
this module does not replace (v2 epic 9 deletes the v1 tree, not this
one).
"""

from __future__ import annotations

import json
import typing
from importlib.metadata import version as package_version

from goldilocks_core.capabilities import (
    SOURCES,
    VOCABULARY_VERSION,
    _json_type,
    _unwrap_optional,
    capabilities,
)
from goldilocks_core.service import (
    KpointsOverrides,
    RelaxOverrides,
    ResourceOverrides,
    SystemOverrides,
)


def _expected_setting_field_count() -> int:
    """Independently recomputes how many settings should exist, by
    walking the same composing classes capabilities.py reflects
    over -- a completeness guard that fails loudly if the walker ever
    silently drops a field, without hardcoding a magic number that
    would need editing by hand every time an advisor gains a field."""
    total = 0
    for cls in (SystemOverrides, KpointsOverrides, ResourceOverrides, RelaxOverrides):
        hints = typing.get_type_hints(cls)
        for name, annotation in hints.items():
            if name.endswith("_llm"):
                continue
            inner = _unwrap_optional(annotation)
            if isinstance(inner, type) and hasattr(inner, "model_fields"):
                total += len(inner.model_fields)
            else:
                total += 1
    return total


class TestTopLevelShape:
    def test_has_every_design_doc_key(self) -> None:
        caps = capabilities()

        for key in (
            "core_version",
            "vocabulary_version",
            "codes",
            "tasks",
            "facts",
            "settings",
            "pseudopotential_tables",
            "hpc_profiles",
            "models",
            "warnings",
            "sources",
        ):
            assert key in caps

    def test_core_version_matches_installed_package(self) -> None:
        assert capabilities()["core_version"] == package_version("goldilocks-core")

    def test_vocabulary_version_is_the_module_constant(self) -> None:
        assert capabilities()["vocabulary_version"] == VOCABULARY_VERSION == "1"

    def test_sources_is_the_closed_priority_order(self) -> None:
        assert (
            capabilities()["sources"]
            == list(SOURCES)
            == [
                "human",
                "ml",
                "llm",
                "heuristic",
            ]
        )

    def test_is_fully_json_serializable(self) -> None:
        """Regression guard for the exact bug module 1's bundle.py hit:
        a stray non-JSON-safe object (e.g. a dataclass or Structure)
        anywhere in this tree would blow up json.dumps."""
        json.dumps(capabilities())

    def test_models_lists_every_registered_model(self) -> None:
        """v2 epic 11 (#11): models[] reflects ml/registry.toml, not
        fabricated -- registered, not necessarily installed (approaches
        below is the honest "is it actually usable right now" signal)."""
        ids = {model["id"] for model in capabilities()["models"]}
        assert ids == {
            "models/qrf-kpoints",
            "models/metallicity-cgcnn",
            "models/is-metal-classifier",
            "models/is-magnetic-classifier",
        }

    def test_warnings_catalogue_is_populated_from_every_advisor(self) -> None:
        caps = capabilities()

        assert caps["warnings"]
        for entry in caps["warnings"]:
            assert entry.keys() == {"code", "level", "category", "message"}
            assert entry["level"] in {"info", "warning", "error"}

    def test_warnings_catalogue_codes_are_unique(self) -> None:
        codes = [entry["code"] for entry in capabilities()["warnings"]]

        assert len(codes) == len(set(codes))

    def test_codes_and_tasks_reflect_only_what_actually_runs(self) -> None:
        """v2 epic 9 (#9, #28): ``dos`` joined ``scf_single_point`` once
        the delivery layers actually routed it, not before -- this list
        names every task a caller can request today, no more, no less.
        ``relax``/``vc-relax`` joined the same way in v2 epic 10 (#10)."""
        caps = capabilities()
        assert caps["codes"] == [
            {
                "id": "quantum_espresso",
                "name": "Quantum ESPRESSO",
                "tasks": ["scf_single_point", "dos", "relax", "vc-relax"],
            }
        ]
        assert len(caps["tasks"]) == 4
        by_id = {task["id"]: task for task in caps["tasks"]}
        assert by_id["scf_single_point"]["executables"] == ["pw.x"]
        assert by_id["scf_single_point"]["step_count"] == 1
        assert by_id["dos"]["executables"] == ["pw.x", "pw.x", "dos.x"]
        assert by_id["dos"]["step_count"] == 3
        assert by_id["relax"]["executables"] == ["pw.x"]
        assert by_id["relax"]["step_count"] == 1
        assert by_id["vc-relax"]["executables"] == ["pw.x"]
        assert by_id["vc-relax"]["step_count"] == 1


class TestSettingsCompleteness:
    def test_every_override_field_has_exactly_one_setting(self) -> None:
        caps = capabilities()

        assert len(caps["settings"]) == _expected_setting_field_count()

    def test_no_duplicate_keys(self) -> None:
        keys = [s["key"] for s in capabilities()["settings"]]

        assert len(keys) == len(set(keys))

    def test_every_setting_has_a_non_empty_description(self) -> None:
        """Every setting must be documented in _SETTING_META -- an empty
        description means a new advisor field was added and nobody
        wrote its capabilities metadata yet."""
        for setting in capabilities()["settings"]:
            assert setting["description"], setting["key"]

    def test_hubbard_needs_correlation_is_renamed_to_avoid_the_fact_collision(
        self,
    ) -> None:
        keys = {s["key"] for s in capabilities()["settings"]}

        assert "hubbard_needs_correlation" in keys
        assert "needs_correlation" not in keys  # only the fact owns this bare name

    def test_system_level_settings_apply_to_all_programs(self) -> None:
        settings = {s["key"]: s for s in capabilities()["settings"]}

        assert settings["functional"]["scope"] == "system"
        assert settings["functional"]["programs"] is None
        assert settings["ecutwfc_ry"]["scope"] == "system"

    def test_per_step_settings_are_scoped_to_pw_x(self) -> None:
        settings = {s["key"]: s for s in capabilities()["settings"]}

        for key in ("occupations", "k_grid", "nbnd", "npool"):
            assert settings[key]["scope"] == "per_step"
            assert settings[key]["programs"] == ["pw.x"]

    def test_functional_carries_enum_from_and_default(self) -> None:
        settings = {s["key"]: s for s in capabilities()["settings"]}

        expected = "pseudopotential_tables.functional"
        assert settings["functional"]["enum_from"] == expected
        assert settings["functional"]["default"] == "PBEsol"

    def test_occupations_enum_matches_the_literal_type(self) -> None:
        settings = {s["key"]: s for s in capabilities()["settings"]}

        assert settings["occupations"]["type"] == "string"
        assert set(settings["occupations"]["enum"]) == {
            "fixed",
            "smearing",
            "tetrahedra_opt",
        }

    def test_k_grid_is_a_fixed_length_integer_array(self) -> None:
        settings = {s["key"]: s for s in capabilities()["settings"]}

        assert settings["k_grid"]["type"] == "array"
        assert settings["k_grid"]["minItems"] == 3
        assert settings["k_grid"]["maxItems"] == 3

    def test_k_distance_declares_its_ml_target(self) -> None:
        settings = {s["key"]: s for s in capabilities()["settings"]}

        assert settings["k_distance"]["ml_target"] == "k_distance"

    def test_approaches_never_includes_ml_since_no_model_is_installed(self) -> None:
        """Design point (1)-b: ml_target is a static declaration,
        approaches is runtime-computed -- k_index declares an ml_target
        (a real, published goldilocks-ml ladder-rung model, #90) but
        must not claim "ml" is usable while that model stays
        registered-but-not-wired."""
        settings = {s["key"]: s for s in capabilities()["settings"]}

        assert settings["k_index"]["approaches"] == ["human", "heuristic"]
        assert settings["functional"]["approaches"] == ["human", "heuristic"]

    def test_k_distance_approaches_gains_ml_once_qrf95_is_installed(
        self, real_assets
    ) -> None:
        """v2 epic 11 follow-up (#92): QRF95 predates ML_CLASSIFIER_ROLES
        and is checked by its own branch in capabilities._ml_model_installed
        -- this exercises that branch is honestly wired end to end, not
        just is_metal/is_magnetic's shared table."""
        settings = {s["key"]: s for s in capabilities()["settings"]}

        assert settings["k_distance"]["approaches"] == ["human", "ml", "heuristic"]

    def test_units_are_present_where_physically_meaningful(self) -> None:
        settings = {s["key"]: s for s in capabilities()["settings"]}

        assert settings["ecutwfc_ry"]["unit"] == "Ry"
        assert settings["walltime_h"]["unit"] == "hours"
        assert settings["u_by_element"]["unit"] == "eV"
        assert settings["npool"]["unit"] is None


class TestFacts:
    def test_exactly_the_four_scalar_overridable_facts(self) -> None:
        keys = {f["key"] for f in capabilities()["facts"]}

        assert keys == {"is_metal", "is_magnetic", "needs_soc", "needs_correlation"}

    def test_is_metal_values_match_the_real_metallicity_type(self) -> None:
        facts = {f["key"]: f for f in capabilities()["facts"]}

        assert set(facts["is_metal"]["values"]) == {"metal", "non_metal"}
        assert facts["is_metal"]["ml_target"] == "is_metal"

    def test_is_metal_approaches_gains_ml_once_its_model_is_installed(
        self, real_assets
    ) -> None:
        """v2 epic 11 (#11): approaches is the honest "is it actually
        usable right now" signal, checked per asset -- real_assets is
        this test suite's own existing "is the default profile actually
        installed on this machine" gate."""
        facts = {f["key"]: f for f in capabilities()["facts"]}

        assert facts["is_metal"]["approaches"] == ["human", "ml", "heuristic"]

    def test_all_facts_are_overridable(self) -> None:
        for fact in capabilities()["facts"]:
            assert fact["overridable"] is True

    def test_every_fact_marked_overridable_actually_has_a_set_binding(self) -> None:
        """`overridable: True` is a promise the settings-schema reflection
        makes about `--set`/override support, not just a display flag --
        this once shipped True for all four facts while `bindings()` (the
        thing `--set` actually consults) had no entry for any of them, so
        `--set is_metal=true` failed with "unknown setting". Every
        overridable fact's key must resolve in `bindings()`."""
        from goldilocks_core.capabilities import bindings

        catalogue = bindings()
        for fact in capabilities()["facts"]:
            if fact["overridable"]:
                assert fact["key"] in catalogue, (
                    f"{fact['key']!r} claims overridable=True but has no --set binding"
                )


class TestPseudopotentialTablesAndHpcProfiles:
    def test_pseudopotential_tables_are_populated_from_the_real_registry(self) -> None:
        tables = capabilities()["pseudopotential_tables"]

        assert tables
        assert all("id" in t and "elements" in t for t in tables)

    def test_hpc_profiles_include_scarf(self) -> None:
        profiles = capabilities()["hpc_profiles"]

        assert any(p["id"] == "scarf" for p in profiles)
        scarf = next(p for p in profiles if p["id"] == "scarf")
        assert scarf["scheduler"] == "slurm"
        assert "scarf" in scarf["partitions"]


class TestTypeHelpers:
    def test_unwrap_optional_strips_none(self) -> None:
        assert _unwrap_optional(int | None) is int
        assert _unwrap_optional(str) is str

    def test_json_type_maps_python_primitives(self) -> None:
        assert _json_type(bool) == {"type": "boolean"}
        assert _json_type(int) == {"type": "integer"}
        assert _json_type(float) == {"type": "number"}
        assert _json_type(str) == {"type": "string"}

    def test_json_type_maps_literal_to_string_enum(self) -> None:
        result = _json_type(typing.Literal["a", "b"] | None)

        assert result == {"type": "string", "enum": ["a", "b"]}

    def test_json_type_maps_dict_to_object(self) -> None:
        assert _json_type(dict[str, float] | None) == {"type": "object"}

    def test_json_type_maps_fixed_tuple_to_array(self) -> None:
        result = _json_type(tuple[int, int, int] | None)

        assert result == {
            "type": "array",
            "items": "integer",
            "minItems": 3,
            "maxItems": 3,
        }
