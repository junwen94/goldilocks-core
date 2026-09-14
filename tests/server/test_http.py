"""Tests for goldilocks_core.server.http (v2 epic 8, #8).

Deliberately thin on business-logic scenarios (test_handlers.py already
covers those, transport-agnostically) -- this file is about what's
actually HTTP-specific: status codes, the error envelope shape, routing,
and the archive response mode.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from goldilocks_core.server.http import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


class TestOperationalRoutes:
    def test_health(self, client: TestClient) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_ready_matches_the_readiness_checker_directly(
        self, client: TestClient
    ) -> None:
        """Not gated on ``real_assets``: ``/ready`` checks the *workbench*
        profile (every pseudopotential table *and* every ml model), a
        stricter bar than the ``default`` profile ``real_assets``
        guarantees -- this machine has the pseudopotentials but was
        deliberately not given the ml models, so asserting a specific
        ready/not-ready value here would depend on machine state this
        test doesn't control. Assert the response is *consistent* and
        well-shaped instead."""
        from goldilocks_core.server._handlers import build_readiness

        expected = build_readiness().check()

        response = client.get("/ready")

        if expected.ready:
            assert response.status_code == 200
            assert response.json() == {
                "status": "ready",
                "asset_count": expected.asset_count,
            }
        else:
            assert response.status_code == 503
            error = response.json()["error"]
            assert error["kind"] == "assets_unavailable"
            assert error["details"]["asset_id"] == expected.asset_id


class TestOpenAPISchema:
    def test_openapi_builds_without_raising(self, client: TestClient) -> None:
        """Regression test for #31: the ``/run`` route's ``-> Response``
        return annotation is a string under ``from __future__ import
        annotations``, and FastAPI/pydantic resolve it against
        ``server/http.py``'s module globals when building the schema --
        not against whatever was imported inside ``create_app()``. This
        is also what ``scripts/export_workbench_openapi.py`` runs during
        the Docker build stage, so a regression here breaks the image."""
        schema = client.app.openapi()

        assert set(schema["paths"]) == {
            "/health",
            "/ready",
            "/capabilities",
            "/inspect",
            "/explain",
            "/run",
        }


class TestCapabilities:
    def test_returns_the_same_shape_capabilities_py_builds(
        self, client: TestClient
    ) -> None:
        from goldilocks_core.capabilities import capabilities

        response = client.get("/capabilities")

        assert response.status_code == 200
        assert response.json() == capabilities()


class TestInspect:
    def test_returns_200_for_real_structure_text(
        self, client: TestClient, silicon_cif: str
    ) -> None:
        response = client.post("/inspect", json={"structure_content": silicon_cif})

        assert response.status_code == 200
        assert response.json()["structure"]["reduced_formula"] == "Si"

    def test_rejects_a_path_shaped_value_as_422(self, client: TestClient) -> None:
        response = client.post("/inspect", json={"structure_content": "/etc/passwd"})

        assert response.status_code == 422
        body = response.json()
        assert body["error"]["kind"] == "invalid_request"
        # No raw exception object anywhere in the sanitized body.
        json.dumps(body)


class TestExplain:
    def test_returns_records_and_warnings(
        self, client: TestClient, real_assets, silicon_cif: str
    ) -> None:
        response = client.post(
            "/explain", json={"structure_content": silicon_cif, "hpc": "scarf"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["records"]["functional"]["value"] == "PBEsol"
        assert "warnings" in body

    def test_dos_task_returns_nscf_prefixed_records_and_a_dos_record(
        self, client: TestClient, real_assets, silicon_cif: str
    ) -> None:
        """v2 epic 9 (#9, #28): task='dos' used to be silently accepted
        and ignored -- this confirms it actually routes to advise_dos,
        not the scf-only pipeline, through the real HTTP transport."""
        response = client.post(
            "/explain",
            json={"structure_content": silicon_cif, "hpc": "scarf", "task": "dos"},
        )

        assert response.status_code == 200
        records = response.json()["records"]
        assert records["occupations"]["value"]["occupations"] == "smearing"
        assert records["nscf_occupations"]["value"]["occupations"] == ("tetrahedra_opt")
        assert records["dos"]["value"]["delta_e"] == 0.01

    def test_relax_task_returns_a_relax_record(
        self, client: TestClient, real_assets, silicon_cif: str
    ) -> None:
        """v2 epic 10 (#10): confirms task='relax' actually routes to
        advise_relax through the real HTTP transport, the same
        end-to-end check #28 added for task='dos'."""
        response = client.post(
            "/explain",
            json={"structure_content": silicon_cif, "hpc": "scarf", "task": "relax"},
        )

        assert response.status_code == 200
        records = response.json()["records"]
        assert records["relax"]["value"]["ion_dynamics"] == "bfgs"

    def test_unknown_task_is_a_422_not_a_silent_fallback(
        self, client: TestClient, silicon_cif: str
    ) -> None:
        response = client.post(
            "/explain",
            json={"structure_content": silicon_cif, "hpc": "scarf", "task": "bands"},
        )

        assert response.status_code == 422


class TestRun:
    def test_json_mode_lists_the_published_files(
        self, client: TestClient, real_assets, silicon_cif: str
    ) -> None:
        response = client.post(
            "/run", json={"structure_content": silicon_cif, "hpc": "scarf"}
        )

        assert response.status_code == 200
        body = response.json()
        assert "scf.in" in body["files"]
        assert "submit.sh" in body["files"]

    def test_archive_mode_returns_a_real_zip(
        self, client: TestClient, real_assets, silicon_cif: str
    ) -> None:
        response = client.post(
            "/run",
            json={
                "structure_content": silicon_cif,
                "hpc": "scarf",
                "respond_with": "archive",
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

    def test_dos_task_generates_all_three_steps(
        self, client: TestClient, real_assets, silicon_cif: str
    ) -> None:
        response = client.post(
            "/run",
            json={"structure_content": silicon_cif, "hpc": "scarf", "task": "dos"},
        )

        assert response.status_code == 200
        files = response.json()["files"]
        assert "scf.in" in files
        assert "nscf.in" in files
        assert "dos.in" in files
        assert "submit.sh" in files

    def test_relax_task_generates_relax_in(
        self, client: TestClient, real_assets, silicon_cif: str
    ) -> None:
        response = client.post(
            "/run",
            json={"structure_content": silicon_cif, "hpc": "scarf", "task": "relax"},
        )

        assert response.status_code == 200
        files = response.json()["files"]
        assert "relax.in" in files
        assert "submit.sh" in files

    def test_vc_relax_task_generates_vc_relax_in_with_a_set_override(
        self, client: TestClient, real_assets, silicon_cif: str
    ) -> None:
        response = client.post(
            "/run",
            json={
                "structure_content": silicon_cif,
                "hpc": "scarf",
                "task": "vc-relax",
                "overrides": {"cell_factor": 3.0},
            },
        )

        assert response.status_code == 200
        files = response.json()["files"]
        assert "vc-relax.in" in files

    def test_unknown_setting_is_a_422_with_did_you_mean(
        self, client: TestClient, real_assets, silicon_cif: str
    ) -> None:
        response = client.post(
            "/run",
            json={
                "structure_content": silicon_cif,
                "hpc": "scarf",
                "overrides": {"ecutwf_ry": 30},
            },
        )

        assert response.status_code == 422
        assert "did you mean: ecutwfc_ry" in response.json()["error"]["message"]

    def test_blocked_pipeline_is_a_422_with_deduplicated_message(
        self, client: TestClient, silicon_cif: str
    ) -> None:
        response = client.post(
            "/run",
            json={
                "structure_content": silicon_cif,
                "hpc": "scarf",
                "overrides": {"pseudo_table_id": "does-not-exist"},
            },
        )

        assert response.status_code == 422
        message = response.json()["error"]["message"]
        assert message.count("unknown pseudopotential table") == 1

    def test_unknown_hpc_profile_is_a_422(
        self, client: TestClient, silicon_cif: str
    ) -> None:
        response = client.post(
            "/run",
            json={"structure_content": silicon_cif, "hpc": "does-not-exist"},
        )

        assert response.status_code == 422
        assert response.json()["error"]["kind"] == "invalid_hpc_profile"

    def test_rejects_unknown_top_level_fields(
        self, client: TestClient, silicon_cif: str
    ) -> None:
        response = client.post(
            "/run",
            json={"structure_content": silicon_cif, "structure_path": "/tmp/x.cif"},
        )

        assert response.status_code == 422
