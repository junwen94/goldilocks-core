from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from goldilocks_core.bundle import (
    ArchiveOutput,
    BundleInput,
    DirectoryOutput,
    archive_bytes,
    bundle_files,
    is_complete,
    publish,
)
from goldilocks_core.resolution import Blocked, Provenance, Resolved, Unavailable


def _artifact(path: str, content: bytes, role: str = "input") -> dict:
    return {"path": path, "role": role, "content": content}


def _bundle_input(**overrides) -> BundleInput:
    defaults = {
        "artifacts": (
            _artifact("source/original.cif", b"cif content", "structure_source"),
            _artifact("inputs/scf.in", b"&CONTROL\n/\n", "input"),
        ),
        "records": {
            "functional": Resolved("PBEsol", Provenance(source="heuristic")),
        },
        "citations": ("A citation.",),
    }
    defaults.update(overrides)
    return BundleInput(**defaults)


def test_publish_directory_preserves_content_and_is_complete(tmp_path: Path) -> None:
    bundle_input = _bundle_input()
    destination = tmp_path / "ready"

    publication = publish(bundle_input, DirectoryOutput(destination))

    files = {item["path"]: item["content"] for item in bundle_files(bundle_input)}
    assert set(files) == {
        "source/original.cif",
        "inputs/scf.in",
        "CITATIONS.md",
        "README.md",
        "goldilocks.json",
    }
    on_disk = {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file() and path.name != ".complete"
    }
    assert on_disk == files
    assert publication["kind"] == "directory"
    assert publication["path"] == str(destination.resolve())
    assert set(publication["files"]) == set(files)
    assert (destination / ".complete").is_file()
    assert is_complete(destination) is True


def test_manifest_hashes_match_the_exact_bytes_written(tmp_path: Path) -> None:
    bundle_input = _bundle_input()
    files = {item["path"]: item["content"] for item in bundle_files(bundle_input)}
    manifest = json.loads(files["goldilocks.json"])

    assert set(manifest["files"]) == set(files) - {"goldilocks.json"}
    for path, descriptor in manifest["files"].items():
        assert descriptor["sha256"] == hashlib.sha256(files[path]).hexdigest()
        assert descriptor["size_bytes"] == len(files[path])
    assert manifest["files"]["source/original.cif"]["role"] == "structure_source"
    assert manifest["citations"] == ["A citation."]


def test_manifest_records_are_tri_state_aware(tmp_path: Path) -> None:
    bundle_input = _bundle_input(
        records={
            "functional": Resolved("PBEsol", Provenance(source="human")),
            "hubbard": Unavailable(reason="no correlated elements"),
            "vdw": Blocked(by="geometry blocked upstream"),
        }
    )

    files = {item["path"]: item["content"] for item in bundle_files(bundle_input)}
    manifest = json.loads(files["goldilocks.json"])
    records = manifest["records"]

    assert records["functional"] == {
        "status": "resolved",
        "value": "PBEsol",
        "source": "human",
        "reason": None,
        "blocked_by": None,
    }
    assert records["hubbard"]["status"] == "unavailable"
    assert records["hubbard"]["reason"] == "no correlated elements"
    assert records["vdw"]["status"] == "blocked"
    assert records["vdw"]["blocked_by"] == "geometry blocked upstream"


def test_manifest_carries_a_top_level_warnings_array(tmp_path: Path) -> None:
    """v2 epic 8's "warnings array in every ... response" requirement
    extends to the published ``goldilocks.json`` manifest, not just
    HTTP/MCP/CLI transport responses -- flattened from every record's
    own structured warnings, the same way ``service.Advice.warnings()``
    does for a live pipeline run."""
    bundle_input = _bundle_input(
        records={
            "functional": Resolved("PBEsol", Provenance(source="human")),
            "job": Resolved(
                {
                    "partition": "scarf",
                    "warnings": [
                        {
                            "code": "job.walltime_defaulted",
                            "level": "warning",
                            "category": "job",
                            "message": "walltime not specified.",
                        }
                    ],
                },
                Provenance(source="heuristic"),
            ),
        }
    )

    files = {item["path"]: item["content"] for item in bundle_files(bundle_input)}
    manifest = json.loads(files["goldilocks.json"])

    assert manifest["warnings"] == [
        {
            "code": "job.walltime_defaulted",
            "level": "warning",
            "category": "job",
            "message": "walltime not specified.",
        }
    ]


def test_archive_bytes_match_directory_publication(tmp_path: Path) -> None:
    bundle_input = _bundle_input()
    directory = tmp_path / "dir-out"
    archive_path = tmp_path / "out.zip"

    publish(bundle_input, DirectoryOutput(directory))
    publish(bundle_input, ArchiveOutput(archive_path))

    files = {item["path"]: item["content"] for item in bundle_files(bundle_input)}
    with zipfile.ZipFile(io.BytesIO(archive_path.read_bytes())) as archive:
        assert archive.namelist() == sorted(files)
        assert {name: archive.read(name) for name in archive.namelist()} == files
    assert archive_bytes(bundle_input) == archive_path.read_bytes()
    assert is_complete(archive_path) is True
    assert Path(f"{archive_path}.complete").is_file()


@pytest.mark.parametrize("output_type", [DirectoryOutput, ArchiveOutput])
def test_publish_never_overwrites_an_existing_destination(
    tmp_path, output_type
) -> None:
    bundle_input = _bundle_input()
    destination = tmp_path / "target"

    publish(bundle_input, output_type(destination))
    with pytest.raises(FileExistsError):
        publish(bundle_input, output_type(destination))


@pytest.mark.parametrize("output_type", [DirectoryOutput, ArchiveOutput])
def test_a_raced_destination_is_preserved_not_clobbered(tmp_path, output_type) -> None:
    """No check-then-act window to race in the first place: os.mkdir()/
    O_EXCL either claims the name atomically or fails outright, so
    whatever got there first survives untouched."""
    bundle_input = _bundle_input()
    destination = tmp_path / "raced"
    if output_type is DirectoryOutput:
        destination.mkdir()
        (destination / "marker").write_text("concurrent owner")
    else:
        destination.write_bytes(b"concurrent owner")

    with pytest.raises(FileExistsError):
        publish(bundle_input, output_type(destination))

    if output_type is DirectoryOutput:
        assert (destination / "marker").read_text() == "concurrent owner"
    else:
        assert destination.read_bytes() == b"concurrent owner"


def test_write_failure_leaves_no_partial_directory_and_no_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import goldilocks_core.bundle as bundle_module

    bundle_input = _bundle_input()
    destination = tmp_path / "failed"

    def fail(root: Path, files) -> None:
        (root / files[0]["path"]).parent.mkdir(parents=True, exist_ok=True)
        (root / files[0]["path"]).write_bytes(files[0]["content"])
        raise OSError("disk full")

    monkeypatch.setattr(bundle_module, "_write_directory_path", fail)
    with pytest.raises(OSError, match="disk full"):
        publish(bundle_input, DirectoryOutput(destination))

    assert not destination.exists()


def test_write_failure_on_archive_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import goldilocks_core.bundle as bundle_module

    bundle_input = _bundle_input()
    destination = tmp_path / "failed.zip"

    def fail(files) -> bytes:
        raise OSError("disk full")

    monkeypatch.setattr(bundle_module, "_archive_bytes", fail)
    with pytest.raises(OSError, match="disk full"):
        publish(bundle_input, ArchiveOutput(destination))

    assert not destination.exists()
    assert not Path(f"{destination}.complete").exists()


def test_a_directory_with_files_but_no_marker_is_not_complete(tmp_path: Path) -> None:
    """The consumer-side contract this protocol relies on: simulates what
    a SIGKILL mid-publish would leave behind -- real bytes on disk, no
    completion marker -- and confirms it reads as untrustworthy."""
    destination = tmp_path / "killed-mid-write"
    destination.mkdir()
    (destination / "inputs").mkdir()
    (destination / "inputs" / "scf.in").write_text("&CONTROL\n/\n")

    assert is_complete(destination) is False


def test_missing_target_is_not_complete(tmp_path: Path) -> None:
    assert is_complete(tmp_path / "does-not-exist") is False
    assert is_complete(tmp_path / "does-not-exist.zip") is False


@pytest.mark.parametrize(
    "unsafe_path", ["/abs/path", "a/../b", "a\\b", "a:b", "", "./x"]
)
def test_unsafe_artifact_paths_are_rejected(unsafe_path: str) -> None:
    bundle_input = _bundle_input(artifacts=(_artifact(unsafe_path, b"x"),))

    with pytest.raises(ValueError, match="unsafe"):
        bundle_files(bundle_input)


def test_duplicate_artifact_paths_are_rejected() -> None:
    bundle_input = _bundle_input(
        artifacts=(
            _artifact("inputs/scf.in", b"a"),
            _artifact("inputs/scf.in", b"b"),
        )
    )

    with pytest.raises(ValueError, match="duplicate"):
        bundle_files(bundle_input)


def test_readme_lists_every_artifact_path_without_naming_a_specific_code() -> None:
    bundle_input = _bundle_input()

    files = {item["path"]: item["content"] for item in bundle_files(bundle_input)}
    readme = files["README.md"].decode("utf-8")

    assert "inputs/scf.in" in readme
    assert "source/original.cif" in readme
    assert "pw.x" not in readme
    assert "quantum" not in readme.lower()
