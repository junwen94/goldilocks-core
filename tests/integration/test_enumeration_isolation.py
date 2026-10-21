"""Overlapping magnetic-ordering listings never share process-wide state
(#136).

pymatgen runs enumlib inside ``ScratchDir(".")``, which ``chdir``s the whole
process and writes ``struct_enum.in``/``vasp.*`` relative to the cwd. Run on
request threads, two listings moved each other's cwd, deleted each other's
files, and came back ferromagnetic-only. These tests run real enumerations,
so they need enumlib on PATH.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from pymatgen.core import Lattice, Structure
from support import run_cli

from goldilocks_core.advisors.magnetic_enumeration import enumlib_installed
from goldilocks_core.examples.structures import structure
from goldilocks_core.service._magnetic_orderings import (
    list_magnetic_orderings,
    report_to_json,
)

pytestmark = pytest.mark.skipif(
    not enumlib_installed(),
    reason="needs the enumlib executables (enum.x, makeStr.py) on PATH",
)

_STRUCTURES = {
    "Fe": Structure.from_file(structure("Fe_bcc.cif")),
    "FeO": Structure(
        Lattice.cubic(4.3), ["Fe", "O"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    ),
}
_ROUNDS = 3


def _listing(candidate: Structure) -> dict[str, Any]:
    return report_to_json(list_magnetic_orderings(candidate))


def _labels_and_warnings(
    listings: dict[str, dict[str, Any]],
) -> dict[str, tuple[list[str], list[str]]]:
    return {
        name: (
            [candidate["label"] for candidate in listing["candidates"]],
            [warning["message"] for warning in listing["warnings"]],
        )
        for name, listing in listings.items()
    }


@contextmanager
def _recorded_cwds() -> Iterator[set[str]]:
    """Every working directory the process is in while the block runs,
    sampled every millisecond from another thread."""
    seen: set[str] = set()
    done = threading.Event()

    def sample() -> None:
        while not done.wait(0.001):
            try:
                seen.add(os.getcwd())
            except FileNotFoundError:
                seen.add("<deleted>")

    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        yield seen
    finally:
        done.set()
        sampler.join()


def test_overlapping_listings_leave_the_cwd_alone_and_match_serial_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    launch_directory = os.getcwd()
    serial = {name: _listing(candidate) for name, candidate in _STRUCTURES.items()}
    assert all(len(listing["candidates"]) > 1 for listing in serial.values())

    with _recorded_cwds() as cwds, ThreadPoolExecutor(len(_STRUCTURES)) as pool:
        parallel = [
            dict(
                zip(_STRUCTURES, pool.map(_listing, _STRUCTURES.values()), strict=True)
            )
            for _ in range(_ROUNDS)
        ]

    assert [_labels_and_warnings(listings) for listings in parallel] == [
        _labels_and_warnings(serial)
    ] * _ROUNDS
    assert parallel == [serial] * _ROUNDS
    assert cwds == {launch_directory}
    assert list(tmp_path.iterdir()) == []


def test_an_enumerating_command_exits_once_it_is_done(tmp_path: Path) -> None:
    """The enumeration worker must never hold the CLI open after the command
    finishes (a process pool inside it once delayed exit by loky's 300 s
    idle timeout). A single enumeration cleans up its own scratch files, so
    stray files in the cwd are the overlapping-listings test's job."""
    completed = run_cli(
        "magnetic-orderings",
        str(structure("Fe_bcc.cif")),
        "--json",
        cwd=tmp_path,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    labels = [
        candidate["label"] for candidate in json.loads(completed.stdout)["candidates"]
    ]
    assert labels[0] == "fm"
    assert len(labels) > 1
