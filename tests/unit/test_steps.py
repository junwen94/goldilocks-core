from __future__ import annotations

import pytest

from goldilocks_core.steps import (
    DEFAULT_OUTDIR,
    DEFAULT_PREFIX,
    DEFAULT_PSEUDO_DIR,
    SharedContext,
    Step,
    default_shared_context,
)


def test_step_defaults_to_no_args_and_empty_files() -> None:
    step = Step(name="scf", executable="pw.x")

    assert step.args == ()
    assert step.files == {}
    assert step.stdout is None
    assert step.workdir is None


def test_step_carries_code_specific_args_opaquely() -> None:
    step = Step(
        name="scf",
        executable="pw.x",
        args=("-npool", "4", "-in", "scf.in"),
        files={"scf.in": "&CONTROL\n/\n"},
        stdout="scf.out",
    )

    assert step.args == ("-npool", "4", "-in", "scf.in")
    assert step.files["scf.in"].startswith("&CONTROL")


def test_step_is_frozen() -> None:
    step = Step(name="scf", executable="pw.x")

    with pytest.raises(AttributeError):
        step.executable = "ph.x"  # type: ignore[misc]


def test_shared_context_is_frozen() -> None:
    ctx = SharedContext(prefix="pwscf", outdir="./out", pseudo_dir="./pseudo")

    with pytest.raises(AttributeError):
        ctx.prefix = "other"  # type: ignore[misc]


def test_default_shared_context_matches_qe_and_v1_defaults() -> None:
    ctx = default_shared_context()

    assert ctx == SharedContext(
        prefix=DEFAULT_PREFIX, outdir=DEFAULT_OUTDIR, pseudo_dir=DEFAULT_PSEUDO_DIR
    )
    assert ctx.prefix == "pwscf"
    assert ctx.outdir == "./out"
    assert ctx.pseudo_dir == "./pseudo"
