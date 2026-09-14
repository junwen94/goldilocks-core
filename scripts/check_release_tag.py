#!/usr/bin/env python3
"""Fail unless the release tag matches the package version in pyproject.toml."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path


def main() -> int:
    """Compare the release tag with the declared project version."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="the release tag, e.g. v0.2.0")
    args = parser.parse_args()

    version = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )["project"]["version"]
    expected = f"v{version}"
    if args.tag != expected:
        print(
            f"release tag {args.tag!r} does not match "
            f"pyproject.toml version {version!r}",
            file=sys.stderr,
        )
        return 1
    print(f"release tag {args.tag} matches pyproject.toml version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
