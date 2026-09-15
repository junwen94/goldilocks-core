#!/usr/bin/env python3
"""Bump the package version in pyproject.toml and refresh the lockfile."""

from __future__ import annotations

import argparse
import re
import subprocess
import tomllib
from pathlib import Path

_SEMVER = re.compile(r"\d+\.\d+\.\d+")


def main() -> int:
    """Rewrite project.version in pyproject.toml, relock, and print next steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="major, minor, patch, or an explicit X.Y.Z")
    args = parser.parse_args()

    pyproject = Path("pyproject.toml")
    current = tomllib.loads(pyproject.read_text())["project"]["version"]
    major, minor, patch = (int(part) for part in current.split("."))
    match args.target:
        case "major":
            new = f"{major + 1}.0.0"
        case "minor":
            new = f"{major}.{minor + 1}.0"
        case "patch":
            new = f"{major}.{minor}.{patch + 1}"
        case _ if _SEMVER.fullmatch(args.target):
            new = args.target
        case _:
            parser.error("target must be major, minor, patch, or an explicit X.Y.Z")

    updated, count = re.subn(
        r'(?m)^version = "[^"]+"$',
        f'version = "{new}"',
        pyproject.read_text(),
        count=1,
    )
    if count != 1:
        parser.error("pyproject.toml has no single top-level version line")
    pyproject.write_text(updated)
    subprocess.run(["uv", "lock"], check=True)

    tag = f"v{new}"
    print(f"bumped {current} -> {new}; commit this change in a release PR")
    print(f"after it merges from main: git tag {tag} && git push origin {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
