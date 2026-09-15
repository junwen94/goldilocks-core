#!/usr/bin/env python3
"""Fail when exported mutmut results fall below the required mutation score."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# The enforced mutation score. This constant is the gate's single home;
# callers (`just mutation`, CI) must not override it.
MINIMUM_SCORE = 0.74


def main() -> int:
    """Read mutmut CI statistics and enforce a killed-mutant ratio."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stats", type=Path)
    parser.add_argument("--minimum", type=float, default=MINIMUM_SCORE)
    args = parser.parse_args()

    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    killed = int(stats["killed"])
    total = int(stats["total"])
    if total <= 0:
        parser.error("mutation statistics contain no mutants")
    if not 0.0 <= args.minimum <= 1.0:
        parser.error("--minimum must be between 0 and 1")

    score = killed / total
    line = f"Mutation score: {killed}/{total} ({score:.1%}); minimum {args.minimum:.1%}"
    print(line)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as summary_file:
            summary_file.write(line + "\n")
    return int(score < args.minimum)


if __name__ == "__main__":
    raise SystemExit(main())
