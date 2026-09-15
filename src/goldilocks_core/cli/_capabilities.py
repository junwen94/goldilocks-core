"""``goldilocks capabilities``: the full reflected payload (v2 epic 8,
#8's ``capabilities()``) as a third thin entry point alongside HTTP's
``GET /capabilities`` and MCP's ``capabilities`` tool -- #62's own gap:
nothing on the CLI previously reached ``codes``/``tasks``/
``pseudopotential_tables``/``hpc_profiles``/``warnings``, only
``settings``/``models``/``assets status`` each covered their own
narrower slice.
"""

from __future__ import annotations

import argparse
import json

from goldilocks_core.capabilities import capabilities


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "capabilities",
        help="Print the full capabilities payload: codes, tasks, facts, "
        "settings, pseudopotential tables, HPC profiles, models, and warnings.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")


def run(args: argparse.Namespace) -> None:
    caps = capabilities()
    if args.json:
        print(json.dumps(caps, indent=2, sort_keys=True))
        return

    print(f"core_version: {caps['core_version']}")
    print(f"vocabulary_version: {caps['vocabulary_version']}")
    for code in caps["codes"]:
        tasks = ", ".join(code["tasks"])
        print(f"code: {code['id']} ({code['name']}) tasks={tasks}")
    for task in caps["tasks"]:
        executables = ", ".join(task["executables"])
        print(
            f"task: {task['id']} ({task['name']}) "
            f"step_count={task['step_count']} executables={executables}"
        )
    print(
        f"facts: {len(caps['facts'])} (see 'goldilocks inspect'/'goldilocks explain')"
    )
    print(f"settings: {len(caps['settings'])} (see 'goldilocks settings')")
    for table in caps["pseudopotential_tables"]:
        default = " [default]" if table["default"] else ""
        print(
            f"pseudopotential_table: {table['id']} "
            f"({table['functional']}, {table['accuracy']}){default}"
        )
    for profile in caps["hpc_profiles"]:
        print(f"hpc_profile: {profile['id']} ({profile['scheduler']})")
    if caps["models"]:
        for model in caps["models"]:
            print(f"model: {model}")
    else:
        print("models: none installed yet (ml integration lands in a later epic)")
    print(f"warnings: {len(caps['warnings'])} known warning codes")
    print(f"sources: {', '.join(caps['sources'])}")
