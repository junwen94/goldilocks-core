"""``goldilocks assets install|status|verify``: unchanged from v1 in
substance (``assets/store.py``/``assets/runtime.py`` were ported
verbatim in v2 epic 3, #4) -- only moved into this package and no
longer wrapping every exception in a local try/except, since v2's
asset-store errors are already ``ExpectedFailure`` subclasses that
``cli.core``'s one top-level handler now catches uniformly (P6).
"""

from __future__ import annotations

import argparse

from goldilocks_core.assets.runtime import (
    install as install_assets,
    statuses as asset_statuses,
    verify as verify_assets,
)
from goldilocks_core.assets.store import AssetStore
from goldilocks_core.examples.structures import structures_path


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    assets = subparsers.add_parser(
        "assets", help="Install and inspect immutable runtime assets."
    )
    asset_commands = assets.add_subparsers(dest="assets_command", required=True)
    for command in ("install", "status", "verify"):
        operation = asset_commands.add_parser(command)
        operation.add_argument(
            "name",
            nargs="?",
            default="default",
            help="Asset id or shipped profile name (default: default).",
        )

    examples = subparsers.add_parser(
        "examples", help="Inspect the example structures bundled with the package."
    )
    example_commands = examples.add_subparsers(dest="examples_command", required=True)
    example_commands.add_parser(
        "path", help="Print the directory holding the bundled example structures."
    )


def run(args: argparse.Namespace) -> None:
    if args.command == "examples":
        print(structures_path())
        return

    store = AssetStore()
    print(f"asset root: {store.root}")
    if args.assets_command == "install":
        for asset in install_assets(args.name, store=store):
            print(f"{asset.id}@{asset.version}: installed")
        return
    if args.assets_command == "status":
        for asset_id, version, state in asset_statuses(args.name, store=store):
            print(f"{asset_id}@{version}: {state}")
        return
    for asset in verify_assets(args.name, store=store):
        print(f"{asset.id}@{asset.version}: verified")
