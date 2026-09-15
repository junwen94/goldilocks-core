# Contributing

How to set up the repository, which checks a change must pass, and how a
release is cut. The code map is in [Architecture](architecture.md).

From the repository root:

```bash
uv sync --group dev
uv run pre-commit install
uv run just check
```

`just check` runs Ruff lint, format, and complexity checks, then pytest with
branch coverage. Use `uv run just fmt` to apply Python formatting. The commit
hooks run the same lint and complexity checks. For frontend checks and
API-schema refresh, follow the [Workbench guide](../web/README.md).

## Checks

| Entry point | Checks |
| --- | --- |
| `just lint` | Ruff lint and format; complexity ceilings |
| `just test` | pytest with branch coverage |
| `just check` | `lint`, then `test`; the pre-PR gate |
| `just mutation` | focused mutation testing against the score gate |
| `just dist` | sdist and wheel build into a fresh directory, then content validation |
| `just web-check` | Workbench lint, unit tests, and production build |
| `just image-e2e` | production image build, boot, and Playwright e2e (needs Docker) |
| `just bump <target>` | version bump in `pyproject.toml` plus `uv.lock` refresh |

Commit hooks run the lint and complexity recipes only; the full `just check`
gate runs before a PR. The distribution build in CI calls `uv build` and the
validator directly so it never syncs the development environment.

CI has two workflows on top of `quality.yml`, which is reusable and holds no
triggers itself:

- `ci.yml` runs on pushes to `main` and pull requests and cancels a superseded
  run on the same ref.
- `release.yml` runs on `v*` tags, the nightly schedule at 03:00 UTC, and
  manual dispatch. It runs the same quality jobs, then publishes. Release runs
  never cancel each other.

`scripts/` holds the Python gates the recipes and workflows call:
`check_complexity.py` (import and cyclomatic ceilings), `check_mutation_score.py`
(the enforced score and the CI step summary), `validate_distribution.py` (wheel
and sdist contents), `check_release_tag.py` (tag equals the `pyproject.toml`
version), `bump_version.py` (version bump and relock), and
`export_workbench_openapi.py` (HTTP contract export for the Workbench and the
image build).

## Cut a release

One version covers the repository: `pyproject.toml` owns it, and the Workbench
frontend ships inside the image rather than carrying its own version. Treat API
or schema changes as at least a minor bump during `0.x`; frontend-only fixes
can be patches.

```bash
# 1. Bump the version. This edits pyproject.toml, refreshes uv.lock, and
#    prints the tag to use.
uv run just bump patch          # or: minor | major | X.Y.Z
git commit -am "chore(release): bump version to X.Y.Z"
git push origin <branch>        # open the release PR and merge it

# 2. Tag the merge commit on an up-to-date main.
git checkout main && git pull
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

The tag push triggers `release.yml`. It runs the full suite on the tagged
commit, refuses the tag unless it equals the `pyproject.toml` version, then
pushes `ghcr.io/stfc/goldilocks-workbench` with `X.Y.Z`, `X.Y`, `X`, and
`latest` tags and creates a GitHub Release containing the sdist and wheel.

Nightly builds of `main` publish `nightly` and `nightly-<date>` image tags at
03:00 UTC. PRs and plain `main` pushes publish nothing. The first publish
creates the container package private; flip it to public once in the package
settings so anonymous pulls work.
## Complexity gates

`scripts/check_complexity.py` runs in contributor checks, hooks, and CI. Its AST
import ceiling applies to every production owner: 12 project origin modules and
24 imported symbols, with stricter limits for CLI, HTTP, MCP, assembly, and SCF.
It counts local and type-only imports, resolves re-exports, and counts accessed
module-alias members. Pure export packages are transparent to consumer counts.
The same gate runs Ruff's McCabe check: at most 10 per production function,
ignoring `noqa` suppressions. Reduce decisions and duplication; moving import
blocks or extracting shallow helper fleets does not deepen an interface.
