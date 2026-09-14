# Task index. Run with `uv run just <recipe>` so the uv environment is active.
# Shell recipes live here; Python gates live in scripts/.

set shell := ["bash", "-uc", "-o", "errexit"]

# The local pre-PR gate: lint, complexity ceilings, and tests.
check: lint test

# Bump the package version (major|minor|patch|X.Y.Z) and refresh the lockfile.
bump target:
    uv run python scripts/bump_version.py "{{target}}"

# Format src, tests, and scripts with ruff.
fmt:
    uv run ruff format src tests scripts

# Check style, import and cyclomatic ceilings, matching the CI gate.
lint:
    uv run ruff check src tests scripts
    uv run ruff format --check src tests scripts
    uv run python scripts/check_complexity.py

# Run the pytest suite with branch coverage.
test:
    uv run pytest --cov --cov-report=term-missing:skip-covered

# Run mutation testing and enforce the score gate from check_mutation_score.py.
mutation:
    uv run mutmut run --max-children 4
    uv run mutmut export-cicd-stats
    uv run python scripts/check_mutation_score.py mutants/mutmut-cicd-stats.json

# Build the sdist and wheel and validate their contents.
dist:
    #!/usr/bin/env bash
    set -euo pipefail
    out="$(mktemp -d)"
    trap 'rm -rf "$out"' EXIT
    uv build --no-sources --out-dir "$out"
    uv run python scripts/validate_distribution.py "$out"

# Lint, unit-test, and build the Workbench frontend.
web-check:
    cd web && npm run check

# Build the production image and run the Playwright e2e suite against it.
image-e2e:
    bash scripts/image_e2e.sh

# Install assets and serve the Core HTTP backend on :8000.
serve:
    #!/usr/bin/env bash
    set -euo pipefail
    if curl -s -m1 http://127.0.0.1:8000/ >/dev/null 2>&1; then
        echo "port 8000 is already in use - stop the other server (just workbench, just serve, or just stage)" >&2
        exit 1
    fi
    goldilocks assets install workbench
    exec goldilocks serve http --host 127.0.0.1 --port 8000

# Build the Workbench bundle and serve it from the Core HTTP process on :8000.
stage:
    #!/usr/bin/env bash
    set -euo pipefail
    if curl -s -m1 http://127.0.0.1:8000/ >/dev/null 2>&1; then
        echo "port 8000 is already in use - stop the other server (just workbench, just serve, or just stage)" >&2
        exit 1
    fi
    (cd web && npm run build)
    goldilocks assets install workbench
    exec goldilocks serve http --host 127.0.0.1 --port 8000 --static-root web/dist

# Install assets, serve the Core HTTP backend on :8000, and run the Vite dev server on :5173.
workbench:
    #!/usr/bin/env bash
    set -euo pipefail
    if curl -s -m1 http://127.0.0.1:8000/ >/dev/null 2>&1; then
        echo "port 8000 is already in use - stop the other server (just workbench, just serve, or just stage)" >&2
        exit 1
    fi
    goldilocks assets install workbench
    goldilocks serve http --host 127.0.0.1 --port 8000 &
    backend_pid=$!
    trap 'kill "$backend_pid"' EXIT
    (cd web && npm run dev)
