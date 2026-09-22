# Set up mMACE (magnetism ML)

Two features share one manual setup, on top of the default install:

| Feature | What it needs mMACE for |
| --- | --- |
| `is_magnetic` classification | Reads a frozen mMACE embedding to classify magnetic vs. non-magnetic (`human > ml > llm > heuristic`, see [Scientific controls](cli.md#scientific-controls)) |
| `goldilocks magnetic-orderings --rank-with-mmace` | Relaxes every candidate ordering's magnetic moments on a frozen mMACE potential energy surface and ranks them by energy per atom (see [Magnetic orderings](cli.md#magnetic-orderings)) |

Both stay permanently manual — not something `pip install goldilocks-core[...]` can ever
complete on its own. The `mace-torch` package this needs is a research collaborator's fork
with no PyPI release, and PyPI's own upload validation rejects a package that declares a
direct git dependency in its metadata regardless. Without this setup, both features degrade
to their next tier (LLM or heuristic for `is_magnetic`; an unranked candidate list with a
warning for `--rank-with-mmace`) rather than failing.

## 1. Install the model asset

Already part of the `default` and `workbench` asset profiles — most installs already have it:

```bash
uv run goldilocks assets install default
uv run goldilocks assets status models/is-magnetic-classifier
# models/is-magnetic-classifier@1: installed
```

This alone is **not** enough to reach the `ml` tier — steps 2 and 3 are the actual gate.

## 2. Install `mace`, `e3nn`, and `sphericart`

None of these are installable as a `goldilocks-core` or `goldilocks-ml` extra. Install them
directly into the same virtualenv `uv` manages for this project:

```bash
uv pip install ase==3.28.0 e3nn==0.4.4 sphericart==1.0.9 sphericart-torch==1.0.9
uv pip install "mace-torch @ git+https://github.com/CheukHinHoJerry/mace.git@19cdf6692c48e068a24e06cfe1ffc670e8aea3dd"
```

`mace-torch` here is `CheukHinHoJerry/mace`, a research collaborator's fork — **not** the
`ACEsuit/mace` package published on PyPI under the same distribution name. The pinned commit
matters: `goldilocks-ml` verified it produces bit-identical classifier embeddings to the
model's original training commit, and it additionally implements the collinear moment
relaxation `--rank-with-mmace` needs (an older commit on the same fork lacks that method).
Installing a different commit, or upstream `ACEsuit/mace`, will not work even though the
package name matches.

**`uv sync` removes this install.** Because these packages aren't in `uv.lock`, any later
`uv sync` (including one run for an unrelated reason, e.g. picking up a new dependency after
`git pull`) silently uninstalls all four again — no error, `is_magnetic` just degrades back to
the LLM/heuristic tier. Re-run the two commands above after every `uv sync`; step 5 below is
how you notice if you forgot.

Verify the install:

```bash
uv run python -c "import mace, e3nn, sphericart; print('ok')"
```

## 3. Download the mMACE backbone checkpoint

The published `is-magnetic-classifier` asset record bundles this checkpoint file, but
`goldilocks assets install` deliberately does not download it — see the comment on
`defaults.is_magnetic` in `src/goldilocks_core/ml/registry.toml` for why. Fetch it directly:

```bash
mkdir -p ~/.local/share/goldilocks/mmace
curl -L -o ~/.local/share/goldilocks/mmace/mace_matpes_pbe_baseline_run-3.model \
  https://data-collections.psdi.ac.uk/api/records/1g8rw-q8128/files/mace_matpes_pbe_baseline_run-3.model/content
```

The `-L` is not optional: this URL 302-redirects to a presigned, short-lived S3 link.
Without it, `curl` saves the ~500-byte redirect response itself (a plain-text URL) instead
of the model — a mistake that succeeds silently and only surfaces at the checksum step
below, or worse, later as `mace` silently mispredicting on truncated weights.

~80.5 MB (84,430,851 bytes). Verify the download before trusting it:

```bash
md5sum ~/.local/share/goldilocks/mmace/mace_matpes_pbe_baseline_run-3.model
# expect: 2486abb4f6380a599759833125565019
```

(`md5sum` on Linux; `md5 ~/.local/.../mace_matpes_pbe_baseline_run-3.model` on macOS.)

The path above is a suggestion, not a requirement — `goldilocks-core` never manages this
file, it only reads whatever path you point it at in step 4.

## 4. Point `GOLDILOCKS_MACE_BACKBONE` at it

Both features read the same environment variable — set it once, before running either:

```bash
export GOLDILOCKS_MACE_BACKBONE=~/.local/share/goldilocks/mmace/mace_matpes_pbe_baseline_run-3.model
```

Add that line to your shell profile (or `.env` / the launcher you use to start
`goldilocks serve http` for the Workbench) so it's set for every session, not just the
current shell.

## 5. Verify it's actually being used

A structure loading successfully is not proof the `ml` tier is live — check the `source`
field explicitly:

`Fe_bcc.cif` (bundled with this repo, and actually magnetic) makes a good check. Note:
importing `mace` prints a UserWarning and a "cuequivariance ... will be disabled" line to
stdout before anything else runs, ahead of the JSON `explain --json` prints — harmless, but
it breaks a naive `json.load(sys.stdin)`. Skip to the first `{`:

```bash
STRUCTURE="$(uv run goldilocks examples path)/Fe_bcc.cif"
uv run goldilocks explain "$STRUCTURE" --json | python3 -c \
  "import json,sys; text=sys.stdin.read(); print(json.loads(text[text.index('{'):])['records']['is_magnetic']['source'])"
# expect: ml   (not: heuristic, or llm)
```

For ordering ranking, a successful `--rank-with-mmace` run reports a `model_id` and real
per-candidate energies instead of degrading with a warning:

```bash
uv run goldilocks magnetic-orderings "$STRUCTURE" --rank-with-mmace --json
```

If `is_magnetic.source` still reads `heuristic` after all four steps, check in this order:
`uv run python -c "import mace"` (step 2 actually installed into *this* venv, not a
different Python), `echo $GOLDILOCKS_MACE_BACKBONE` (set in *this* shell, and pointing at a
file that exists), then `ls -la` the checkpoint — a file around 500 bytes instead of ~80.5 MB
means step 3's `-L` got dropped and you saved the redirect's own presigned-URL text instead
of the model (`cat` it to confirm: it reads as a `https://s3...` URL, not binary). Re-run the
`curl -L` command. A genuinely truncated/corrupted download of the right rough size loads
without error but predicts garbage rather than raising — the md5sum check is what catches
that case, not the file-size eyeball check.

## 6. Start the Workbench with mMACE enabled

`GOLDILOCKS_MACE_BACKBONE` has to be set in the shell that launches the **backend** process
— set it before running the usual startup command, not after:

```bash
export GOLDILOCKS_MACE_BACKBONE=~/.local/share/goldilocks/mmace/mace_matpes_pbe_baseline_run-3.model
uv run poe workbench   # dev mode: backend on :8000, Vite dev server on :5173
# or: uv run poe stage # production-like: one built bundle, served from :8000
```

If the backend is already running from an earlier session, the variable won't retroactively
apply — stop it and restart from a shell that has the export.

Load a magnetic structure (e.g. `src/goldilocks_core/examples/structures/Fe_bcc.cif`) and
check the **Analysis** column's "is magnetic" field: once the `ml` tier is live, its caption
switches from "Heuristic default" to **"Goldilocks-ML prediction"**. A "Magnetic Orderings"
column also appears automatically once a structure classifies as magnetic (it stays hidden
for non-magnetic ones) — this is where `--rank-with-mmace` lives in the UI.

**Known limitation (as of 2026-09-22): the "Run mMACE" button in the Workbench does not
currently work.** Clicking it fails with *"mMACE-based magnetic ordering ranking failed to
run: signal only works in main thread of the main interpreter"* — confirmed 100%
reproducible, not intermittent. Root cause: `goldilocks-ml`'s
`fm_fim_relax.seed_moments` guards a slow oxidation-state guess with a
`signal.signal(signal.SIGALRM, ...)`-based timeout, which CPython only permits from the
main thread of the main interpreter. `goldilocks-core`'s `POST /magnetic-orderings` HTTP
handler is a synchronous (`def`, not `async def`) FastAPI route, which Starlette
automatically runs in a worker thread — so this fails on *every* HTTP (and MCP, same
reason) request that reaches it, not just occasionally. `rank_orderings()` already catches
this and degrades gracefully (no crash, just an unranked list with a warning), but the
feature is silently non-functional over HTTP/MCP until the upstream timeout mechanism is
made thread-safe. Filed as
[stfc/goldilocks-ml#95](https://github.com/stfc/goldilocks-ml/issues/95).

**The CLI path is unaffected** (a single-threaded process is always "the main thread"):

```bash
uv run goldilocks magnetic-orderings "$STRUCTURE" --rank-with-mmace --json
```

is the reliable way to use ranking today; treat the Workbench button as broken until #95
above is resolved.
