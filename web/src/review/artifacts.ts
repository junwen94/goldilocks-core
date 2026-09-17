import { unzipSync } from "fflate";

import type { ArchiveDownload } from "../api/coreClient";

/** Mirrors `goldilocks.json`'s shape, written by `bundle.py`'s
 * `bundle_files` -- the manifest v1's combined /compute response used
 * to hand back as JSON is now only ever found inside the downloaded
 * archive itself (v2's `/run` never returns JSON and a file's bytes in
 * the same response). */
export interface InputManifest {
  readonly schema_version: number;
  readonly records: Readonly<Record<string, unknown>>;
  readonly warnings: readonly unknown[];
  readonly citations: readonly string[];
  readonly files: Readonly<
    Record<
      string,
      {
        readonly role: string;
        readonly sha256: string;
        readonly size_bytes: number;
      }
    >
  >;
}

export interface ArchiveFile {
  readonly path: string;
  readonly content: string;
}

export interface ArchiveContents {
  readonly manifest: InputManifest | null;
  readonly files: readonly ArchiveFile[];
}

/** Review order, most-relevant-to-inspect first: the code input(s) a
 * reviewer actually reads, then the submission script, then the
 * human-facing summary, then the machine manifest, then everything
 * else (citations, pseudopotential files) -- not `bundle.py`'s own
 * alphabetical `sorted(files.items())`, which only exists there for
 * deterministic archive contents, not review ergonomics. */
function displayPriority(path: string): number {
  const name = path.split("/").at(-1) ?? path;
  if (path.endsWith(".in")) return 0;
  if (name === "submit.sh") return 1;
  if (name === "README.md") return 2;
  if (path.endsWith(".json")) return 3;
  if (path.endsWith(".md")) return 4;
  return 5;
}

/** Unzips a downloaded archive client-side to recover per-file text and
 * `goldilocks.json`'s manifest -- purely a presentation concern (the
 * API client only ever hands back the raw zip bytes it was given). */
export async function unzipArchive(
  archive: ArchiveDownload,
): Promise<ArchiveContents> {
  const entries = unzipSync(new Uint8Array(await archive.blob.arrayBuffer()));
  const decoder = new TextDecoder("utf-8");
  const files: ArchiveFile[] = [];
  let manifest: InputManifest | null = null;
  for (const [path, bytes] of Object.entries(entries)) {
    const content = decoder.decode(bytes);
    if (path === "goldilocks.json") {
      manifest = parseManifest(content);
    }
    files.push({ path, content });
  }
  files.sort((a, b) => {
    const priority = displayPriority(a.path) - displayPriority(b.path);
    return priority !== 0 ? priority : a.path.localeCompare(b.path);
  });
  return { manifest, files };
}

function parseManifest(content: string): InputManifest | null {
  try {
    const parsed: unknown = JSON.parse(content);
    return isManifest(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function isManifest(value: unknown): value is InputManifest {
  return (
    typeof value === "object" &&
    value !== null &&
    "files" in value &&
    typeof value.files === "object"
  );
}

export function artifactMetadata(
  manifest: InputManifest | null,
  filename: string,
): { sha256: string | undefined; size_bytes: number | undefined } | undefined {
  const descriptor = manifest?.files[filename];
  return descriptor === undefined
    ? undefined
    : { sha256: descriptor.sha256, size_bytes: descriptor.size_bytes };
}
