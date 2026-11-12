import { unzipSync, zipSync } from "fflate";

import type { ArchiveDownload } from "../api/coreClient";

/** One archive named `filename` that holds each of `archives` in a folder
 * named after it, its filename without `.zip` -- Bundle's download of
 * several magnetic orderings at once (#293). */
export async function combineArchives(
  filename: string,
  archives: readonly ArchiveDownload[],
): Promise<ArchiveDownload> {
  const entries: Record<string, Uint8Array> = {};
  for (const archive of archives) {
    const folder = archive.filename.replace(/\.zip$/, "");
    const files = unzipSync(new Uint8Array(await archive.blob.arrayBuffer()));
    for (const [path, bytes] of Object.entries(files)) {
      entries[`${folder}/${path}`] = bytes;
    }
  }
  return {
    blob: new Blob([new Uint8Array(zipSync(entries))], {
      type: "application/zip",
    }),
    filename,
  };
}
