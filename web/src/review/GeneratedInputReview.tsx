import { useEffect, useState } from "react";
import { Group, Stack, Tabs, Text, Title } from "@mantine/core";

import type { ArchiveDownload } from "../api/coreClient";
import {
  type ArchiveContents,
  artifactMetadata,
  unzipArchive,
} from "./artifacts";
import { GeneratedInputPreview } from "./GeneratedInputPreview";

export function GeneratedInputReview({
  archive,
}: {
  readonly archive: ArchiveDownload | null;
}) {
  const contents = useUnzippedArchive(archive);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);

  if (archive === null) {
    return (
      <Stack component="section" gap="xs" miw={0}>
        <Title order={3}>Generated inputs</Title>
        <Text c="dimmed">
          Generate input files to preview them here before saving.
        </Text>
      </Stack>
    );
  }
  if (contents === null) {
    return (
      <Stack component="section" gap="xs" miw={0}>
        <Title order={3}>Generated inputs</Title>
        <Text c="dimmed">Reading the generated archive…</Text>
      </Stack>
    );
  }

  const { manifest, files } = contents;
  const file =
    files.find((candidate) => candidate.path === selectedPath) ?? files[0];

  return (
    <Stack component="section" gap="xs" miw={0}>
      <Group component="header" justify="space-between">
        <Title order={3}>Generated inputs</Title>
        <Text size="sm" c="dimmed">
          {files.length} files
        </Text>
      </Group>
      {file === undefined ? (
        <Text c="dimmed">No generated input files.</Text>
      ) : (
        <Tabs value={file.path} onChange={setSelectedPath}>
          <Tabs.List aria-label="Generated input files">
            {files.map((candidate) => (
              <Tabs.Tab key={candidate.path} value={candidate.path}>
                {candidate.path.split("/").at(-1)}
              </Tabs.Tab>
            ))}
          </Tabs.List>
          {files.map((candidate) => (
            <Tabs.Panel key={candidate.path} value={candidate.path}>
              {candidate.path === file.path ? (
                <GeneratedInputPreview
                  path={file.path}
                  content={file.content}
                  digest={artifactMetadata(manifest, file.path)?.sha256}
                />
              ) : null}
            </Tabs.Panel>
          ))}
        </Tabs>
      )}
    </Stack>
  );
}

/** Unzipping is genuinely async work driven by an external blob, not
 * derivable during render -- a legitimate effect, unlike CalculationForm's
 * JsonOverrideControl (which only needed to resync local text state). */
function useUnzippedArchive(
  archive: ArchiveDownload | null,
): ArchiveContents | null {
  const [state, setState] = useState<{
    readonly archive: ArchiveDownload;
    readonly contents: ArchiveContents;
  } | null>(null);

  useEffect(() => {
    if (archive === null) return;
    let cancelled = false;
    void unzipArchive(archive).then((contents) => {
      if (!cancelled) setState({ archive, contents });
    });
    return () => {
      cancelled = true;
    };
  }, [archive]);

  return state?.archive === archive ? state.contents : null;
}
