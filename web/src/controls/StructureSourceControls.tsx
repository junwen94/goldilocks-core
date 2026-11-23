import { type DragEvent, useRef, useState } from "react";
import {
  Button,
  FileButton,
  Group,
  Loader,
  NativeSelect,
  Paper,
  Stack,
  Text,
} from "@mantine/core";
import { Upload } from "lucide-react";

import type { StructureInput, StructureInspection } from "../api/coreClient";
import { formatSiteCount } from "../viewer/siteCount";

/** A structure an embedding host offers next to Browse files: the file
 * name and text it would have as an opened file. `format` defaults to the
 * one inferred from `name` (cif for a `.cif` name, else poscar). */
export interface StructureChoice {
  readonly name: string;
  readonly content: string;
  readonly format?: "cif" | "poscar";
}

/** The host's structures for the Structure card, listed under `label`
 * (e.g. "From this chat"). */
export interface StructureChoices {
  readonly label: string;
  readonly items: readonly StructureChoice[];
}

const MAX_STRUCTURE_BYTES = 5 * 1024 * 1024;

export function StructureSourceControls({
  source,
  inspection,
  inspecting,
  onOpen,
  choices,
}: {
  readonly source: StructureInput | null;
  readonly inspection: StructureInspection | null;
  readonly inspecting: boolean;
  readonly onOpen: (input: StructureInput) => Promise<void>;
  readonly choices?: StructureChoices | undefined;
}) {
  const resetFileInput = useRef<(() => void) | null>(null);
  const selectionEpoch = useRef(0);
  const [dragging, setDragging] = useState(false);
  const [readError, setReadError] = useState<string | null>(null);

  async function openFile(file: File): Promise<void> {
    const selection = ++selectionEpoch.current;
    setReadError(null);
    const sizeError = structureSizeError(file.size);
    if (sizeError !== null) {
      setReadError(sizeError);
      return;
    }
    try {
      const content = await file.text();
      if (selection !== selectionEpoch.current) return;
      await onOpen({
        structure_content: content,
        structure_name: file.name,
        structure_format: structureFormat(file.name),
      });
    } catch {
      if (selection === selectionEpoch.current) {
        setReadError("The selected file could not be read.");
      }
    }
  }

  function fileSelected(file: File | null): void {
    resetFileInput.current?.();
    if (file !== null) void openFile(file);
  }

  function fileDropped(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file !== undefined) void openFile(file);
  }

  /** Opens a host structure as an opened file is opened: the same checks,
   * the same `onOpen`, and it supersedes a file still being read. */
  function openChoice(choice: StructureChoice): void {
    ++selectionEpoch.current;
    const sizeError = structureSizeError(
      new TextEncoder().encode(choice.content).length,
    );
    setReadError(sizeError);
    if (sizeError !== null) return;
    void onOpen({
      structure_content: choice.content,
      structure_name: choice.name,
      structure_format: choice.format ?? structureFormat(choice.name),
    });
  }

  let sourceHelp = "CIF or POSCAR · 5 MB maximum file size";
  if (source !== null) sourceHelp = "Inspecting structure";
  if (inspection !== null) {
    sourceHelp = `${formatSiteCount(inspection.structure.site_count, "site")} · parsed`;
  }

  const dropzoneProps = {
    onDragEnter: (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setDragging(true);
    },
    onDragOver: (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
    },
    onDragLeave: () => {
      setDragging(false);
    },
    onDrop: fileDropped,
    "aria-busy": inspecting,
    bg: dragging
      ? "var(--mantine-primary-color-light)"
      : "var(--mantine-color-body)",
  };

  const fileButton = (
    <FileButton
      resetRef={resetFileInput}
      onChange={fileSelected}
      disabled={inspecting}
    >
      {(fileButtonProps) => (
        <Button
          {...fileButtonProps}
          variant="subtle"
          type="button"
          aria-describedby="structure-source-help"
          disabled={inspecting}
        >
          {source === null ? "Browse files" : "Replace file"}
        </Button>
      )}
    </FileButton>
  );

  const items = choices?.items ?? [];
  const chosen = items.findIndex(
    (item) =>
      source !== null &&
      item.name === source.structure_name &&
      item.content === source.structure_content,
  );

  return (
    <>
      {choices === undefined || items.length === 0 ? null : (
        <NativeSelect
          label={choices.label}
          disabled={inspecting}
          value={chosen === -1 ? "" : String(chosen)}
          data={[
            { value: "", label: "Choose a structure", disabled: true },
            ...items.map((item, index) => ({
              value: String(index),
              label: item.name,
            })),
          ]}
          onChange={(event) => {
            const item = items[Number(event.currentTarget.value)];
            if (item !== undefined) openChoice(item);
          }}
        />
      )}
      {/* Once a structure is loaded the viewer needs most of this card's
       * height, so the dropzone collapses from the spacious first-run
       * empty state into a single compact status row (same drop target,
       * same controls) instead of staying full-size forever. */}
      {source === null ? (
        <Paper withBorder p="md" {...dropzoneProps}>
          <Stack align="center" gap="xs">
            <Upload aria-hidden="true" size={18} />
            <Text fw={600} truncate w="100%" ta="center">
              Drop a structure
            </Text>
            <Text id="structure-source-help" c="dimmed" size="sm" ta="center">
              {sourceHelp}
            </Text>
            {fileButton}
          </Stack>
        </Paper>
      ) : (
        <Paper withBorder p="xs" {...dropzoneProps}>
          <Stack gap={4}>
            {/* The filename gets its own full-width row -- sharing a row
             * with the site-count/status text left too little room for
             * either, so both truncated into near-unreadable fragments. */}
            <Group gap="xs" wrap="nowrap" style={{ minWidth: 0 }}>
              {inspecting ? (
                <Loader size="xs" aria-hidden="true" />
              ) : (
                <Upload aria-hidden="true" size={16} />
              )}
              <Text fw={600} truncate>
                {source.structure_name}
              </Text>
            </Group>
            <Group justify="space-between" wrap="nowrap" gap="xs">
              <Text id="structure-source-help" c="dimmed" size="sm" truncate>
                {sourceHelp}
              </Text>
              {fileButton}
            </Group>
            {inspection === null ? null : (
              <StructureSummary inspection={inspection} />
            )}
          </Stack>
        </Paper>
      )}
      {readError === null ? null : (
        <Text c="red" size="sm" role="alert">
          {readError}
        </Text>
      )}
    </>
  );
}

function StructureSummary({
  inspection,
}: {
  readonly inspection: StructureInspection;
}) {
  const structure = inspection.structure;
  const elements = [
    ...new Set(
      structure.sites.flatMap((site) =>
        site.species.map((species) => species.symbol),
      ),
    ),
  ];
  // No periodicity token: pymatgen marks every CIF/POSCAR fully periodic,
  // so it never varies, and next to the formula it reads as the
  // structure's dimensionality (a monolayer would show "3D").
  return (
    <Text
      size="sm"
      c="dimmed"
      truncate
      aria-label="Inspected structure summary"
    >
      {structure.formula} · {elements.join(" ")} ·{" "}
      {structure.lattice.volume_angstrom3.toFixed(1)} Å³
    </Text>
  );
}

function structureFormat(name: string): "cif" | "poscar" {
  return name.toLowerCase().endsWith(".cif") ? "cif" : "poscar";
}

/** Why a structure of `bytes` bytes cannot be opened, or null if it can. */
function structureSizeError(bytes: number): string | null {
  if (bytes > MAX_STRUCTURE_BYTES) {
    return "Structure files must be 5 MB or smaller.";
  }
  if (bytes === 0) return "The selected structure file is empty.";
  return null;
}
