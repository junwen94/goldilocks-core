import { type DragEvent, useRef, useState } from "react";
import {
  Button,
  FileButton,
  Group,
  Loader,
  Paper,
  Text,
} from "@mantine/core";
import { Upload } from "lucide-react";

import type { StructureInput, StructureInspection } from "../api/coreClient";

export function StructureSourceControls({
  source,
  inspection,
  inspecting,
  onOpen,
}: {
  readonly source: StructureInput | null;
  readonly inspection: StructureInspection | null;
  readonly inspecting: boolean;
  readonly onOpen: (input: StructureInput) => Promise<void>;
}) {
  const resetFileInput = useRef<(() => void) | null>(null);
  const selectionEpoch = useRef(0);
  const [dragging, setDragging] = useState(false);
  const [readError, setReadError] = useState<string | null>(null);

  async function openFile(file: File): Promise<void> {
    const selection = ++selectionEpoch.current;
    setReadError(null);
    if (file.size > 5 * 1024 * 1024) {
      setReadError("Structure files must be 5 MB or smaller.");
      return;
    }
    if (file.size === 0) {
      setReadError("The selected structure file is empty.");
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

  let sourceHelp = "CIF or POSCAR · 5 MB maximum file size";
  if (source !== null) sourceHelp = "Inspecting structure";
  if (inspection !== null) {
    sourceHelp = `${String(inspection.structure.site_count)} sites · parsed`;
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
          aria-label={
            source === null
              ? "Choose a CIF or POSCAR structure"
              : "Replace structure file"
          }
          disabled={inspecting}
        >
          {source === null ? "Browse files" : "Replace file"}
        </Button>
      )}
    </FileButton>
  );

  return (
    <>
      {/* Once a structure is loaded the viewer needs most of this card's
       * height, so the dropzone collapses from a compact first-run empty
       * state into a single compact status row (same drop target, same
       * controls) instead of staying full-size forever. Kept to two lines
       * even in the empty state -- the drop target still covers the whole
       * Paper, so it doesn't need to be tall to stay easy to hit. */}
      {source === null ? (
        <Paper withBorder p="xs" {...dropzoneProps}>
          <Group justify="center" gap="xs" wrap="wrap">
            <Upload aria-hidden="true" size={16} />
            <Text fw={600} size="sm">
              Drop a structure
            </Text>
            <Text size="sm" c="dimmed">
              or
            </Text>
            {fileButton}
          </Group>
          <Text id="structure-source-help" c="dimmed" size="xs" ta="center" mt={2}>
            {sourceHelp}
          </Text>
        </Paper>
      ) : (
        <Paper withBorder p="xs" {...dropzoneProps}>
          <Group justify="space-between" wrap="nowrap" gap="xs">
            <Group gap="xs" wrap="nowrap" style={{ minWidth: 0 }}>
              {inspecting ? (
                <Loader size="xs" aria-hidden="true" />
              ) : (
                <Upload aria-hidden="true" size={16} />
              )}
              <Text fw={600} truncate>
                {source.structure_name}
              </Text>
              <Text id="structure-source-help" c="dimmed" size="sm" truncate>
                {sourceHelp}
              </Text>
            </Group>
            {fileButton}
          </Group>
        </Paper>
      )}
      {readError === null ? null : (
        <Text c="red" size="sm" role="alert">
          {readError}
        </Text>
      )}
      {inspection === null ? null : (
        <StructureSummary inspection={inspection} />
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
  const periodicity = structure.periodicity.every(Boolean) ? "3D" : "Partial";
  return (
    <Text
      size="sm"
      c="dimmed"
      mt="xs"
      truncate
      aria-label="Inspected structure summary"
    >
      {structure.formula} · {elements.join(" ")} ·{" "}
      {structure.lattice.volume_angstrom3.toFixed(1)} Å³ · {periodicity}
    </Text>
  );
}

function structureFormat(name: string): "cif" | "poscar" {
  return name.toLowerCase().endsWith(".cif") ? "cif" : "poscar";
}
