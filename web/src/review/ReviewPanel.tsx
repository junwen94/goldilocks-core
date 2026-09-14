import {
  Alert,
  Button,
  Group,
  Loader,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import { ArrowLeft, Download } from "lucide-react";

import type { AdvisorWarning } from "../api/coreClient";
import { useWorkspace, useWorkspaceSnapshot } from "../workspace/useWorkspace";
import { GeneratedInputReview } from "./GeneratedInputReview";
import { RecordReview } from "./RecordReview";
import { WARNING_LEVEL_COLORS } from "./warnings";
import "./ReviewPanel.css";

export function ReviewPanel({
  onShowStructure,
}: {
  readonly onShowStructure: () => void;
}) {
  const workspace = useWorkspace();
  const snapshot = useWorkspaceSnapshot();
  const reviewed = snapshot.reviewed;

  return (
    <Stack
      component="section"
      id="recommendation-panel"
      aria-label="Recommendation results"
      aria-busy={snapshot.operation === "explain"}
      p="md"
      gap="lg"
      miw={0}
    >
      <Group component="header" justify="space-between">
        <Title order={2}>Recommendation</Title>
        <Button
          variant="subtle"
          leftSection={<ArrowLeft aria-hidden="true" size={15} />}
          aria-label="Back to structure"
          onClick={onShowStructure}
        >
          Structure
        </Button>
      </Group>
      {reviewed === null && (
        <Stack
          align="center"
          justify="center"
          mih={240}
          role={snapshot.operation === "explain" ? "status" : undefined}
        >
          {snapshot.operation === "explain" && <Loader size="sm" />}
          <Text fw={600}>
            {snapshot.operation === "explain"
              ? "Computing recommendation"
              : "No recommendation"}
          </Text>
        </Stack>
      )}
      {reviewed !== null && (
        <>
          {snapshot.outOfDate && (
            <Alert
              role="status"
              aria-label="Recommendation notice"
              aria-live="polite"
              aria-atomic="true"
            >
              Your settings changed. Update the recommendation before generating
              input files.
            </Alert>
          )}
          <Group justify="space-between">
            <Button
              rightSection={<Download aria-hidden="true" size={14} />}
              loading={snapshot.operation === "download"}
              disabled={snapshot.outOfDate || snapshot.operation !== null}
              onClick={() =>
                void workspace.dispatch({ type: "review.download" })
              }
            >
              Generate input files (.zip)
            </Button>
            {snapshot.lastDownload === null || snapshot.outOfDate ? null : (
              <Text
                size="sm"
                role="status"
                aria-label="Archive status"
                aria-live="polite"
              >
                {snapshot.lastDownload.filename} is ready
              </Text>
            )}
          </Group>
          <GeneratedInputReview
            archive={snapshot.outOfDate ? null : snapshot.lastDownload}
          />
          <RecordReview records={reviewed.records} />
          <Warnings warnings={reviewed.warnings} />
        </>
      )}
    </Stack>
  );
}

const WARNING_LEVEL_TITLES: Readonly<Record<AdvisorWarning["level"], string>> =
  {
    error: "Errors",
    warning: "Warnings",
    info: "Notices",
  };

function Warnings({
  warnings,
}: {
  readonly warnings: readonly AdvisorWarning[];
}) {
  if (warnings.length === 0) return null;
  // Mantine's Alert already pairs its own background/text colors
  // accessibly per `color` -- one alert per severity level conveys the
  // level through that pairing instead of a per-item colored badge
  // nested inside an already-tinted alert, which axe flags for
  // insufficient contrast at small text sizes regardless of which
  // Mantine color/variant is chosen for the inner badge.
  return (
    <Stack gap="sm">
      {(["error", "warning", "info"] as const).map((level) => {
        const atLevel = warnings.filter((warning) => warning.level === level);
        if (atLevel.length === 0) return null;
        return (
          <Alert
            key={level}
            color={WARNING_LEVEL_COLORS[level]}
            variant="filled"
            autoContrast
            title={WARNING_LEVEL_TITLES[level]}
            role="status"
            aria-live="polite"
            aria-atomic="true"
          >
            <Stack component="ul" gap="xs" m={0} pl="md">
              {atLevel.map((warning) => (
                <Text component="li" key={warning.code} size="sm">
                  {warning.message}
                </Text>
              ))}
            </Stack>
          </Alert>
        );
      })}
    </Stack>
  );
}
