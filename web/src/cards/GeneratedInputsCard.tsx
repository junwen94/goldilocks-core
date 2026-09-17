import {
  Alert,
  Button,
  Group,
  Loader,
  Paper,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import { Download } from "lucide-react";

import { GeneratedInputReview } from "../review/GeneratedInputReview";
import { WarningsPanel } from "../review/WarningsPanel";
import { useWorkspace, useWorkspaceSnapshot } from "../workspace/useWorkspace";

export function GeneratedInputsCard() {
  const workspace = useWorkspace();
  const snapshot = useWorkspaceSnapshot();
  const reviewed = snapshot.reviewed;

  return (
    <Paper
      component="section"
      id="generated-inputs-panel"
      aria-label="Generated input files"
      aria-busy={snapshot.operation === "explain"}
      withBorder
      p="md"
      className="workbench-card card-inputs"
    >
      <Group component="header" className="card-header" mb="md" wrap="nowrap">
        <Text className="card-kicker">03</Text>
        <Title order={2}>Generation of input files</Title>
      </Group>
      <div className="card-body">
        {reviewed === null ? (
          <Stack
            align="center"
            justify="center"
            mih={160}
            role={snapshot.operation === "explain" ? "status" : undefined}
          >
            {snapshot.operation === "explain" && <Loader size="sm" />}
            <Text fw={600}>
              {snapshot.operation === "explain"
                ? "Computing recommendation"
                : "No recommendation yet"}
            </Text>
            {snapshot.operation === "explain" ? null : (
              <Text c="dimmed" size="sm">
                Load a structure to generate input files.
              </Text>
            )}
          </Stack>
        ) : (
          <Stack gap="lg">
            {snapshot.outOfDate && (
              <Alert
                role="status"
                aria-label="Recommendation notice"
                aria-live="polite"
                aria-atomic="true"
              >
                Settings changed — recomputing the recommendation automatically.
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
                Download (.zip)
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
            <WarningsPanel warnings={reviewed.warnings} />
          </Stack>
        )}
      </div>
    </Paper>
  );
}
