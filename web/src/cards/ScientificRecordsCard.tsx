import { Group, Loader, Paper, Stack, Text, Title } from "@mantine/core";

import { RecordReview } from "../review/RecordReview";
import { useWorkspaceSnapshot } from "../workspace/useWorkspace";

export function ScientificRecordsCard() {
  const snapshot = useWorkspaceSnapshot();
  const reviewed = snapshot.reviewed;

  return (
    <Paper
      component="section"
      id="scientific-records-panel"
      aria-label="Scientific facts"
      aria-busy={snapshot.operation === "explain"}
      withBorder
      p="md"
      className="workbench-card card-records"
    >
      <Group component="header" className="card-header" mb="md" wrap="nowrap">
        <Text className="card-kicker">04</Text>
        <Title order={2}>Scientific facts</Title>
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
          </Stack>
        ) : (
          <RecordReview records={reviewed.records} />
        )}
      </div>
    </Paper>
  );
}
