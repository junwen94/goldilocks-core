import { Group, Paper, Text, Title } from "@mantine/core";

import { AnalysisSection } from "../analysis/AnalysisSection";
import { useWorkspaceSnapshot } from "../workspace/useWorkspace";

export function AnalysisCard() {
  const snapshot = useWorkspaceSnapshot();
  return (
    <Paper
      component="section"
      id="analysis-panel"
      aria-label="Goldilocks analysis"
      withBorder
      p="md"
      className="workbench-card card-analysis"
    >
      <Group component="header" className="card-header" mb="md" wrap="nowrap">
        <Text className="card-kicker">02</Text>
        <Title order={2}>Goldilocks analysis</Title>
      </Group>
      <div className="card-body">
        {snapshot.capabilities === null ? (
          <Text c="dimmed">Load a structure to see the Goldilocks analysis.</Text>
        ) : (
          <AnalysisSection />
        )}
      </div>
    </Paper>
  );
}
