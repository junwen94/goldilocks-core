import { Group, Paper, Text, Title } from "@mantine/core";

import { CalculationForm } from "../controls/CalculationForm";
import { useWorkspaceSnapshot } from "../workspace/useWorkspace";

export function AdvisorsCard({ kicker }: { readonly kicker: string }) {
  const snapshot = useWorkspaceSnapshot();
  return (
    <Paper
      component="section"
      id="advisors-panel"
      aria-label="Goldilocks advisors"
      withBorder
      p="md"
      className="workbench-card card-advisors"
    >
      <Group component="header" className="card-header" mb="md" wrap="nowrap">
        <Text className="card-kicker">{kicker}</Text>
        <Title order={2}>Advisors</Title>
      </Group>
      <div className="card-body">
        {snapshot.capabilities === null ? (
          <Text c="dimmed">Load a structure to configure a calculation.</Text>
        ) : (
          <CalculationForm />
        )}
      </div>
    </Paper>
  );
}
