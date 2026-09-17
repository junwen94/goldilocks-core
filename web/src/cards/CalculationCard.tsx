import { Group, Paper, Text, Title } from "@mantine/core";

import { CalculationForm } from "../controls/CalculationForm";
import { useWorkspaceSnapshot } from "../workspace/useWorkspace";

export function CalculationCard() {
  const snapshot = useWorkspaceSnapshot();
  return (
    <Paper
      component="section"
      id="calculation-panel"
      aria-label="Calculation setup"
      withBorder
      p="md"
      className="workbench-card card-calculation"
    >
      <Group component="header" className="card-header" mb="md" wrap="nowrap">
        <Text className="card-kicker">02</Text>
        <Title order={2}>Calculation</Title>
      </Group>
      <div className="card-body">
        {snapshot.inspection === null ? (
          <Text c="dimmed">Load a structure to configure a calculation.</Text>
        ) : (
          <CalculationForm />
        )}
      </div>
    </Paper>
  );
}
