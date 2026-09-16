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
    >
      <Group component="header" mb="md" wrap="nowrap">
        <Text c="dimmed">02</Text>
        <Title order={2}>Calculation</Title>
      </Group>
      {snapshot.inspection === null ? (
        <Text c="dimmed">Load a structure to configure a calculation.</Text>
      ) : (
        <CalculationForm />
      )}
    </Paper>
  );
}
