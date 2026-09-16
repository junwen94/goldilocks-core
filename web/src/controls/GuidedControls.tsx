import { Group, Paper, Stack, Text, Title } from "@mantine/core";

import { useWorkspace, useWorkspaceSnapshot } from "../workspace/useWorkspace";
import { CalculationForm } from "./CalculationForm";
import { StructureSourceControls } from "./StructureSourceControls";

export function GuidedControls({
  onShowStructure,
  onShowRecommendation,
}: {
  readonly onShowStructure: () => void;
  readonly onShowRecommendation: () => void;
}) {
  const workspace = useWorkspace();
  const snapshot = useWorkspaceSnapshot();
  return (
    <Stack
      component="section"
      id="calculation-panel"
      aria-label="Calculation setup"
      gap={0}
      miw={0}
    >
      <Paper component="section" withBorder p="md">
        <Group component="header" mb="md" wrap="nowrap">
          <Text c="dimmed">01</Text>
          <Title order={2}>Structure</Title>
        </Group>
        <StructureSourceControls
          source={snapshot.structureInput}
          inspection={snapshot.inspection}
          inspecting={snapshot.operation === "inspect"}
          onOpen={(input) => {
            onShowStructure();
            return workspace.dispatch({ type: "source.open", input });
          }}
        />
      </Paper>

      <Paper component="section" withBorder p="md">
        <Group component="header" mb="md" wrap="nowrap">
          <Text c="dimmed">02</Text>
          <Title order={2}>Calculation</Title>
        </Group>
        <CalculationForm onShowRecommendation={onShowRecommendation} />
      </Paper>
    </Stack>
  );
}
