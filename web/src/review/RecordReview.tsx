import {
  Accordion,
  Badge,
  Code,
  Group,
  Stack,
  Text,
  Title,
} from "@mantine/core";

import type { ResolvedField } from "../api/coreClient";
import { ScientificRecord } from "./ScientificRecord";

const STATUS_COLORS: Record<ResolvedField["status"], string> = {
  resolved: "green",
  unavailable: "yellow",
  blocked: "red",
};

export function RecordReview({
  records,
}: {
  readonly records: Readonly<Record<string, ResolvedField>>;
}) {
  const names = Object.keys(records).sort();
  return (
    <Stack component="section" gap="xs" miw={0}>
      <Group component="header" justify="space-between">
        <Title order={3}>Scientific records</Title>
        <Text size="sm" c="dimmed">
          {names.length} records
        </Text>
      </Group>
      <Accordion multiple order={4}>
        {names.map((name) => {
          const field = records[name];
          if (field === undefined) return null;
          return (
            <Accordion.Item key={name} value={name} className="record-card">
              <Accordion.Control>
                <Group justify="space-between" wrap="nowrap">
                  <Group gap="xs" wrap="nowrap">
                    <Badge
                      size="xs"
                      circle
                      color={STATUS_COLORS[field.status]}
                      aria-label={`Status: ${field.status}`}
                    />
                    <span>{readableName(name)}</span>
                  </Group>
                  <Code>{name}</Code>
                </Group>
              </Accordion.Control>
              <Accordion.Panel>
                <ScientificRecord field={field} />
              </Accordion.Panel>
            </Accordion.Item>
          );
        })}
      </Accordion>
    </Stack>
  );
}

function readableName(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}
