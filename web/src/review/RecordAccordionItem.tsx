import { Accordion, Badge, Code, Group } from "@mantine/core";

import type { ResolvedField } from "../api/coreClient";
import { readableName } from "./readableName";
import { ScientificRecord } from "./ScientificRecord";

const STATUS_COLORS: Record<ResolvedField["status"], string> = {
  resolved: "green",
  unavailable: "yellow",
  blocked: "red",
};

/** One resolved record as a collapsed-by-default accordion item --
 * status dot + human name in the always-visible control, full detail
 * (value, source, per-field sources) behind the expand. Shared by
 * every place a record is shown read-only (the analysis and advisors
 * sections both use this, not a bespoke presenter each). */
export function RecordAccordionItem({
  name,
  field,
}: {
  readonly name: string;
  readonly field: ResolvedField;
}) {
  return (
    <Accordion.Item value={name} className="record-card">
      <Accordion.Control>
        <Group justify="space-between" wrap="nowrap">
          <Group gap="xs" wrap="nowrap">
            <Badge
              size="xs"
              circle
              color={STATUS_COLORS[field.status]}
              aria-hidden="true"
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
}
