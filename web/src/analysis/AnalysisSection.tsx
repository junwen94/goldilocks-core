import { Accordion, Paper, Stack } from "@mantine/core";

import { OverrideControl } from "../controls/OverrideControl";
import { RecordAccordionItem } from "../review/RecordAccordionItem";
import { ScientificRecord } from "../review/ScientificRecord";
import { isAdvisorRecordKey } from "../workspace/recordGroups";
import { useWorkspace, useWorkspaceSnapshot } from "../workspace/useWorkspace";

/** goldilocks-core's own analysis tier only ever needs the structure --
 * `capabilities.facts` (is_metal/is_magnetic/needs_soc/needs_correlation,
 * each overridable) plus any other resolved record that isn't tied to a
 * settings group (composition/geometry/symmetry/... -- see
 * recordGroups.ts for exactly how that split is derived). Advisor
 * records live in the Advisors card instead, next to the override
 * control they belong to. Rendered inside `AnalysisCard`, which owns the
 * "Analysis" heading -- this component starts directly with its content. */
export function AnalysisSection() {
  const workspace = useWorkspace();
  const snapshot = useWorkspaceSnapshot();
  const { capabilities, draft, inspection, reviewed } = snapshot;
  if (capabilities === null || draft === null) {
    return null;
  }

  const disabled = snapshot.operation === "inspect" || inspection === null;
  const overrides = draft.overrides;
  const factKeys = new Set(capabilities.facts.map((fact) => fact.key));

  const analysisOnlyRecords = Object.entries(reviewed?.records ?? {}).filter(
    ([key]) => !factKeys.has(key) && !isAdvisorRecordKey(key, capabilities),
  );

  return (
    <Stack gap="sm">
      {capabilities.facts.length === 0 ? null : (
        <Stack gap="sm">
          {capabilities.facts.map((fact) => {
            const record = reviewed?.records[fact.key];
            return (
              <div key={fact.key}>
                {record === undefined ? null : (
                  <Paper
                    withBorder
                    p="xs"
                    mb={4}
                    bg="var(--mantine-color-default)"
                  >
                    <ScientificRecord field={record} />
                  </Paper>
                )}
                <OverrideControl
                  meta={{
                    key: fact.key,
                    type: fact.type,
                    enum: fact.values ?? undefined,
                    unit: null,
                    description: fact.description,
                  }}
                  value={overrides[fact.key]}
                  disabled={disabled}
                  onChange={(value) => {
                    void workspace.dispatch({
                      type: "draft.patch",
                      overrides: { [fact.key]: value },
                    });
                  }}
                />
              </div>
            );
          })}
        </Stack>
      )}
      {analysisOnlyRecords.length === 0 ? null : (
        <Accordion multiple transitionDuration={0}>
          {analysisOnlyRecords.map(([key, field]) => (
            <RecordAccordionItem key={key} name={key} field={field} />
          ))}
        </Accordion>
      )}
    </Stack>
  );
}
