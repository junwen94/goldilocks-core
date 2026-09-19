import {
  Accordion,
  Badge,
  Checkbox,
  Fieldset,
  Group,
  NativeSelect,
  NumberInput,
  Paper,
  SimpleGrid,
  Stack,
  Text,
} from "@mantine/core";

import type {
  CalcTask,
  ExplainResult,
  PseudopotentialTable,
  ResolvedField,
  Setting,
  StructureInspection,
} from "../api/coreClient";
import { ScientificRecord } from "../review/ScientificRecord";
import { useWorkspace, useWorkspaceSnapshot } from "../workspace/useWorkspace";
import { OverrideControl } from "./OverrideControl";

/** Groups whose overrides only make sense for a subset of tasks.
 * `capabilities.py`'s own `Setting.tasks` field is always `null` today
 * (not wired -- see its docstring), so this is a small, honestly-scoped
 * frontend-only allowlist rather than something derived from the API;
 * a setting outside this map is shown regardless of task. */
const GROUP_TASK_RELEVANCE: Readonly<Record<string, readonly CalcTask[]>> = {
  relax: ["relax", "vc-relax"],
};

const STATUS_COLORS: Readonly<Record<ResolvedField["status"], string>> = {
  resolved: "green",
  unavailable: "yellow",
  blocked: "red",
};

/** Code/Task/HPC profile: which target this calculation is even for, as
 * opposed to `CalculationForm` below's per-step *settings* for that
 * target. Lives in `StructureCard` (v2 epic 12 follow-up) -- right
 * between loading a structure and viewing it, since it's the other
 * half of "what am I calculating" alongside the structure itself,
 * rather than a settings-tuning concern like the accordion is. */
export function CalculationContextControls() {
  const workspace = useWorkspace();
  const snapshot = useWorkspaceSnapshot();
  const { draft, capabilities, inspection } = snapshot;
  if (draft === null || capabilities === null) {
    return null;
  }

  const disabled = snapshot.operation === "inspect" || inspection === null;

  return (
    <Stack gap="sm">
      <NativeSelect
        label="Code"
        disabled={disabled}
        value={draft.code}
        data={capabilities.codes.map((code) => ({
          value: code.id,
          label: code.name,
        }))}
        onChange={(event) => {
          void workspace.dispatch({
            type: "draft.patch",
            code: event.currentTarget.value,
          });
        }}
      />
      <NativeSelect
        label="Task"
        disabled={disabled}
        value={draft.task}
        data={capabilities.tasks.map((task) => ({
          value: task.id,
          label: task.name,
        }))}
        onChange={(event) => {
          void workspace.dispatch({
            type: "draft.patch",
            task: event.currentTarget.value as CalcTask,
          });
        }}
      />

      <NativeSelect
        label="HPC profile"
        disabled={disabled}
        value={draft.hpc ?? ""}
        data={[
          { value: "", label: "Automatic" },
          ...capabilities.hpc_profiles.map((profile) => ({
            value: profile.id,
            label: profile.name,
          })),
        ]}
        onChange={(event) => {
          void workspace.dispatch({
            type: "draft.patch",
            hpc: event.currentTarget.value || null,
          });
        }}
      />
    </Stack>
  );
}

export function CalculationForm() {
  const workspace = useWorkspace();
  const snapshot = useWorkspaceSnapshot();
  const { draft, capabilities, inspection, reviewed } = snapshot;
  if (draft === null || capabilities === null) {
    return null;
  }

  const inspecting = snapshot.operation === "inspect";
  const disabled = inspecting || inspection === null;

  const elements = inspection === null ? [] : uniqueElements(inspection);
  const overrides = draft.overrides;

  function patchOverrides(patch: Readonly<Record<string, unknown>>): void {
    void workspace.dispatch({ type: "draft.patch", overrides: patch });
  }

  return (
    <Stack>
      {inspecting ? (
        <Text c="red" size="sm" role="status">
          Calculation settings are disabled while the new structure loads.
        </Text>
      ) : null}
      {inspection === null ? (
        <Text c="dimmed" size="sm">
          Showing default settings — load a structure to configure a
          calculation.
        </Text>
      ) : null}

      <Accordion multiple transitionDuration={0}>
        <SettingsGroupItems
          settings={capabilities.settings}
          task={draft.task}
          overrides={overrides}
          reviewed={reviewed}
          elements={elements}
          pseudopotentialTables={capabilities.pseudopotential_tables}
          disabled={disabled}
          onChange={patchOverrides}
        />
      </Accordion>
    </Stack>
  );
}

function uniqueElements(inspection: StructureInspection): string[] {
  return [
    ...new Set(
      inspection.structure.sites.flatMap((site) =>
        site.species.map((species) => species.symbol),
      ),
    ),
  ];
}

function uniqueFunctionals(tables: readonly PseudopotentialTable[]): string[] {
  return [...new Set(tables.map((table) => table.functional))];
}

function PseudoTableControl({
  tables,
  elements,
  functionalOverride,
  pinned,
  disabled,
  onChange,
}: {
  readonly tables: readonly PseudopotentialTable[];
  readonly elements: readonly string[];
  readonly functionalOverride: string | undefined;
  readonly pinned: unknown;
  readonly disabled: boolean;
  readonly onChange: (patch: Readonly<Record<string, unknown>>) => void;
}) {
  const matching = tables.filter(
    (table) =>
      (functionalOverride === undefined ||
        table.functional === functionalOverride) &&
      elements.every((element) => table.elements.includes(element)),
  );
  return (
    <NativeSelect
      label="Pseudopotential table"
      description={
        matching.length === 0
          ? "No registered table supports this structure with this functional"
          : undefined
      }
      disabled={disabled}
      value={typeof pinned === "string" ? pinned : ""}
      data={[
        { value: "", label: "Automatic" },
        ...matching.map((table) => ({
          value: table.id,
          label: `${table.provider} · ${table.functional} · ${table.accuracy} · ${table.relativistic}`,
        })),
      ]}
      onChange={(event) => {
        const raw = event.currentTarget.value;
        onChange({ pseudo_table_id: raw === "" ? undefined : raw });
      }}
    />
  );
}

function FunctionalControl({
  tables,
  value,
  disabled,
  onChange,
}: {
  readonly tables: readonly PseudopotentialTable[];
  readonly value: string | undefined;
  readonly disabled: boolean;
  readonly onChange: (patch: Readonly<Record<string, unknown>>) => void;
}) {
  return (
    <NativeSelect
      label="Functional"
      disabled={disabled}
      value={value ?? ""}
      data={[
        { value: "", label: "Automatic" },
        ...uniqueFunctionals(tables).map((functional) => ({
          value: functional,
          label: functional,
        })),
      ]}
      onChange={(event) => {
        const raw = event.currentTarget.value;
        // A pinned table pinned under the old functional may no
        // longer be a valid choice (and won't even appear in the
        // now-refiltered dropdown) -- clear it rather than silently
        // keep submitting a now-invisible override, matching how a
        // pseudopotential-table pin has always been invalidated by
        // changing the functional it was chosen under.
        onChange({
          functional: raw === "" ? undefined : raw,
          pseudo_table_id: undefined,
        });
      }}
    />
  );
}

/** `pseudo_table_id` is the settings group's internal name; the
 * accordion header shows the same human label as the control it
 * contains instead of a literal, ID-suffixed rendering of the group
 * name. */
const GROUP_DISPLAY_NAMES: Readonly<Record<string, string>> = {
  pseudo_table_id: "Pseudopotential table",
};

function humanizeGroup(group: string): string {
  const override = GROUP_DISPLAY_NAMES[group];
  if (override !== undefined) return override;
  const spaced = group.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/** A settings group's own record key doesn't always equal the group
 * name verbatim (`pseudo_table_id` the group vs `pseudo_table` the
 * record, see recordGroups.ts) -- this is the one alias the advisors
 * accordion needs to look its own resolved value up. */
function recordKeyForGroup(group: string): string {
  return group === "pseudo_table_id" ? "pseudo_table" : group;
}

function SettingsGroupItems({
  settings,
  task,
  overrides,
  reviewed,
  elements,
  pseudopotentialTables,
  disabled,
  onChange,
}: {
  readonly settings: readonly Setting[];
  readonly task: CalcTask;
  readonly overrides: Readonly<Record<string, unknown>>;
  readonly reviewed: ExplainResult | null;
  readonly elements: readonly string[];
  readonly pseudopotentialTables: readonly PseudopotentialTable[];
  readonly disabled: boolean;
  readonly onChange: (patch: Readonly<Record<string, unknown>>) => void;
}) {
  const groups = new Map<string, Setting[]>();
  for (const setting of settings) {
    const relevantTasks = GROUP_TASK_RELEVANCE[setting.group];
    if (relevantTasks !== undefined && !relevantTasks.includes(task)) {
      continue;
    }
    const bucket = groups.get(setting.group);
    if (bucket === undefined) {
      groups.set(setting.group, [setting]);
    } else {
      bucket.push(setting);
    }
  }

  const functionalOverride =
    typeof overrides.functional === "string" ? overrides.functional : undefined;

  return (
    <>
      {[...groups.entries()].map(([group, groupSettings]) => {
        const record = reviewed?.records[recordKeyForGroup(group)];
        return (
          <Accordion.Item key={group} value={group}>
            <Accordion.Control>
              <Group gap="xs" wrap="nowrap">
                {record === undefined ? null : (
                  <Badge
                    size="xs"
                    circle
                    color={STATUS_COLORS[record.status]}
                    aria-hidden="true"
                  />
                )}
                <span>{humanizeGroup(group)}</span>
              </Group>
            </Accordion.Control>
            <Accordion.Panel>
              <Stack gap="sm">
                {record === undefined ? null : (
                  <Paper withBorder p="xs" bg="var(--mantine-color-default)">
                    <ScientificRecord field={record} />
                  </Paper>
                )}
                <GroupBody
                  group={group}
                  groupSettings={groupSettings}
                  elements={elements}
                  pseudopotentialTables={pseudopotentialTables}
                  functionalOverride={functionalOverride}
                  overrides={overrides}
                  disabled={disabled}
                  onChange={onChange}
                />
              </Stack>
            </Accordion.Panel>
          </Accordion.Item>
        );
      })}
    </>
  );
}

function GroupBody({
  group,
  groupSettings,
  elements,
  pseudopotentialTables,
  functionalOverride,
  overrides,
  disabled,
  onChange,
}: {
  readonly group: string;
  readonly groupSettings: readonly Setting[];
  readonly elements: readonly string[];
  readonly pseudopotentialTables: readonly PseudopotentialTable[];
  readonly functionalOverride: string | undefined;
  readonly overrides: Readonly<Record<string, unknown>>;
  readonly disabled: boolean;
  readonly onChange: (patch: Readonly<Record<string, unknown>>) => void;
}) {
  if (group === "functional") {
    return (
      <FunctionalControl
        tables={pseudopotentialTables}
        value={functionalOverride}
        disabled={disabled}
        onChange={onChange}
      />
    );
  }
  if (group === "pseudo_table_id") {
    return (
      <PseudoTableControl
        tables={pseudopotentialTables}
        elements={elements}
        functionalOverride={functionalOverride}
        pinned={overrides.pseudo_table_id}
        disabled={disabled}
        onChange={onChange}
      />
    );
  }
  return (
    <>
      {groupSettings.map((setting) =>
        setting.key === "k_grid" ? (
          <KGridControl
            key={setting.key}
            value={overrides.k_grid}
            disabled={disabled}
            onChange={(value) => {
              onChange({ k_grid: value });
            }}
          />
        ) : (
          <OverrideControl
            key={setting.key}
            meta={setting}
            value={overrides[setting.key]}
            disabled={disabled}
            onChange={(value) => {
              onChange({ [setting.key]: value });
            }}
          />
        ),
      )}
    </>
  );
}

function KGridControl({
  value,
  disabled,
  onChange,
}: {
  readonly value: unknown;
  readonly disabled: boolean;
  readonly onChange: (value: unknown) => void;
}) {
  const grid =
    Array.isArray(value) && value.length === 3
      ? (value as [number, number, number])
      : null;
  return (
    <Fieldset legend="K-point grid">
      <Checkbox
        label="Set an explicit grid"
        checked={grid !== null}
        disabled={disabled}
        onChange={(event) => {
          onChange(event.currentTarget.checked ? [1, 1, 1] : undefined);
        }}
      />
      <SimpleGrid cols={3} mt="sm" spacing="xs">
        {(["x", "y", "z"] as const).map((axis, index) => (
          <NumberInput
            key={axis}
            aria-label={`K-point grid ${axis}`}
            min={1}
            max={99}
            disabled={grid === null || disabled}
            value={grid?.[index] ?? ""}
            placeholder="Auto"
            onChange={(raw) => {
              if (grid === null) return;
              const parsed = Number(raw);
              if (!Number.isInteger(parsed) || parsed < 1 || parsed > 99)
                return;
              const next: [number, number, number] = [...grid];
              next[index] = parsed;
              onChange(next);
            }}
          />
        ))}
      </SimpleGrid>
    </Fieldset>
  );
}
