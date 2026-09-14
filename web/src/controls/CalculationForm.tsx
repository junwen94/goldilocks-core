import { useState } from "react";
import {
  Accordion,
  Button,
  Checkbox,
  Fieldset,
  NativeSelect,
  NumberInput,
  SimpleGrid,
  Stack,
  Text,
  Textarea,
  TextInput,
} from "@mantine/core";
import { ArrowRight } from "lucide-react";

import type {
  CalcTask,
  Fact,
  PseudopotentialTable,
  Setting,
  StructureInspection,
} from "../api/coreClient";
import { useWorkspace, useWorkspaceSnapshot } from "../workspace/useWorkspace";

/** Groups whose overrides only make sense for a subset of tasks.
 * `capabilities.py`'s own `Setting.tasks` field is always `null` today
 * (not wired -- see its docstring), so this is a small, honestly-scoped
 * frontend-only allowlist rather than something derived from the API;
 * a setting outside this map is shown regardless of task. */
const GROUP_TASK_RELEVANCE: Readonly<Record<string, readonly CalcTask[]>> = {
  relax: ["relax", "vc-relax"],
};

export function CalculationForm({
  onShowRecommendation,
}: {
  readonly onShowRecommendation: () => void;
}) {
  const workspace = useWorkspace();
  const snapshot = useWorkspaceSnapshot();
  const { draft, capabilities, inspection } = snapshot;
  if (draft === null || capabilities === null || inspection === null) {
    return null;
  }

  const inspecting = snapshot.operation === "inspect";
  const busy = snapshot.operation !== null;
  let submitLabel =
    snapshot.reviewed === null
      ? "Generate recommendation"
      : "Update recommendation";
  if (snapshot.operation === "explain") submitLabel = "Computing";

  const elements = uniqueElements(inspection);
  const overrides = draft.overrides;
  const functionalOverride =
    typeof overrides.functional === "string" ? overrides.functional : undefined;

  function patchOverrides(patch: Readonly<Record<string, unknown>>): void {
    void workspace.dispatch({ type: "draft.patch", overrides: patch });
  }

  return (
    <Stack
      component="form"
      onSubmit={(event) => {
        event.preventDefault();
        onShowRecommendation();
        void workspace.dispatch({ type: "review.compute" });
      }}
    >
      {inspecting ? (
        <Text c="red" size="sm" role="status">
          Calculation settings are disabled while the new structure loads.
        </Text>
      ) : null}

      <SimpleGrid cols={{ base: 1, xs: 2 }}>
        <NativeSelect
          label="Task"
          disabled={inspecting}
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
          disabled={inspecting}
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
      </SimpleGrid>

      <SimpleGrid cols={{ base: 1, xs: 2 }}>
        <NativeSelect
          label="Functional"
          disabled={inspecting}
          value={functionalOverride ?? ""}
          data={[
            { value: "", label: "Automatic" },
            ...uniqueFunctionals(capabilities.pseudopotential_tables).map(
              (functional) => ({ value: functional, label: functional }),
            ),
          ]}
          onChange={(event) => {
            const raw = event.currentTarget.value;
            // A pinned table pinned under the old functional may no
            // longer be a valid choice (and won't even appear in the
            // now-refiltered dropdown) -- clear it rather than silently
            // keep submitting a now-invisible override, matching how a
            // pseudopotential-table pin has always been invalidated by
            // changing the functional it was chosen under.
            patchOverrides({
              functional: raw === "" ? undefined : raw,
              pseudo_table_id: undefined,
            });
          }}
        />
        <PseudoTableControl
          tables={capabilities.pseudopotential_tables}
          elements={elements}
          functionalOverride={functionalOverride}
          pinned={overrides.pseudo_table_id}
          disabled={inspecting}
          onChange={patchOverrides}
        />
      </SimpleGrid>

      <FactsSection
        facts={capabilities.facts}
        overrides={overrides}
        disabled={inspecting}
        onChange={patchOverrides}
      />

      <SettingsGroups
        settings={capabilities.settings}
        task={draft.task}
        overrides={overrides}
        disabled={inspecting}
        onChange={patchOverrides}
      />

      <Button
        type="submit"
        fullWidth
        rightSection={<ArrowRight aria-hidden="true" size={14} />}
        disabled={busy}
      >
        {submitLabel}
      </Button>
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

function FactsSection({
  facts,
  overrides,
  disabled,
  onChange,
}: {
  readonly facts: readonly Fact[];
  readonly overrides: Readonly<Record<string, unknown>>;
  readonly disabled: boolean;
  readonly onChange: (patch: Readonly<Record<string, unknown>>) => void;
}) {
  if (facts.length === 0) return null;
  return (
    <Fieldset legend="Structure facts">
      <Stack gap="sm">
        {facts.map((fact) => (
          <OverrideControl
            key={fact.key}
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
              onChange({ [fact.key]: value });
            }}
          />
        ))}
      </Stack>
    </Fieldset>
  );
}

function humanizeGroup(group: string): string {
  const spaced = group.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

function SettingsGroups({
  settings,
  task,
  overrides,
  disabled,
  onChange,
}: {
  readonly settings: readonly Setting[];
  readonly task: CalcTask;
  readonly overrides: Readonly<Record<string, unknown>>;
  readonly disabled: boolean;
  readonly onChange: (patch: Readonly<Record<string, unknown>>) => void;
}) {
  const groups = new Map<string, Setting[]>();
  for (const setting of settings) {
    // functional/pseudo_table_id are rendered above, next to the task
    // selector, since they're the two most commonly tuned system knobs.
    if (setting.key === "functional" || setting.key === "pseudo_table_id") {
      continue;
    }
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
  if (groups.size === 0) return null;

  return (
    <Accordion multiple transitionDuration={0}>
      {[...groups.entries()].map(([group, groupSettings]) => (
        <Accordion.Item key={group} value={group}>
          <Accordion.Control>{humanizeGroup(group)}</Accordion.Control>
          <Accordion.Panel>
            <Stack gap="sm">
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
            </Stack>
          </Accordion.Panel>
        </Accordion.Item>
      ))}
    </Accordion>
  );
}

interface OverrideFieldMeta {
  readonly key: string;
  readonly type: string;
  readonly enum?: readonly string[] | undefined;
  readonly unit?: string | null;
  readonly default?: unknown;
  readonly description: string;
}

function fieldLabel(meta: OverrideFieldMeta): string {
  const humanized = meta.key.replace(/_/g, " ");
  return meta.unit ? `${humanized} · ${meta.unit}` : humanized;
}

/** `unknown`-safe stringification -- most override values are
 * primitives, but array/object-shaped ones (`k_grid`, `u_by_element`)
 * would otherwise stringify to the useless `"[object Object]"`. */
function stringifyValue(value: unknown): string {
  return typeof value === "object" && value !== null
    ? JSON.stringify(value)
    : String(value);
}

function automaticPlaceholder(meta: OverrideFieldMeta): string {
  return meta.default === undefined
    ? "Automatic"
    : `Automatic (${stringifyValue(meta.default)})`;
}

function OverrideControl({
  meta,
  value,
  disabled,
  onChange,
}: {
  readonly meta: OverrideFieldMeta;
  readonly value: unknown;
  readonly disabled: boolean;
  readonly onChange: (value: unknown) => void;
}) {
  const pinned = value !== undefined;
  const label = fieldLabel(meta);

  if (meta.type === "boolean") {
    return (
      <NativeSelect
        label={label}
        description={meta.description}
        disabled={disabled}
        value={pinned ? stringifyValue(value) : ""}
        data={[
          { value: "", label: automaticPlaceholder(meta) },
          { value: "true", label: "On" },
          { value: "false", label: "Off" },
        ]}
        onChange={(event) => {
          const raw = event.currentTarget.value;
          onChange(raw === "" ? undefined : raw === "true");
        }}
      />
    );
  }

  if (meta.enum && meta.enum.length > 0) {
    return (
      <NativeSelect
        label={label}
        description={meta.description}
        disabled={disabled}
        value={pinned ? stringifyValue(value) : ""}
        data={[
          { value: "", label: automaticPlaceholder(meta) },
          ...meta.enum.map((option) => ({ value: option, label: option })),
        ]}
        onChange={(event) => {
          const raw = event.currentTarget.value;
          onChange(raw === "" ? undefined : raw);
        }}
      />
    );
  }

  if (meta.type === "integer" || meta.type === "number") {
    return (
      <NumberInput
        label={label}
        description={meta.description}
        disabled={disabled}
        placeholder={automaticPlaceholder(meta)}
        value={pinned ? (value as number) : ""}
        onChange={(raw) => {
          if (raw === "") {
            onChange(undefined);
            return;
          }
          const parsed = Number(raw);
          if (Number.isFinite(parsed)) onChange(parsed);
        }}
      />
    );
  }

  if (meta.type === "string") {
    return (
      <TextInput
        label={label}
        description={meta.description}
        disabled={disabled}
        placeholder={automaticPlaceholder(meta)}
        value={pinned ? stringifyValue(value) : ""}
        onChange={(event) => {
          const raw = event.currentTarget.value;
          onChange(raw === "" ? undefined : raw);
        }}
      />
    );
  }

  return (
    <JsonOverrideControl
      label={label}
      description={meta.description}
      disabled={disabled}
      value={value}
      onChange={onChange}
    />
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

function JsonOverrideControl({
  label,
  description,
  disabled,
  value,
  onChange,
}: {
  readonly label: string;
  readonly description: string;
  readonly disabled: boolean;
  readonly value: unknown;
  readonly onChange: (value: unknown) => void;
}) {
  const [lastValue, setLastValue] = useState(value);
  const [text, setText] = useState(
    value === undefined ? "" : JSON.stringify(value),
  );
  const [error, setError] = useState<string | null>(null);
  // Resync the editable text when `value` changes for a reason other than
  // this control's own onChange (e.g. cleared elsewhere, structure
  // reopened) -- adjusting state during render, not in an effect, per
  // https://react.dev/learn/you-might-not-need-an-effect#adjusting-some-state-when-a-prop-changes.
  if (value !== lastValue) {
    setLastValue(value);
    setText(value === undefined ? "" : JSON.stringify(value));
    setError(null);
  }

  return (
    <Stack gap={4}>
      <Textarea
        label={label}
        description={`${description} (JSON)`}
        disabled={disabled}
        placeholder="Automatic"
        autosize
        minRows={1}
        value={text}
        onChange={(event) => {
          const raw = event.currentTarget.value;
          setText(raw);
          if (raw.trim() === "") {
            setError(null);
            onChange(undefined);
            return;
          }
          try {
            onChange(JSON.parse(raw) as unknown);
            setError(null);
          } catch {
            setError("Not valid JSON");
          }
        }}
      />
      {error === null ? null : (
        <Text c="red" size="xs">
          {error}
        </Text>
      )}
    </Stack>
  );
}
