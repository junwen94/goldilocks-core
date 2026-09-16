import { useState } from "react";
import {
  NativeSelect,
  NumberInput,
  Stack,
  Text,
  Textarea,
  TextInput,
} from "@mantine/core";

import {
  automaticPlaceholder,
  fieldLabel,
  type OverrideFieldMeta,
  stringifyValue,
} from "./overrideFieldMeta";

/** One human/heuristic override control, generic over the field's
 * declared `type` -- shared by the advisors accordion (settings) and
 * the analysis section (facts), since both are the same
 * human > ml > llm > heuristic override shape underneath. */
export function OverrideControl({
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
