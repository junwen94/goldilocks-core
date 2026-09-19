import type { Source } from "../api/coreClient";
import { SOURCE_NAMES } from "../api/sourceNames";

export interface OverrideFieldMeta {
  readonly key: string;
  readonly type: string;
  readonly enum?: readonly string[] | undefined;
  readonly unit?: string | null;
  readonly default?: unknown;
  readonly description: string;
}

export function fieldLabel(meta: OverrideFieldMeta): string {
  const humanized = meta.key.replace(/_/g, " ");
  return meta.unit ? `${humanized} · ${meta.unit}` : humanized;
}

/** `unknown`-safe stringification -- most override values are
 * primitives, but array/object-shaped ones (`k_grid`, `u_by_element`)
 * would otherwise stringify to the useless `"[object Object]"`. */
export function stringifyValue(value: unknown): string {
  return typeof value === "object" && value !== null
    ? JSON.stringify(value)
    : String(value);
}

export function automaticPlaceholder(meta: OverrideFieldMeta): string {
  return meta.default === undefined
    ? "Automatic"
    : `Automatic (${stringifyValue(meta.default)})`;
}

/** A boolean/enum `OverrideControl`'s own description names whichever
 * tier actually produced the value the select shows -- "Your override"
 * once pinned, else whatever tier resolved it (heuristic/ml/llm),
 * else the field's own description alone when nothing has resolved yet. */
export function describeSource(
  description: string,
  pinned: boolean,
  source: Source | null | undefined,
): string {
  if (pinned) return `${description} · ${SOURCE_NAMES.human}`;
  if (source) return `${description} · ${SOURCE_NAMES[source]}`;
  return description;
}
