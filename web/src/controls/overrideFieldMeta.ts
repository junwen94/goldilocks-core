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
