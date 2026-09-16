import type { Capabilities } from "../api/coreClient";

/** Which side of the UI a resolved record belongs on, mirroring
 * goldilocks-core's own analysis (structure-only) vs advisors
 * (per-calculation decision) architecture. Verified against a real
 * `goldilocks explain --json` run rather than assumed: a record's key
 * usually equals a `Setting.group`, except `pseudo_table_id` (the
 * setting group) vs `pseudo_table` (the record key), and
 * `pseudopotentials` (a record with no settings-group counterpart at
 * all, tied to the pseudo-selection advisor). Everything else -- the
 * four `capabilities.facts` keys, and analysis-only outputs like
 * composition/geometry/symmetry that have no override control anywhere
 * -- is an analysis-tier record. */
export function isAdvisorRecordKey(
  key: string,
  capabilities: Capabilities,
): boolean {
  if (key === "pseudo_table" || key === "pseudopotentials") return true;
  return capabilities.settings.some((setting) => setting.group === key);
}
