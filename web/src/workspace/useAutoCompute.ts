import { useEffect } from "react";

import type { Workspace, WorkspaceSnapshot } from "./workspace";

const AUTO_COMPUTE_DEBOUNCE_MS = 400;

/** There is no manual "generate"/"update recommendation" control in the
 * UI -- whatever structure is loaded should always drive what the
 * review/scientific-facts cards show, so a recommendation is computed
 * automatically (debounced) on load and after every later settings
 * change, reusing `outOfDate`'s existing staleness tracking to also
 * catch an edit that landed while a request was already in flight. A
 * live, undismissed `explain` failure blocks further auto-attempts --
 * otherwise a persistently failing structure would retry forever. */
export function useAutoCompute(
  workspace: Workspace,
  snapshot: WorkspaceSnapshot,
): void {
  const {
    draft,
    structureInput,
    operation,
    outOfDate,
    reviewed,
    failureOperation,
  } = snapshot;
  useEffect(() => {
    if (draft === null || structureInput === null) return;
    if (operation !== null) return;
    if (failureOperation === "explain") return;
    if (reviewed !== null && !outOfDate) return;

    const timer = setTimeout(() => {
      void workspace.dispatch({ type: "review.compute" });
    }, AUTO_COMPUTE_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
    };
  }, [
    workspace,
    draft,
    structureInput,
    operation,
    outOfDate,
    reviewed,
    failureOperation,
  ]);
}
