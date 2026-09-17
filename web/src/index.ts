// Library entry point for consumers outside this repo (currently:
// goldilocks-agent/app's Workbench tab -- see junwen94/goldilocks-agent#1).
// This is a *separate* build (vite.lib.config.ts / `npm run build:lib`) from
// the app entry (main.tsx) that ships in this repo's own Docker image --
// that build is untouched by this file.

import "@mantine/core/styles.layer.css";
import "./App.css";
import "./styles.css";

export { MantineProvider } from "@mantine/core";

// Full standalone page (used by this repo's own main.tsx) -- includes its
// own <AppHeader> and MantineProvider. A host app with its own page chrome
// should use `WorkbenchContent` instead.
export { App } from "./App";
// Header-less content only, for embedding inside a host app's own tab/panel
// chrome. The host must wrap it in `MantineProvider` (theme/colorSchemeManager
// exported below) and `WorkspaceProvider` (workspace exported below).
export { WorkbenchContent } from "./App";

export { colorSchemeManager, workbenchTheme } from "./theme";
export type { Theme } from "./theme";

export { WorkspaceProvider } from "./workspace/WorkspaceProvider";
export { createWorkspace } from "./workspace/workspace";
export type {
  CalculationDraft,
  Workspace,
  WorkspaceAction,
  WorkspaceOperation,
  WorkspaceSnapshot,
} from "./workspace/workspace";
export { useWorkspace, useWorkspaceSnapshot } from "./workspace/useWorkspace";

export { HttpCoreClient } from "./api/coreClient";
export type { CoreClient } from "./api/coreClient";
