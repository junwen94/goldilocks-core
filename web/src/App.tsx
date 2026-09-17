import { useEffect } from "react";
import {
  MantineProvider,
  VisuallyHidden,
  useComputedColorScheme,
  useMantineColorScheme,
} from "@mantine/core";

import { AnalysisCard } from "./cards/AnalysisCard";
import { CalculationCard } from "./cards/CalculationCard";
import { GeneratedInputsCard } from "./cards/GeneratedInputsCard";
import { StructureCard } from "./cards/StructureCard";
import { AppHeader } from "./layout/AppHeader";
import { FailureBanner } from "./status/FailureBanner";
import { OperationStatus } from "./status/OperationStatus";
import { colorSchemeManager, workbenchTheme } from "./theme";
import { useAutoCompute } from "./workspace/useAutoCompute";
import { useWorkspace, useWorkspaceSnapshot } from "./workspace/useWorkspace";
import "./App.css";

export function App() {
  return (
    <MantineProvider
      theme={workbenchTheme}
      colorSchemeManager={colorSchemeManager}
      defaultColorScheme="light"
    >
      <Workbench />
    </MantineProvider>
  );
}

function Workbench() {
  const theme = useComputedColorScheme("light");
  const { toggleColorScheme: toggleTheme } = useMantineColorScheme();
  useEffect(() => {
    document
      .querySelector<HTMLMetaElement>('meta[name="theme-color"]')
      ?.setAttribute("content", theme === "light" ? "#ffffff" : "#242424");
  }, [theme]);

  return (
    <>
      <AppHeader theme={theme} onToggleTheme={toggleTheme} />
      <WorkbenchContent />
    </>
  );
}

// Exported separately from `Workbench` (not just used by it above) so a host
// app that already has its own page chrome -- e.g. goldilocks-agent's
// tab-switcher -- can mount the workbench content without a second,
// redundant <AppHeader>. `Workbench`/`App` (this repo's own standalone page)
// still render their own header; this is the header-less variant.
export function WorkbenchContent() {
  const workspace = useWorkspace();
  const snapshot = useWorkspaceSnapshot();
  useAutoCompute(workspace, snapshot);

  return (
    <>
      <OperationStatus
        operation={snapshot.operation}
        hasFailure={snapshot.failure !== null}
      />

      {snapshot.failure === null ? null : (
        <FailureBanner
          failure={snapshot.failure}
          retryAvailable={
            snapshot.failure.retryable ||
            snapshot.failureOperation === "capabilities"
          }
          dismissAvailable={snapshot.capabilities !== null}
          onRetry={() => {
            void workspace.dispatch({ type: "failure.retry" });
          }}
          onDismiss={() => {
            void workspace.dispatch({ type: "failure.dismiss" });
          }}
        />
      )}

      <main className="workbench-grid" aria-labelledby="workbench-title">
        <VisuallyHidden>
          <h1 id="workbench-title">Goldilocks SCF setup</h1>
        </VisuallyHidden>
        <StructureCard />
        <AnalysisCard />
        <CalculationCard />
        <GeneratedInputsCard />
      </main>
    </>
  );
}
