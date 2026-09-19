import { useEffect } from "react";
import {
  MantineProvider,
  VisuallyHidden,
  useComputedColorScheme,
  useMantineColorScheme,
} from "@mantine/core";

import { AdvisorsCard } from "./cards/AdvisorsCard";
import { AnalysisCard } from "./cards/AnalysisCard";
import { BundleCard } from "./cards/BundleCard";
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
  const workspace = useWorkspace();
  const snapshot = useWorkspaceSnapshot();
  const theme = useComputedColorScheme("light");
  const { toggleColorScheme: toggleTheme } = useMantineColorScheme();
  useEffect(() => {
    document
      .querySelector<HTMLMetaElement>('meta[name="theme-color"]')
      ?.setAttribute("content", theme === "light" ? "#ffffff" : "#242424");
  }, [theme]);
  useAutoCompute(workspace, snapshot);

  return (
    <>
      <AppHeader theme={theme} onToggleTheme={toggleTheme} />
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
        <AdvisorsCard />
        <BundleCard />
      </main>
    </>
  );
}
