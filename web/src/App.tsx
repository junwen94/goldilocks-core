import { useEffect } from "react";
import type { CSSProperties } from "react";
import {
  MantineProvider,
  VisuallyHidden,
  useComputedColorScheme,
  useMantineColorScheme,
} from "@mantine/core";

import { AdvisorsCard } from "./cards/AdvisorsCard";
import { AnalysisCard } from "./cards/AnalysisCard";
import { BundleCard } from "./cards/BundleCard";
import { MagneticOrderingsCard } from "./cards/MagneticOrderingsCard";
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

  // Magnetic-ordering exploration is only meaningful once the structure
  // is actually classified magnetic -- rather than a fifth column that's
  // always present but usually empty, it pops in between Advisors and
  // Bundle exactly when that classification resolves true, and the grid
  // itself grows from four to five columns to match.
  const showMagneticOrderings =
    snapshot.reviewed?.records.is_magnetic?.value === "magnetic";
  const columns = showMagneticOrderings ? 5 : 4;

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

      <main
        className="workbench-grid"
        aria-labelledby="workbench-title"
        style={{ "--workbench-columns": columns } as CSSProperties}
      >
        <VisuallyHidden>
          <h1 id="workbench-title">Goldilocks SCF setup</h1>
        </VisuallyHidden>
        <StructureCard kicker="01" />
        <AnalysisCard kicker="02" />
        <AdvisorsCard kicker="03" />
        {showMagneticOrderings ? <MagneticOrderingsCard kicker="04" /> : null}
        <BundleCard kicker={showMagneticOrderings ? "05" : "04"} />
      </main>
    </>
  );
}
