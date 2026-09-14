import { Alert, Button, Group, Text } from "@mantine/core";
import { RotateCw } from "lucide-react";

import type { CoreFailure } from "../api/coreClient";

export function FailureBanner({
  failure,
  retryAvailable,
  dismissAvailable,
  onRetry,
  onDismiss,
}: {
  readonly failure: CoreFailure;
  readonly retryAvailable: boolean;
  readonly dismissAvailable: boolean;
  readonly onRetry: () => void;
  readonly onDismiss: () => void;
}) {
  return (
    <Alert
      color="red"
      title={FAILURE_TITLES[failure.kind] ?? "Calculation failed"}
      withCloseButton={dismissAvailable}
      closeButtonLabel="Dismiss error"
      styles={{ closeButton: { width: 44, height: 44 } }}
      onClose={onDismiss}
      role="alert"
    >
      <Group justify="space-between">
        <Text flex={1}>{failure.message}</Text>
        {retryAvailable ? (
          <Button
            color="red"
            variant="light"
            onClick={onRetry}
            leftSection={<RotateCw aria-hidden="true" size={15} />}
          >
            Retry
          </Button>
        ) : null}
      </Group>
    </Alert>
  );
}

// Kept in sync with every ExpectedFailure.kind reachable through the
// v2 HTTP transport (grep `kind = "..."` across src/goldilocks_core/),
// plus the client-only kinds coreClient.ts synthesizes itself. An
// unrecognized kind still degrades to the generic fallback title below
// rather than failing loudly -- acceptable per #12's own scope note,
// since `failure.message` is always shown regardless.
const FAILURE_TITLES: Readonly<Record<string, string>> = {
  invalid_request: "Check the request",
  invalid_setting: "Check your overrides",
  advice_incomplete: "Recommendation incomplete",
  invalid_hpc_profile: "Check the HPC profile",
  invalid_structure: "Check the structure",
  assets_unavailable: "Runtime assets unavailable",
  asset_not_installed: "Runtime assets unavailable",
  asset_corrupt: "Runtime assets unavailable",
  generation_error: "Could not generate input files",
  submission_error: "Could not generate the submission script",
  network_error: "Cannot reach Core",
  invalid_response: "Unexpected server response",
  http_error: "Request failed",
  incompatible_vocabulary: "Frontend/backend version mismatch",
};
