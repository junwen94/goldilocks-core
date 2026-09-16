import { Alert, Stack, Text } from "@mantine/core";

import type { AdvisorWarning } from "../api/coreClient";
import { WARNING_LEVEL_COLORS } from "./warnings";

const WARNING_LEVEL_TITLES: Readonly<Record<AdvisorWarning["level"], string>> =
  {
    error: "Errors",
    warning: "Warnings",
    info: "Notices",
  };

export function WarningsPanel({
  warnings,
}: {
  readonly warnings: readonly AdvisorWarning[];
}) {
  if (warnings.length === 0) return null;
  // Mantine's Alert already pairs its own background/text colors
  // accessibly per `color` -- one alert per severity level conveys the
  // level through that pairing instead of a per-item colored badge
  // nested inside an already-tinted alert, which axe flags for
  // insufficient contrast at small text sizes regardless of which
  // Mantine color/variant is chosen for the inner badge.
  return (
    <Stack gap="sm">
      {(["error", "warning", "info"] as const).map((level) => {
        const atLevel = warnings.filter((warning) => warning.level === level);
        if (atLevel.length === 0) return null;
        return (
          <Alert
            key={level}
            color={WARNING_LEVEL_COLORS[level]}
            variant="filled"
            autoContrast
            title={WARNING_LEVEL_TITLES[level]}
            role="status"
            aria-live="polite"
            aria-atomic="true"
          >
            <Stack component="ul" gap="xs" m={0} pl="md">
              {atLevel.map((warning) => (
                <Text component="li" key={warning.code} size="sm">
                  {warning.message}
                </Text>
              ))}
            </Stack>
          </Alert>
        );
      })}
    </Stack>
  );
}
