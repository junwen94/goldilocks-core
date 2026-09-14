import type { AdvisorWarning } from "../api/coreClient";

export function isAdvisorWarning(value: unknown): value is AdvisorWarning {
  return (
    typeof value === "object" &&
    value !== null &&
    "code" in value &&
    "level" in value &&
    "category" in value &&
    "message" in value &&
    typeof value.message === "string"
  );
}

export const WARNING_LEVEL_COLORS: Readonly<
  Record<AdvisorWarning["level"], string>
> = {
  info: "blue",
  warning: "yellow",
  error: "red",
};
