import { useId } from "react";
import {
  Alert,
  Button,
  Group,
  Loader,
  Paper,
  Stack,
  Text,
  Title,
  VisuallyHidden,
} from "@mantine/core";
import { Download, RotateCw } from "lucide-react";

import {
  type BlockingFinding,
  type ResolvedField,
  findingsOf,
} from "../api/coreClient";
import { GeneratedInputReview } from "../review/GeneratedInputReview";
import { WarningsPanel } from "../review/WarningsPanel";
import { FindingList } from "../status/FindingNote";
import {
  type BundleStatus,
  type DownloadState,
  bundleStatus,
  downloadState,
} from "../workspace/pipelineStatus";
import { useWorkspace, useWorkspaceSnapshot } from "../workspace/useWorkspace";
import { currentOrderings } from "../workspace/workspace";

export function BundleCard({ kicker }: { readonly kicker: string }) {
  const snapshot = useWorkspaceSnapshot();
  const status = bundleStatus(snapshot);
  const listed =
    (currentOrderings(snapshot)?.result?.candidates.length ?? 0) > 0;

  return (
    <Paper
      component="section"
      id="bundle-panel"
      aria-label="Bundle"
      aria-busy={snapshot.operation === "explain"}
      withBorder
      p="md"
      className="workbench-card card-bundle"
    >
      <Group component="header" className="card-header" mb="md" wrap="nowrap">
        <Text className="card-kicker">{kicker}</Text>
        <Title order={2}>Bundle</Title>
      </Group>
      {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- axe's scrollable-region-focusable rule requires this overflow-y:auto container itself to be keyboard-reachable, not just its children. */}
      <div className="card-body" tabIndex={0}>
        {snapshot.reviewed === null ? (
          <NoRecommendation status={status} />
        ) : (
          <Stack gap="lg">
            <BundleNotice status={status} />
            {listed ? (
              <BundledCell magnetic={snapshot.reviewed.records.magnetic} />
            ) : null}
            <DownloadRow
              status={status}
              download={downloadState(snapshot)}
              previewing={snapshot.operation === "preview"}
            />
            <GeneratedInputReview
              archive={status.kind === "ready" ? status.archive : null}
              placeholder={placeholder(status)}
            />
            <WarningsPanel warnings={snapshot.reviewed.warnings} />
          </Stack>
        )}
      </div>
    </Paper>
  );
}

/** The card before the open structure has any recommendation. */
function NoRecommendation({ status }: { readonly status: BundleStatus }) {
  if (status.kind === "explainFailed") return <BundleNotice status={status} />;
  const computing = status.kind === "computing";
  return (
    <Stack
      align="center"
      justify="center"
      mih={160}
      role={computing ? "status" : undefined}
    >
      {computing && <Loader size="sm" />}
      <Text fw={600}>
        {computing ? "Computing recommendation" : "No recommendation yet"}
      </Text>
      {computing ? null : (
        <Text c="dimmed" size="sm">
          Load a structure to generate input files.
        </Text>
      )}
    </Stack>
  );
}

/** What Bundle holds, said while Magnetic Orderings lists candidates for
 * the open structure (#293): downloading a selected ordering there never
 * changes the draft, so Bundle keeps the cell the magnetic record applied
 * -- the uploaded cell (`fm`), or the ordering a `magnetic_ordering`
 * setting applied -- and says its atom count. Nothing for a run without
 * spin polarization, which applies no ordering. */
function BundledCell({
  magnetic,
}: {
  readonly magnetic: ResolvedField | undefined;
}) {
  const cell = appliedCell(magnetic);
  if (cell === null) return null;
  const atoms = `${String(cell.nat)} atom${cell.nat === 1 ? "" : "s"}`;
  const held =
    cell.ordering === "fm"
      ? `Inputs for the uploaded cell (${atoms}), with starting magnetization per species.`
      : `Inputs for the ${cell.ordering} ordering (${atoms}).`;
  return (
    <Text size="sm">
      {held} To get another ordering&apos;s inputs, select it in Magnetic
      Orderings and download it there.
    </Text>
  );
}

/** The ordering label the magnetic record applied and the atom count of the
 * cell it writes; null when it applied none. */
function appliedCell(
  record: ResolvedField | undefined,
): { readonly ordering: string; readonly nat: number } | null {
  if (record?.status !== "resolved") return null;
  const value = record.value as {
    readonly ordering?: { readonly applied?: unknown } | null;
    readonly cell?: { readonly nat?: unknown } | null;
  } | null;
  const ordering = value?.ordering?.applied;
  const nat = value?.cell?.nat;
  return typeof ordering === "string" && typeof nat === "number"
    ? { ordering, nat }
    : null;
}

/** Why the current draft has no input files yet, when that needs saying: it
 * is being recomputed, its request failed (with Retry), or its `/explain`
 * verdict refuses it. A failure stays here after its banner is dismissed.
 * Findings, of the verdict or of a refused preview, are listed with a button
 * per remedy (#151). */
function BundleNotice({ status }: { readonly status: BundleStatus }) {
  const workspace = useWorkspace();
  switch (status.kind) {
    case "recomputing":
      return (
        <Alert
          role="status"
          aria-label="Recommendation notice"
          aria-live="polite"
          aria-atomic="true"
        >
          Settings changed — recomputing the recommendation automatically.
        </Alert>
      );
    case "explainFailed":
    case "generationFailed": {
      const operation = status.kind === "explainFailed" ? "explain" : "preview";
      const findings = findingsOf(status.failure);
      return (
        <Alert role="status" color="red" title={FAILURE_TITLES[operation]}>
          <Stack gap="xs" align="flex-start">
            {findings.length === 0 ? (
              <Text size="sm">{status.failure.message}</Text>
            ) : (
              <FindingList findings={findings} />
            )}
            <Button
              size="xs"
              color="red"
              variant="light"
              leftSection={<RotateCw aria-hidden="true" size={14} />}
              onClick={() =>
                void workspace.dispatch({ type: "failure.retry", operation })
              }
            >
              {RETRY_LABELS[operation]}
            </Button>
          </Stack>
        </Alert>
      );
    }
    case "blocked":
      return (
        <Alert role="status" color="red" title={blockedTitle(status.findings)}>
          <FindingList findings={status.findings} />
        </Alert>
      );
    default:
      return null;
  }
}

/** What the user has to do when every finding offers a remedy, one setting
 * to confirm per finding; otherwise that no input files can be generated
 * (#223). */
function blockedTitle(findings: readonly BlockingFinding[]): string {
  const choices = findings.filter((finding) => finding.remedies.length > 0);
  if (choices.length === 0 || choices.length < findings.length) {
    return "Cannot generate input files";
  }
  return choices.length === 1
    ? "Confirm one setting to generate input files"
    : `Confirm ${String(choices.length)} settings to generate input files`;
}

const FAILURE_TITLES = {
  explain: "Recommendation failed",
  preview: "Input generation failed",
} as const;

const RETRY_LABELS = {
  explain: "Retry recommendation",
  preview: "Retry generation",
} as const;

/** Download saves the current draft's archive and nothing else; while there
 * is none, it is disabled and describes why. */
function DownloadRow({
  status,
  download,
  previewing,
}: {
  readonly status: BundleStatus;
  readonly download: DownloadState;
  readonly previewing: boolean;
}) {
  const workspace = useWorkspace();
  const reasonId = useId();
  return (
    <Group justify="space-between">
      <Button
        rightSection={<Download aria-hidden="true" size={14} />}
        loading={previewing}
        disabled={!download.enabled}
        aria-describedby={download.enabled ? undefined : reasonId}
        onClick={() => void workspace.dispatch({ type: "review.download" })}
      >
        Download (.zip)
      </Button>
      {download.reason === null ? null : (
        <VisuallyHidden id={reasonId}>{download.reason}</VisuallyHidden>
      )}
      {status.kind === "ready" ? (
        <Text
          size="sm"
          role="status"
          aria-label="Archive status"
          aria-live="polite"
        >
          {status.archive.filename} is ready
        </Text>
      ) : null}
    </Group>
  );
}

/** What the generated-inputs section says while it has no archive. */
function placeholder(status: BundleStatus): string {
  switch (status.kind) {
    case "explainFailed":
      return "No input files were generated: the recommendation for the current settings failed.";
    case "blocked":
      return "No input files were generated for the current settings.";
    case "generationFailed":
      return "No input files were generated: generation failed for the current settings.";
    case "generating":
      return "Generating the input files for the current settings…";
    default:
      return "The generated files will appear here once the recommendation for the current settings is ready.";
  }
}
