import { useId } from "react";
import {
  Alert,
  Button,
  Group,
  Loader,
  NativeSelect,
  Paper,
  Stack,
  Text,
  Title,
  VisuallyHidden,
} from "@mantine/core";
import { Download, RotateCw } from "lucide-react";

import {
  type BlockingFinding,
  type MagneticOrderingCandidate,
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
import {
  bundleOrderings,
  currentOrderings,
  shownOrdering,
} from "../workspace/workspace";

/** The last card. While Magnetic Orderings is shown (`orderings`), its menu
 * picks which of `bundleOrderings` the card previews and its Download
 * saves them all (#293); otherwise Download saves the previewed archive. */
export function BundleCard({
  kicker,
  orderings,
}: {
  readonly kicker: string;
  readonly orderings: boolean;
}) {
  const snapshot = useWorkspaceSnapshot();
  const status = bundleStatus(snapshot);
  const listing = currentOrderings(snapshot);
  const labels = orderings ? bundleOrderings(snapshot) : [];

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
            {labels.length === 0 || snapshot.draft === null ? null : (
              <OrderingSelect
                labels={labels}
                candidates={listing?.result?.candidates ?? []}
                overrides={snapshot.draft.overrides}
                shown={shownOrdering(snapshot.draft)}
                disabled={snapshot.operation === "inspect"}
              />
            )}
            <BundleNotice status={status} />
            <DownloadRow
              status={status}
              download={downloadState(snapshot)}
              previewing={snapshot.operation === "preview"}
              orderingCount={labels.length}
              downloadingOrderings={listing?.operation === "download"}
            />
            {labels.length === 0 || !listing?.downloadError ? null : (
              <Text c="red" size="sm" role="alert">
                {listing.downloadError}
              </Text>
            )}
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

/** The menu of the orderings Bundle offers (`labels`, #293): the one
 * chosen is the one Bundle previews, made the draft's ordering by
 * `magneticOrderings.show`, which also sets `starting_magnetization` back
 * to automatic, as its description says while it is set. An ordering set
 * in Advisors that is not a listed candidate, such as `afm`, is listed as
 * it is. */
function OrderingSelect({
  labels,
  candidates,
  overrides,
  shown,
  disabled,
}: {
  readonly labels: readonly string[];
  readonly candidates: readonly MagneticOrderingCandidate[];
  readonly overrides: Readonly<Record<string, unknown>>;
  readonly shown: string;
  readonly disabled: boolean;
}) {
  const workspace = useWorkspace();
  const data = labels.map((label) => {
    const candidate = candidates.find((entry) => entry.label === label);
    return {
      value: label,
      label:
        candidate === undefined
          ? `${label} (set in Advisors)`
          : orderingOption(candidate),
    };
  });
  return (
    <NativeSelect
      label="Magnetic ordering"
      description={
        overrides.starting_magnetization === undefined
          ? undefined
          : "Choosing another ordering sets starting magnetization back to automatic."
      }
      disabled={disabled}
      value={shown}
      data={data}
      onChange={(event) => {
        void workspace.dispatch({
          type: "magneticOrderings.show",
          label: event.currentTarget.value,
        });
      }}
    />
  );
}

/** A candidate as the ordering menu lists it: its label and atom count, fm
 * as the uploaded cell and the default, and the one mMACE ranked lowest
 * marked recommended. */
function orderingOption(candidate: MagneticOrderingCandidate): string {
  const atoms = `${String(candidate.natoms)} atom${candidate.natoms === 1 ? "" : "s"}`;
  const option =
    candidate.label === "fm"
      ? `fm — uploaded cell (${atoms}), default`
      : `${candidate.label} (${atoms})`;
  return candidate.is_recommended ? `${option} · recommended` : option;
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

/** Download saves the current draft's archive; with more than one
 * ordering in Bundle's menu (`orderingCount`), the inputs of all of them,
 * one archive with a folder per ordering (#293). While the current draft
 * has no archive, it is disabled and describes why. */
function DownloadRow({
  status,
  download,
  previewing,
  orderingCount,
  downloadingOrderings,
}: {
  readonly status: BundleStatus;
  readonly download: DownloadState;
  readonly previewing: boolean;
  readonly orderingCount: number;
  readonly downloadingOrderings: boolean;
}) {
  const workspace = useWorkspace();
  const reasonId = useId();
  const several = orderingCount > 1;
  return (
    <Group justify="space-between">
      <Button
        rightSection={<Download aria-hidden="true" size={14} />}
        loading={previewing || downloadingOrderings}
        disabled={!download.enabled}
        aria-describedby={download.enabled ? undefined : reasonId}
        onClick={() =>
          void workspace.dispatch({
            type: several ? "magneticOrderings.download" : "review.download",
          })
        }
      >
        {several
          ? `Download ${String(orderingCount)} orderings (.zip)`
          : "Download (.zip)"}
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
