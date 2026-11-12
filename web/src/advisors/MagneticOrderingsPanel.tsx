import { useEffect } from "react";
import {
  Badge,
  Button,
  Checkbox,
  Group,
  Stack,
  Table,
  Text,
  VisuallyHidden,
} from "@mantine/core";
import { RotateCw } from "lucide-react";

import { WarningsPanel } from "../review/WarningsPanel";
import { useWorkspace, useWorkspaceSnapshot } from "../workspace/useWorkspace";
import { currentOrderings } from "../workspace/workspace";
import type {
  MagneticOrderingsListing,
  RankingProgress,
} from "../workspace/workspace";

/** What the card shows for a structure that has no listing yet. */
const UNLISTED = { result: null, operation: null, error: null } as const;

/** #87 layer 4: lists candidate magnetic orderings (FM plus any AFM
 * candidates enumerate_magnetic_orderings finds) for the currently
 * inspected structure, with an opt-in "Rank with mMACE" pass -- a server
 * job whose place in the queue and progress the card shows (#261) -- and
 * a checkbox per candidate. The ticked candidates are the orderings Bundle
 * offers in its menu and downloads (#293, `bundleOrderings`) -- see
 * workspace.ts's `magneticOrderings.*` actions for the request/response
 * plumbing.
 * Rendered inside `MagneticOrderingsCard`, which owns the heading --
 * this component starts directly with its content. */
export function MagneticOrderingsPanel() {
  const workspace = useWorkspace();
  const snapshot = useWorkspaceSnapshot();
  const { inspection, selectedOrderings } = snapshot;
  const listing = currentOrderings(snapshot);
  const { result: magneticOrderings, operation, error } = listing ?? UNLISTED;

  // Lists the open structure while it has no listing of its own. A failed
  // listing is its listing too, so its error stays and only Retry listing
  // sends it again.
  useEffect(() => {
    if (inspection !== null && listing === null) {
      void workspace.dispatch({ type: "magneticOrderings.list" });
    }
  }, [inspection, listing, workspace]);

  if (inspection === null) return null;

  const candidates = magneticOrderings?.candidates ?? [];
  const busy = operation !== null;

  function toggleSelected(label: string, checked: boolean): void {
    void workspace.dispatch({
      type: "magneticOrderings.select",
      labels: checked
        ? [...selectedOrderings, label]
        : selectedOrderings.filter((entry) => entry !== label),
    });
  }

  return (
    <Stack gap="sm">
      <Stack gap={4}>
        <Group justify="flex-start">
          <Button
            size="xs"
            variant="light"
            miw={160}
            loading={operation === "rank"}
            disabled={busy || candidates.length === 0}
            onClick={() =>
              void workspace.dispatch({ type: "magneticOrderings.rank" })
            }
          >
            Run mMACE
          </Button>
        </Group>
        <Text size="xs" c="dimmed">
          Relaxes each candidate ordering with mMACE (a magnetic MLIP model) and
          ranks them by energy per atom.
        </Text>
        <RankingStatus listing={listing} />
      </Stack>
      {error === null ? null : (
        <OrderingsError error={error} listed={magneticOrderings !== null} />
      )}
      <WarningsPanel warnings={magneticOrderings?.warnings ?? []} />
      {candidates.length === 0 ? (
        <Text c="dimmed" size="sm">
          {operation === "list" ? "Listing candidates…" : "No candidates yet."}
        </Text>
      ) : (
        <>
          {/* `layout="fixed"` used to force these 5 columns into
           * whatever width the card happened to have, however narrow --
           * "Ordering"/"Atoms" ran into each other and "afm-1" wrapped
           * mid-word. Natural column widths plus a scroll container let
           * every column keep its own readable width and the table
           * scroll horizontally instead, on any card width. */}
          <Table.ScrollContainer minWidth={420}>
            <Table verticalSpacing={4}>
              <Table.Caption>
                {magneticOrderings?.ranked
                  ? "Ranked by mMACE-relaxed energy per atom"
                  : "Not yet ranked"}
              </Table.Caption>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th style={{ width: "2.5rem" }}>
                    <VisuallyHidden>Select for Bundle</VisuallyHidden>
                  </Table.Th>
                  <Table.Th>Ordering</Table.Th>
                  <Table.Th>Atoms</Table.Th>
                  <Table.Th>E/atom (eV)</Table.Th>
                  <Table.Th>Status</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {candidates.map((candidate) => (
                  <Table.Tr key={candidate.label}>
                    <Table.Td>
                      <Checkbox
                        aria-label={`Select ${candidate.label} for Bundle`}
                        checked={selectedOrderings.includes(candidate.label)}
                        onChange={(event) => {
                          toggleSelected(
                            candidate.label,
                            event.currentTarget.checked,
                          );
                        }}
                      />
                    </Table.Td>
                    <Table.Td>
                      <Group gap={4} wrap="nowrap">
                        <Text size="sm">{candidate.label}</Text>
                        {candidate.is_recommended ? (
                          <Badge size="xs" color="teal">
                            Recommended
                          </Badge>
                        ) : null}
                      </Group>
                    </Table.Td>
                    <Table.Td>{candidate.natoms}</Table.Td>
                    <Table.Td>
                      {candidate.energy_per_atom_ev === null
                        ? "—"
                        : candidate.energy_per_atom_ev.toFixed(4)}
                    </Table.Td>
                    <Table.Td>{candidate.status ?? "—"}</Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>
        </>
      )}
    </Stack>
  );
}

/** Where the ranking in flight stands (#261): how many rankings are ahead
 * of it on the server, then which candidate it is relaxing. Nothing while
 * no ranking is in flight. */
function RankingStatus({
  listing,
}: {
  readonly listing: MagneticOrderingsListing | null;
}) {
  if (listing?.operation !== "rank") return null;
  return (
    <Text size="sm" aria-live="polite">
      {rankingText(listing.ranking)}
    </Text>
  );
}

function rankingText(ranking: RankingProgress | null): string {
  if (ranking === null) return "Submitting the ranking…";
  if (ranking.state === "queued") {
    const ahead = ranking.position;
    if (ahead === 0) return "Queued: yours runs next";
    return `Queued: ${String(ahead)} ranking${ahead === 1 ? "" : "s"} ahead of yours`;
  }
  const { progress } = ranking;
  if (progress === null) return "Ranking: listing the candidates…";
  if (progress.done < progress.total) {
    return `Ranking candidate ${String(progress.done + 1)} of ${String(progress.total)}`;
  }
  return "Ranking: comparing the energies…";
}

/** A failed listing, ranking or download of the open structure. A failed
 * listing stays the structure's listing and is not re-sent on its own
 * (#150), so it offers Retry listing; after a failed ranking or download the
 * listing is still there, and its own buttons send that request again. */
function OrderingsError({
  error,
  listed,
}: {
  readonly error: string;
  readonly listed: boolean;
}) {
  const workspace = useWorkspace();
  return (
    <Stack gap={4} align="flex-start">
      <Text c="red" size="sm" role="alert">
        {error}
      </Text>
      {listed ? null : (
        <Button
          size="xs"
          variant="light"
          leftSection={<RotateCw aria-hidden="true" size={14} />}
          onClick={() =>
            void workspace.dispatch({ type: "magneticOrderings.list" })
          }
        >
          Retry listing
        </Button>
      )}
    </Stack>
  );
}
