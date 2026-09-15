import { useState } from "react";
import { Atom, ExternalLink, Info } from "lucide-react";
import {
  PROJECT_DESCRIPTION,
  PROJECT_TEAM,
  PROJECT_URL,
  FUNDING_GRANT,
  FUNDING_URL,
  ALC_URL,
} from "./project";
import {
  ActionIcon,
  Anchor,
  Drawer,
  Group,
  List,
  Paper,
  Stack,
  Text,
  Title,
} from "@mantine/core";

function OrcidIcon({ size = 14 }: { readonly size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 256 256"
      aria-label="ORCID iD"
      role="img"
      style={{ flex: "0 0 auto" }}
    >
      <path
        fill="#A6CE39"
        d="M256 128c0 70.7-57.3 128-128 128S0 198.7 0 128 57.3 0 128 0s128 57.3 128 128z"
      />
      <path
        fill="#fff"
        d="M86.3 186.2H70.9V79.1h15.4v107.1zM108.9 79.1h41.6c39.6 0 57 28.3 57 53.6 0 27.5-21.5 53.6-56.8 53.6h-41.8V79.1zm15.4 93.3h24.5c34.9 0 42.9-26.5 42.9-39.7 0-21.5-13.7-39.7-43.7-39.7h-23.7v79.4zM88.7 56.8c0 5.5-4.5 10.1-10.1 10.1-5.6 0-10.1-4.6-10.1-10.1 0-5.5 4.5-10.1 10.1-10.1 5.6 0 10.1 4.6 10.1 10.1z"
      />
    </svg>
  );
}

export function ProjectHeader() {
  const [opened, setOpened] = useState(false);
  return (
    <>
      <Paper
        component="header"
        withBorder
        radius={0}
        className="workbench-header"
        px="md"
        py={10}
      >
        <Group justify="space-between" wrap="nowrap" gap="md">
          <Group gap="sm" wrap="nowrap">
            <Atom size={20} strokeWidth={1.75} aria-hidden />
            <Text fw={700} fz="lg" lh={1}>
              goldilocks
            </Text>
            <Text c="dimmed" fz="sm" lh={1} visibleFrom="xs">
              Workbench
            </Text>
          </Group>
          <ActionIcon
            variant="default"
            aria-label="About the Goldilocks project"
            onClick={() => {
              setOpened(true);
            }}
          >
            <Info size={18} strokeWidth={1.75} aria-hidden />
          </ActionIcon>
        </Group>
      </Paper>

      <Drawer
        opened={opened}
        onClose={() => {
          setOpened(false);
        }}
        position="right"
        title="About Goldilocks"
        overlayProps={{ backgroundOpacity: 0.35 }}
        styles={{
          header: {
            borderBottom: "1px solid var(--mantine-color-default-border)",
          },
        }}
      >
        <Stack gap="lg">
          <Text>{PROJECT_DESCRIPTION}</Text>

          <Stack gap="xs">
            <Title order={4}>Project team</Title>
            <List spacing="xs" size="sm">
              {PROJECT_TEAM.map((member) => (
                <List.Item key={member.name}>
                  <Group gap="xs" wrap="nowrap">
                    <Text span>{member.name}</Text>
                    {member.role ? (
                      <Text span c="dimmed" fz="xs">
                        {member.role}
                      </Text>
                    ) : null}
                  </Group>
                  {member.orcid ? (
                    <Anchor
                      href={`https://orcid.org/${member.orcid}`}
                      target="_blank"
                      rel="noreferrer"
                      fz="xs"
                      c="dimmed"
                      aria-label={`ORCID iD of ${member.name}: ${member.orcid}`}
                    >
                      <Group gap={5} wrap="nowrap">
                        <OrcidIcon size={14} />
                        {member.orcid}
                      </Group>
                    </Anchor>
                  ) : null}
                </List.Item>
              ))}
            </List>
          </Stack>

          <Stack gap="xs">
            <Title order={4}>Funding</Title>
            <Text size="sm">
              Funded by the Science and Technology Facilities Council (STFC),
              grant{" "}
              <Anchor
                href={FUNDING_URL}
                target="_blank"
                rel="noreferrer"
                fz="sm"
              >
                {FUNDING_GRANT}
              </Anchor>
              , and the{" "}
              <Anchor href={ALC_URL} target="_blank" rel="noreferrer" fz="sm">
                Ada Lovelace Centre (ALC)
              </Anchor>
              .
            </Text>
          </Stack>

          <Anchor href={PROJECT_URL} target="_blank" rel="noreferrer" size="sm">
            <Group gap={6} wrap="nowrap">
              goldilocks.ac.uk
              <ExternalLink size={14} strokeWidth={1.75} aria-hidden />
            </Group>
          </Anchor>
        </Stack>
      </Drawer>
    </>
  );
}
