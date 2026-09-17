import { ActionIcon, Badge, Group, Text } from "@mantine/core";
import { Moon, Sun } from "lucide-react";

import type { Theme } from "../theme";

export function AppHeader({
  theme,
  onToggleTheme,
}: {
  readonly theme: Theme;
  readonly onToggleTheme: () => void;
}) {
  return (
    <Group
      component="header"
      justify="space-between"
      wrap="nowrap"
      px="md"
      className="app-header"
    >
      <Group gap="xs" wrap="nowrap">
        <img src="/brand/goldilocks-logo.png" alt="" width={24} height={24} />
        <Text fw={600} size="sm">
          Goldilocks
        </Text>
        <Badge variant="light" size="sm" radius="sm">
          Workbench
        </Badge>
      </Group>
      <ActionIcon
        variant="default"
        aria-label={
          theme === "light" ? "Switch to dark mode" : "Switch to light mode"
        }
        title={theme === "light" ? "Dark mode" : "Light mode"}
        onClick={onToggleTheme}
      >
        {theme === "light" ? (
          <Sun aria-hidden="true" size={17} />
        ) : (
          <Moon aria-hidden="true" size={17} />
        )}
      </ActionIcon>
    </Group>
  );
}
