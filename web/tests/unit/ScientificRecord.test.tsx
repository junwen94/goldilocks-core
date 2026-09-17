import { MantineProvider } from "@mantine/core";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { ResolvedField } from "../../src/api/coreClient";
import { buildArchive } from "../support/workbenchFixtures";
import { GeneratedInputReview } from "../../src/review/GeneratedInputReview";
import { ScientificRecord } from "../../src/review/ScientificRecord";

describe("ScientificRecord", () => {
  it("renders an unavailable field's reason", () => {
    const field: ResolvedField = {
      status: "unavailable",
      reason: "No pseudopotential table matches this structure.",
    };
    render(<ScientificRecord field={field} />, { wrapper: MantineProvider });

    expect(
      screen.getByText(/No pseudopotential table matches this structure\./),
    ).toBeInTheDocument();
  });

  it("renders a blocked field's upstream cause", () => {
    const field: ResolvedField = {
      status: "blocked",
      blocked_by: "pseudo_table",
    };
    render(<ScientificRecord field={field} />, { wrapper: MantineProvider });

    expect(screen.getByText(/pseudo_table/)).toBeInTheDocument();
  });

  it("renders a resolved field's source, nested value, and per-field sources", () => {
    const field: ResolvedField = {
      status: "resolved",
      value: { grid: [4, 6, 8], shift: [1, 0, 1], mesh_type: "monkhorst-pack" },
      source: "human",
      field_sources: { grid: "human", shift: "heuristic" },
    };
    render(<ScientificRecord field={field} />, { wrapper: MantineProvider });

    expect(screen.getAllByText("Your override").length).toBeGreaterThan(0);
    expect(screen.getByText("4, 6, 8")).toBeInTheDocument();
    expect(screen.getByText("monkhorst-pack")).toBeInTheDocument();
    expect(screen.getByText("Per-field sources")).toBeInTheDocument();
    expect(screen.getByText("Heuristic default")).toBeInTheDocument();
  });

  it("renders a record's own nested warnings inline", () => {
    const field: ResolvedField = {
      status: "resolved",
      value: {
        grid: [4, 6, 8],
        warnings: [
          {
            code: "kmesh.dense",
            level: "warning",
            category: "kmesh",
            message: "Check convergence for this mesh.",
          },
        ],
      },
      source: "heuristic",
    };
    render(<ScientificRecord field={field} />, { wrapper: MantineProvider });

    expect(
      screen.getByText("Check convergence for this mesh."),
    ).toBeInTheDocument();
  });
});

describe("GeneratedInputReview", () => {
  it("shows a placeholder before any archive has been generated", () => {
    render(<GeneratedInputReview archive={null} />, {
      wrapper: MantineProvider,
    });

    expect(
      screen.getByText(/will appear here automatically/),
    ).toBeInTheDocument();
  });

  it("unzips a real archive and previews each file with its manifest digest", async () => {
    const user = userEvent.setup();
    const archive = buildArchive();
    render(<GeneratedInputReview archive={archive} />, {
      wrapper: MantineProvider,
    });

    // Files are ordered by their manifest role -- the rendered QE input
    // first, then the pseudopotential -- each collapsed by default
    // behind an accordion row rather than a tab.
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /scf\.in/ }),
      ).toBeInTheDocument();
    });
    await user.click(screen.getByRole("button", { name: /scf\.in/ }));

    expect(
      screen.getByRole("region", { name: "Generated input scf.in" }),
    ).toHaveTextContent("&CONTROL");
    expect(screen.getByText("cccccccccc")).toBeInTheDocument();
  });

  it("shows a pseudopotential's entries instead of its raw file content", async () => {
    const user = userEvent.setup();
    const archive = buildArchive();
    render(<GeneratedInputReview archive={archive} />, {
      wrapper: MantineProvider,
    });

    await user.click(await screen.findByRole("button", { name: /Si\.upf/ }));

    expect(screen.getByText("SHA-256")).toBeInTheDocument();
    expect(screen.queryByText("UPF content")).not.toBeInTheDocument();
  });
});
