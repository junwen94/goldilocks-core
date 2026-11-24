import { MantineProvider } from "@mantine/core";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type {
  StructureInput,
  StructureInspection,
} from "../../src/api/coreClient";
import {
  type StructureChoices,
  StructureSourceControls,
} from "../../src/controls/StructureSourceControls";
import { inspection } from "../support/workbenchFixtures";

const chatStructures: StructureChoices = {
  label: "From this chat",
  items: [
    { name: "Fe.cif", content: "data_Fe" },
    { name: "NiO", content: "NiO\n1.0\n", format: "poscar" },
    { name: "Si.vasp", content: "Si\n1.0\n" },
  ],
};

function renderControls({
  source = null,
  choices,
  onOpen = vi.fn(() => Promise.resolve()),
}: {
  readonly source?: StructureInput | null;
  readonly choices?: StructureChoices;
  readonly onOpen?: (input: StructureInput) => Promise<void>;
}) {
  render(
    <MantineProvider env="test">
      <StructureSourceControls
        source={source}
        inspection={null}
        inspecting={false}
        onOpen={onOpen}
        choices={choices}
      />
    </MantineProvider>,
  );
  return onOpen;
}

/** A graphene monolayer as /inspect returns it: pymatgen marks every CIF
 * fully periodic, so `periodicity` is all true even though the cell is a
 * 2D sheet in vacuum. */
const graphene: StructureInspection = {
  ...inspection,
  source: { ...inspection.source, name: "graphene.cif" },
  structure: {
    ...inspection.structure,
    formula: "C2",
    reduced_formula: "C",
    site_count: 2,
    periodicity: [true, true, true],
    lattice: {
      vectors_angstrom: [
        [2.467, 0, 0],
        [-1.2335, 2.1365, 0],
        [0, 0, 19.97],
      ],
      lengths_angstrom: [2.467, 2.467, 19.97],
      angles_degrees: [90, 90, 120],
      volume_angstrom3: 105.26,
    },
    sites: [
      {
        fractional_coordinates: [1 / 3, 2 / 3, 0.5],
        cartesian_coordinates_angstrom: [0, 1.4243, 9.985],
        label: "C1",
        species: [
          { symbol: "C", label: "C1", occupancy: 1, oxidation_state: null },
        ],
      },
      {
        fractional_coordinates: [2 / 3, 1 / 3, 0.5],
        cartesian_coordinates_angstrom: [1.2335, 0.7122, 9.985],
        label: "C2",
        species: [
          { symbol: "C", label: "C2", occupancy: 1, oxidation_state: null },
        ],
      },
    ],
  },
};

describe("StructureSourceControls", () => {
  it("summarizes a monolayer without a dimensionality claim", () => {
    render(
      <MantineProvider env="test">
        <StructureSourceControls
          source={{
            structure_content: "data_graphene",
            structure_name: "graphene.cif",
            structure_format: "cif",
          }}
          inspection={graphene}
          inspecting={false}
          onOpen={vi.fn()}
        />
      </MantineProvider>,
    );

    const summary = screen.getByLabelText("Inspected structure summary");
    expect(summary.textContent).toBe("C2 · C · 105.3 Å³");
    expect(summary.textContent).not.toMatch(/\b[23]D\b|Partial/);
  });

  it("lists the host's structures under its label and opens the chosen one as an opened file", async () => {
    const user = userEvent.setup();
    const onOpen = renderControls({ choices: chatStructures });

    const select = screen.getByRole("combobox", { name: "From this chat" });
    expect(
      Array.from(select.querySelectorAll("option"), (option) => option.text),
    ).toEqual(["Choose a structure", "Fe.cif", "NiO", "Si.vasp"]);
    expect(screen.getByRole("button", { name: "Browse files" })).toBeVisible();

    await user.selectOptions(select, "Fe.cif");
    await user.selectOptions(select, "NiO");
    await user.selectOptions(select, "Si.vasp");

    expect(onOpen).toHaveBeenNthCalledWith(1, {
      structure_content: "data_Fe",
      structure_name: "Fe.cif",
      structure_format: "cif",
    });
    expect(onOpen).toHaveBeenNthCalledWith(2, {
      structure_content: "NiO\n1.0\n",
      structure_name: "NiO",
      structure_format: "poscar",
    });
    expect(onOpen).toHaveBeenNthCalledWith(3, {
      structure_content: "Si\n1.0\n",
      structure_name: "Si.vasp",
      structure_format: "poscar",
    });
  });

  it("shows the open structure as selected when it is one of the host's, and nothing otherwise", () => {
    renderControls({
      choices: chatStructures,
      source: {
        structure_content: "NiO\n1.0\n",
        structure_name: "NiO",
        structure_format: "poscar",
      },
    });
    expect(
      screen.getByRole<HTMLSelectElement>("combobox", {
        name: "From this chat",
      }).selectedOptions[0]?.text,
    ).toBe("NiO");
  });

  it("leaves the host select empty for a structure opened from elsewhere, even under a host name", () => {
    renderControls({
      choices: chatStructures,
      source: {
        structure_content: "data_Fe_edited",
        structure_name: "Fe.cif",
        structure_format: "cif",
      },
    });
    expect(
      screen.getByRole<HTMLSelectElement>("combobox", {
        name: "From this chat",
      }).value,
    ).toBe("");
    expect(screen.getByText("Fe.cif", { selector: "p" })).toBeVisible();
  });

  it("refuses an empty or oversized host structure as it refuses such a file", async () => {
    const user = userEvent.setup();
    const onOpen = renderControls({
      choices: {
        label: "From this chat",
        items: [
          { name: "empty.cif", content: "" },
          { name: "huge.cif", content: "a".repeat(5 * 1024 * 1024 + 1) },
        ],
      },
    });
    const select = screen.getByRole("combobox", { name: "From this chat" });

    await user.selectOptions(select, "empty.cif");
    expect(screen.getByRole("alert").textContent).toBe(
      "The selected structure file is empty.",
    );
    await user.selectOptions(select, "huge.cif");
    expect(screen.getByRole("alert").textContent).toBe(
      "Structure files must be 5 MB or smaller.",
    );
    expect(onOpen).not.toHaveBeenCalled();
  });

  it("offers no host select without choices or with an empty list", () => {
    renderControls({ choices: { label: "From this chat", items: [] } });
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Browse files" })).toBeVisible();
  });
});
