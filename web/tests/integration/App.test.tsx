import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { App } from "../../src/App";
import type {
  ArchiveDownload,
  Capabilities,
  ComputeRequest,
  CoreClient,
  ExplainResult,
  PseudopotentialTable,
  RunResult,
  StructureInput,
  StructureInspection,
} from "../../src/api/coreClient";
import { CoreFailure } from "../../src/api/coreClient";
import {
  buildArchive,
  capabilities,
  explainResult,
  inspection,
} from "../support/workbenchFixtures";
import { WorkspaceProvider } from "../../src/workspace/WorkspaceProvider";
import { createWorkspace } from "../../src/workspace/workspace";

vi.mock("../../src/viewer/StructureViewport", () => ({
  StructureViewport: () => (
    <div aria-label="Crystal structure viewer">3D crystal</div>
  ),
}));

class CoreStub implements CoreClient {
  inspectionResults: Promise<StructureInspection>[] = [];
  explainResults: Promise<ExplainResult>[] = [];
  archiveResults: Promise<ArchiveDownload>[] = [];
  inspectedInputs: StructureInput[] = [];
  explainCalls: ComputeRequest[] = [];
  archiveCalls: ComputeRequest[] = [];

  constructor(readonly capabilitiesResult: Promise<Capabilities>) {}

  capabilities(): Promise<Capabilities> {
    return this.capabilitiesResult;
  }

  inspectStructure(input: StructureInput): Promise<StructureInspection> {
    this.inspectedInputs.push(input);
    return (
      this.inspectionResults.shift() ??
      Promise.reject(new Error("inspection not configured"))
    );
  }

  explain(request: ComputeRequest): Promise<ExplainResult> {
    this.explainCalls.push(request);
    return (
      this.explainResults.shift() ??
      Promise.reject(new Error("explain not configured"))
    );
  }

  run(): Promise<RunResult> {
    return Promise.reject(new Error("run not configured"));
  }

  runArchive(request: ComputeRequest): Promise<ArchiveDownload> {
    this.archiveCalls.push(request);
    return (
      this.archiveResults.shift() ??
      Promise.reject(new Error("archive not configured"))
    );
  }
}

function renderApp(core: CoreStub, saveArchive = vi.fn()) {
  const workspace = createWorkspace(core, saveArchive);
  return render(
    <WorkspaceProvider workspace={workspace}>
      <App />
    </WorkspaceProvider>,
  );
}

async function openStructure(
  user: ReturnType<typeof userEvent.setup>,
  container: HTMLElement,
) {
  await screen.findByRole("button", {
    name: "Choose a CIF or POSCAR structure",
  });
  await user.upload(structureInputElement(container), structureFile());
  await screen.findByLabelText("Functional");
}

describe("Goldilocks Workbench", () => {
  it("filters the pseudopotential table by the chosen functional", async () => {
    const user = userEvent.setup();
    const pbesol = capabilities.pseudopotential_tables[0];
    const pbe = capabilities.pseudopotential_tables[1];
    if (pbesol === undefined || pbe === undefined) {
      throw new Error("Missing pseudopotential fixture");
    }
    const core = new CoreStub(Promise.resolve(capabilities));
    core.inspectionResults = [Promise.resolve(inspection)];
    const { container } = renderApp(core);

    await openStructure(user, container);
    const table = screen.getByLabelText("Pseudopotential table");
    expect(optionValues(table)).toEqual(["", pbesol.id, pbe.id]);

    await user.selectOptions(screen.getByLabelText("Functional"), "PBE");

    expect(optionValues(table)).toEqual(["", pbe.id]);
  });

  it("exposes a resizable two-panel structure workflow", async () => {
    renderApp(new CoreStub(Promise.resolve(capabilities)));

    expect(
      await screen.findByRole("region", { name: "Calculation setup" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("region", { name: "Structure workspace" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("region", { name: "Recommendation results" }),
    ).not.toBeInTheDocument();
    expect(screen.getAllByRole("separator")).toHaveLength(1);
  });

  it("uses light mode by default and persists an explicit dark mode", async () => {
    const user = userEvent.setup();
    const { unmount } = renderApp(new CoreStub(Promise.resolve(capabilities)));
    const toggle = await screen.findByRole("button", {
      name: "Switch to dark mode",
    });

    await user.click(toggle);

    expect(
      screen.getByRole("button", { name: "Switch to light mode" }),
    ).toBeInTheDocument();

    unmount();
    renderApp(new CoreStub(Promise.resolve(capabilities)));
    expect(
      await screen.findByRole("button", { name: "Switch to light mode" }),
    ).toBeInTheDocument();
  });

  it("announces the current Workbench operation in a persistent status", async () => {
    const core = new CoreStub(Promise.resolve(capabilities));
    let finishInspection: (value: StructureInspection) => void = () =>
      undefined;
    core.inspectionResults = [
      new Promise((resolve) => {
        finishInspection = resolve;
      }),
    ];
    const { container } = renderApp(core);
    await screen.findByRole("button", {
      name: "Choose a CIF or POSCAR structure",
    });

    expect(
      screen.getByRole("status", { name: "Workbench status" }),
    ).toHaveTextContent("Ready");
    await userEvent.upload(structureInputElement(container), structureFile());
    expect(
      screen.getByRole("status", { name: "Workbench status" }),
    ).toHaveTextContent("Inspecting structure");

    finishInspection(inspection);
  });

  it("preserves the inspected viewport when switching theme", async () => {
    const user = userEvent.setup();
    const core = new CoreStub(Promise.resolve(capabilities));
    core.inspectionResults = [Promise.resolve(inspection)];
    const { container } = renderApp(core);
    await openStructure(user, container);
    const viewport = await screen.findByLabelText("Crystal structure viewer");
    await user.click(
      screen.getByRole("button", { name: "Switch to dark mode" }),
    );
    expect(screen.getByLabelText("Crystal structure viewer")).toBe(viewport);
    await user.click(
      screen.getByRole("button", { name: "Switch to light mode" }),
    );
    expect(screen.getByLabelText("Crystal structure viewer")).toBe(viewport);
  });

  it("resizes the calculation panel from the keyboard", async () => {
    const user = userEvent.setup();
    renderApp(new CoreStub(Promise.resolve(capabilities)));
    const controls = await screen.findByRole("separator", {
      name: "Resize calculation setup",
    });
    const initial = controls.getAttribute("aria-valuenow");
    controls.focus();
    await user.keyboard("{ArrowRight}");
    expect(controls.getAttribute("aria-valuenow")).not.toBe(initial);
    await user.keyboard("{ArrowLeft}");
    expect(controls).toHaveAttribute("aria-valuenow", initial);
  });

  it("opens only the latest file when an earlier read resolves last", async () => {
    let finishFirstRead: (content: string) => void = () => undefined;
    const firstRead = new Promise<string>((resolve) => {
      finishFirstRead = resolve;
    });
    const first = new File(["A"], "A.cif");
    Object.defineProperty(first, "text", { value: () => firstRead });
    const second = new File(["B"], "B.cif");
    Object.defineProperty(second, "text", {
      value: () => Promise.resolve("structure B"),
    });
    const core = new CoreStub(Promise.resolve(capabilities));
    core.inspectionResults = [Promise.resolve(inspection)];
    const { container } = renderApp(core);
    await screen.findByRole("button", {
      name: "Choose a CIF or POSCAR structure",
    });

    await userEvent.upload(structureInputElement(container), first);
    await userEvent.upload(structureInputElement(container), second);
    await waitFor(() => {
      expect(core.inspectedInputs).toEqual([
        {
          structure_content: "structure B",
          structure_name: "B.cif",
          structure_format: "cif",
        },
      ]);
    });
    finishFirstRead("structure A");
    await firstRead;
    await Promise.resolve();

    expect(core.inspectedInputs).toHaveLength(1);
  });

  it("recomputes edits before generating input files, then downloads the exact archive", async () => {
    const user = userEvent.setup();
    const archive = buildArchive();
    const core = new CoreStub(Promise.resolve(capabilities));
    core.inspectionResults = [Promise.resolve(inspection)];
    core.explainResults = [
      Promise.resolve(explainResult),
      Promise.resolve(explainResult),
    ];
    core.archiveResults = [Promise.resolve(archive)];
    const saveArchive = vi.fn<(download: ArchiveDownload) => void>();
    const { container } = renderApp(core, saveArchive);

    await openStructure(user, container);
    await user.click(
      await screen.findByRole("button", { name: "Generate recommendation" }),
    );
    await screen.findByText("K Sampling");
    await user.click(within(calculationSetup()).getByText("Magnetic"));
    await user.selectOptions(screen.getByLabelText("spin polarized"), "true");

    await user.click(
      screen.getByRole("button", { name: "Update recommendation" }),
    );
    await waitFor(() => {
      expect(
        screen.queryByRole("status", { name: "Recommendation notice" }),
      ).not.toBeInTheDocument();
    });
    await user.click(
      screen.getByRole("button", { name: "Generate input files (.zip)" }),
    );

    await waitFor(() => {
      expect(saveArchive).toHaveBeenCalledWith(archive);
    });
    expect(
      screen.getByRole("status", { name: "Archive status" }),
    ).toHaveTextContent(`${archive.filename} is ready`);
    expect(core.explainCalls[1]?.overrides).toMatchObject({
      spin_polarized: true,
    });
    expect(core.archiveCalls[0]?.overrides).toMatchObject({
      spin_polarized: true,
    });
  });

  it("submits smearing treatment and width as overrides in the occupations group", async () => {
    const user = userEvent.setup();
    const core = new CoreStub(Promise.resolve(capabilities));
    core.inspectionResults = [Promise.resolve(inspection)];
    core.explainResults = [Promise.resolve(explainResult)];
    const { container } = renderApp(core);

    await openStructure(user, container);
    await user.click(within(calculationSetup()).getByText("Occupations"));
    await user.selectOptions(screen.getByLabelText("smearing type"), "cold");
    const degauss = screen.getByLabelText("degauss · Ry");
    expect(degauss).toBeEnabled();
    fireEvent.change(degauss, { target: { value: "0.02" } });

    await user.click(
      screen.getByRole("button", { name: "Generate recommendation" }),
    );

    expect(core.explainCalls[0]?.overrides).toMatchObject({
      smearing_type: "cold",
      degauss: 0.02,
    });
  });

  it("keeps the old review visible and disables input generation after an edit", async () => {
    const user = userEvent.setup();
    const core = new CoreStub(Promise.resolve(capabilities));
    core.inspectionResults = [Promise.resolve(inspection)];
    core.explainResults = [Promise.resolve(explainResult)];
    const { container } = renderApp(core, vi.fn());

    await openStructure(user, container);
    await user.click(
      await screen.findByRole("button", { name: "Generate recommendation" }),
    );
    await screen.findByText("K Sampling");

    await user.click(within(calculationSetup()).getByText("Magnetic"));
    await user.selectOptions(screen.getByLabelText("spin polarized"), "true");

    expect(
      screen.getByRole("status", { name: "Recommendation notice" }),
    ).toHaveTextContent(
      "Your settings changed. Update the recommendation before generating input files.",
    );
    expect(screen.getByText("K Sampling")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Generate input files (.zip)" }),
    ).toBeDisabled();
  });

  it("computes a recommendation and renders tri-state scientific records", async () => {
    const user = userEvent.setup();
    const core = new CoreStub(Promise.resolve(capabilities));
    core.inspectionResults = [Promise.resolve(inspection)];
    core.explainResults = [Promise.resolve(explainResult)];
    const { container } = renderApp(core);

    await openStructure(user, container);
    await user.click(
      await screen.findByRole("button", { name: "Generate recommendation" }),
    );

    const recommendation = await screen.findByRole("region", {
      name: "Recommendation results",
    });
    expect(recommendation).toBeInTheDocument();
    expect(
      screen.queryByRole("region", { name: "Structure workspace" }),
    ).not.toBeInTheDocument();
    expect(core.explainCalls[0]).toMatchObject({
      structure_content: "data_Si",
      structure_name: "Si.cif",
      task: "scf_single_point",
    });
    expect(screen.getByText("K Sampling")).toBeInTheDocument();
    expect(screen.getByText("Cutoffs")).toBeInTheDocument();
    expect(
      screen.getByText(/Generate input files to preview them here/),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Back to structure" }));
    expect(
      screen.getByRole("region", { name: "Structure workspace" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("region", { name: "Recommendation results" }),
    ).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Recommendation" }));
    expect(
      screen.getByRole("region", { name: "Recommendation results" }),
    ).toBeInTheDocument();
  });

  it("announces structured scientific warnings returned with a recommendation", async () => {
    const user = userEvent.setup();
    const core = new CoreStub(Promise.resolve(capabilities));
    core.inspectionResults = [Promise.resolve(inspection)];
    core.explainResults = [
      Promise.resolve({
        ...explainResult,
        warnings: [
          {
            code: "occupations.smearing_defaulted",
            level: "warning",
            category: "occupations",
            message: "Review smearing before production use.",
          },
        ],
      }),
    ];
    const { container } = renderApp(core);

    await openStructure(user, container);
    await user.click(
      await screen.findByRole("button", { name: "Generate recommendation" }),
    );

    expect(
      await screen.findByRole("status", { name: "Warnings" }),
    ).toHaveTextContent("Review smearing before production use.");
  });

  it("opens a CIF and builds calculation controls from Capabilities", async () => {
    const user = userEvent.setup();
    const core = new CoreStub(Promise.resolve(capabilities));
    core.inspectionResults = [Promise.resolve(inspection)];
    const { container } = renderApp(core);

    await openStructure(user, container);

    expect(await screen.findByText("Si1")).toBeInTheDocument();
    expect(
      screen.getByLabelText("Crystal structure viewer"),
    ).toBeInTheDocument();
    expect(core.inspectedInputs).toEqual([
      {
        structure_content: "data_Si",
        structure_name: "Si.cif",
        structure_format: "cif",
      },
    ]);
    const tableSelect = screen.getByLabelText("Pseudopotential table");
    expect(optionValues(tableSelect)).toEqual([
      "",
      ...capabilities.pseudopotential_tables.map(
        (table: PseudopotentialTable) => table.id,
      ),
    ]);
    // Relax-only settings stay hidden for the default scf_single_point task.
    expect(screen.queryByText("Relax")).not.toBeInTheDocument();
  });

  it("reveals relax-only settings once the relax task is selected", async () => {
    const user = userEvent.setup();
    const core = new CoreStub(Promise.resolve(capabilities));
    core.inspectionResults = [Promise.resolve(inspection)];
    const { container } = renderApp(core);

    await openStructure(user, container);
    await user.selectOptions(screen.getByLabelText("Task"), "relax");

    expect(screen.getByText("Relax")).toBeInTheDocument();
  });

  it("keeps retry available and dismiss hidden for a Capabilities failure", async () => {
    const failure = new CoreFailure(
      "assets_unavailable",
      "Runtime assets are unavailable.",
      false,
    );
    renderApp(new CoreStub(Promise.reject(failure)));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Runtime assets unavailable");
    expect(alert).toHaveTextContent("Runtime assets are unavailable.");
    const status = screen.getByRole("status", { name: "Workbench status" });
    expect(status).toHaveTextContent("Needs attention");
    expect(status).not.toHaveTextContent("Ready");
    expect(screen.getByRole("button", { name: "Retry" })).toBeEnabled();
    expect(
      screen.queryByRole("button", { name: "Dismiss error" }),
    ).not.toBeInTheDocument();
  });

  it("announces Capabilities loading at startup", async () => {
    const pending = new Promise<Capabilities>(() => undefined);
    renderApp(new CoreStub(pending));

    expect(
      await screen.findByRole("heading", { name: "Loading Workbench" }),
    ).toBeInTheDocument();
  });
});

function structureFile(): File {
  const file = new File(["data_Si"], "Si.cif", {
    type: "chemical/x-cif",
  });
  Object.defineProperty(file, "text", {
    value: () => Promise.resolve("data_Si"),
  });
  return file;
}

function structureInputElement(container: HTMLElement): HTMLInputElement {
  const input = container.querySelector<HTMLInputElement>('input[type="file"]');
  if (input === null) throw new Error("structure input missing");
  return input;
}

function optionValues(select: HTMLElement): string[] {
  return Array.from(
    select.querySelectorAll<HTMLOptionElement>("option"),
    (option) => option.value,
  );
}

/** A settings group's humanized name (e.g. "Magnetic") can collide with
 * a scientific record's own humanized name once results render -- both
 * are independently derived from real, unrelated vocabularies (an
 * override group vs. an advisor's record key). Scoping to this region
 * disambiguates in tests the same way a sighted user would from layout. */
function calculationSetup(): HTMLElement {
  return screen.getByRole("region", { name: "Calculation setup" });
}
