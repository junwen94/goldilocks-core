import { describe, expect, it, vi } from "vitest";

import type {
  ArchiveDownload,
  Capabilities,
  ComputeRequest,
  CoreClient,
  ExplainResult,
  MagneticOrderingsRequest,
  MagneticOrderingsResult,
  RunResult,
  StructureInput,
  StructureInspection,
} from "../../src/api/coreClient";
import { CoreFailure } from "../../src/api/coreClient";
import {
  buildArchive,
  capabilities,
  draft,
  explainResult,
  inspection,
  structureInput,
} from "../support/workbenchFixtures";
import {
  createWorkspace,
  currentOrderings,
} from "../../src/workspace/workspace";

class CoreStub implements CoreClient {
  capabilitiesResult: Promise<Capabilities> = Promise.resolve(capabilities);
  capabilitiesCalls = 0;
  inspectionResults: Promise<StructureInspection>[] = [
    Promise.resolve(inspection),
  ];
  inspectedInputs: StructureInput[] = [];
  explainResults: Promise<ExplainResult>[] = [];
  explainCalls: ComputeRequest[] = [];
  archiveResults: Promise<ArchiveDownload>[] = [];
  archiveCalls: ComputeRequest[] = [];
  magneticOrderingsResults: Promise<MagneticOrderingsResult>[] = [];
  magneticOrderingsCalls: MagneticOrderingsRequest[] = [];
  magneticOrderingsSignals: (AbortSignal | undefined)[] = [];

  capabilities(): Promise<Capabilities> {
    this.capabilitiesCalls += 1;
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

  /** Records the signal and, like a client that ignores it, still settles
   * the queued result; tests reject it with `signal.reason` where they
   * stand in for fetch, which rejects an aborted request that way. */
  magneticOrderings(
    request: MagneticOrderingsRequest,
    signal?: AbortSignal,
  ): Promise<MagneticOrderingsResult> {
    this.magneticOrderingsCalls.push(request);
    this.magneticOrderingsSignals.push(signal);
    return (
      this.magneticOrderingsResults.shift() ??
      Promise.reject(new Error("magnetic orderings not configured"))
    );
  }
}

const replacementInput: StructureInput = {
  structure_content: "Silicon B",
  structure_name: "POSCAR",
  structure_format: "poscar",
};

/** A listing of `labels`, each a cell of `natoms` atoms. */
function orderingsListing(
  natoms: number,
  ...labels: readonly string[]
): MagneticOrderingsResult {
  return {
    ranked: false,
    candidates: labels.map((label) => ({
      label,
      formula: "Si",
      natoms,
      energy_per_atom_ev: null,
      status: null,
      is_recommended: false,
      overrides: { magnetic_ordering: label },
    })),
    warnings: [],
  };
}

function baseRequest(): ComputeRequest {
  return {
    structure_content: structureInput.structure_content,
    structure_name: structureInput.structure_name,
    structure_format: "cif",
    code: draft.code,
    task: draft.task,
    hpc: draft.hpc,
    overrides: draft.overrides,
    fetch_missing: draft.fetchMissing,
  };
}

describe("Workspace", () => {
  it("opens an inline structure input and initializes a canonical draft", async () => {
    const core = new CoreStub();
    const workspace = createWorkspace(core);
    await workspace.dispatch({ type: "workspace.start" });

    await workspace.dispatch({ type: "source.open", input: structureInput });

    expect(core.inspectedInputs).toEqual([structureInput]);
    expect(workspace.getSnapshot()).toMatchObject({
      structureInput,
      attemptedStructureInput: null,
      inspection,
      draft,
      operation: null,
      failure: null,
    });
  });

  it("ignores a superseded inspection failure without releasing its replacement", async () => {
    const obsolete = deferred<StructureInspection>();
    const replacement = deferred<StructureInspection>();
    const core = new CoreStub();
    core.inspectionResults = [obsolete.promise, replacement.promise];
    const workspace = createWorkspace(core);
    await workspace.dispatch({ type: "workspace.start" });

    const first = workspace.dispatch({
      type: "source.open",
      input: structureInput,
    });
    const replacementInput = {
      ...structureInput,
      structure_name: "replacement.cif",
    };
    const second = workspace.dispatch({
      type: "source.open",
      input: replacementInput,
    });
    obsolete.reject(
      new CoreFailure("invalid_structure", "Obsolete source failed", false),
    );
    await first;

    expect(workspace.getSnapshot()).toMatchObject({
      attemptedStructureInput: replacementInput,
      operation: "inspect",
      failure: null,
    });
    replacement.resolve(inspection);
    await second;
    expect(workspace.getSnapshot()).toMatchObject({
      structureInput: replacementInput,
      operation: null,
      failure: null,
    });
  });

  it("submits the exact request and stores the explain preview", async () => {
    const core = new CoreStub();
    core.explainResults = [Promise.resolve(explainResult)];
    const workspace = createWorkspace(core);
    await workspace.dispatch({ type: "workspace.start" });
    await workspace.dispatch({ type: "source.open", input: structureInput });

    await workspace.dispatch({ type: "review.compute" });

    expect(core.explainCalls).toEqual([baseRequest()]);
    expect(workspace.getSnapshot()).toMatchObject({
      reviewed: explainResult,
      outOfDate: false,
      operation: null,
      failure: null,
    });
  });

  it("merges overrides and marks the review out of date after a draft edit", async () => {
    const core = new CoreStub();
    core.explainResults = [Promise.resolve(explainResult)];
    const workspace = createWorkspace(core);
    await workspace.dispatch({ type: "workspace.start" });
    await workspace.dispatch({ type: "source.open", input: structureInput });
    await workspace.dispatch({ type: "review.compute" });

    await workspace.dispatch({
      type: "draft.patch",
      overrides: { k_grid: [5, 5, 5] },
    });

    expect(workspace.getSnapshot()).toMatchObject({
      draft: { overrides: { k_grid: [5, 5, 5] } },
      reviewed: explainResult,
      outOfDate: true,
    });
  });

  it("sends only the overrides that apply to the draft's task", async () => {
    // #154: the core refuses a setting no step of the task reads, so a
    // relax-only override kept in the draft must not reach an scf request.
    const core = new CoreStub();
    core.explainResults = [
      Promise.resolve(explainResult),
      Promise.resolve(explainResult),
    ];
    const workspace = createWorkspace(core);
    await workspace.dispatch({ type: "workspace.start" });
    await workspace.dispatch({ type: "source.open", input: structureInput });
    await workspace.dispatch({
      type: "draft.patch",
      task: "relax",
      overrides: { ion_dynamics: "fire", k_grid: [5, 5, 5], is_metal: "metal" },
    });
    await workspace.dispatch({ type: "review.compute" });

    await workspace.dispatch({ type: "draft.patch", task: "scf_single_point" });
    await workspace.dispatch({ type: "review.compute" });

    expect(core.explainCalls.map((request) => request.overrides)).toEqual([
      { ion_dynamics: "fire", k_grid: [5, 5, 5], is_metal: "metal" },
      { k_grid: [5, 5, 5], is_metal: "metal" },
    ]);
    expect(workspace.getSnapshot().draft?.overrides).toEqual({
      ion_dynamics: "fire",
      k_grid: [5, 5, 5],
      is_metal: "metal",
    });
  });

  it("clears an override back to automatic when patched with undefined", async () => {
    const core = new CoreStub();
    const workspace = createWorkspace(core);
    await workspace.dispatch({ type: "workspace.start" });
    await workspace.dispatch({ type: "source.open", input: structureInput });
    await workspace.dispatch({
      type: "draft.patch",
      overrides: { k_grid: [5, 5, 5] },
    });

    await workspace.dispatch({
      type: "draft.patch",
      overrides: { k_grid: undefined },
    });

    expect(workspace.getSnapshot().draft?.overrides).toEqual({});
  });

  it("retries a failed recomputation and replaces the reviewed snapshot", async () => {
    const failure = new CoreFailure(
      "advice_incomplete",
      "Core failed temporarily.",
      false,
    );
    const replacementResult: ExplainResult = {
      ...explainResult,
      warnings: [
        {
          code: "job.retry",
          level: "info",
          category: "job",
          message: "retried",
          record: "job",
        },
      ],
    };
    const core = new CoreStub();
    core.explainResults = [
      Promise.resolve(explainResult),
      Promise.reject(failure),
      Promise.resolve(replacementResult),
    ];
    const workspace = createWorkspace(core);
    await workspace.dispatch({ type: "workspace.start" });
    await workspace.dispatch({ type: "source.open", input: structureInput });
    await workspace.dispatch({ type: "review.compute" });
    await workspace.dispatch({
      type: "draft.patch",
      overrides: { k_grid: [5, 5, 5] },
    });
    await workspace.dispatch({ type: "review.compute" });

    await workspace.dispatch({ type: "failure.retry" });

    expect(core.explainCalls).toHaveLength(3);
    expect(workspace.getSnapshot()).toMatchObject({
      reviewed: replacementResult,
      outOfDate: false,
      failure: null,
    });
  });

  it("preserves the old review when recomputation fails", async () => {
    const failure = new CoreFailure(
      "advice_incomplete",
      "Core failed temporarily.",
      false,
    );
    const core = new CoreStub();
    core.explainResults = [
      Promise.resolve(explainResult),
      Promise.reject(failure),
    ];
    const workspace = createWorkspace(core);
    await workspace.dispatch({ type: "workspace.start" });
    await workspace.dispatch({ type: "source.open", input: structureInput });
    await workspace.dispatch({ type: "review.compute" });
    await workspace.dispatch({
      type: "draft.patch",
      overrides: { k_grid: [5, 5, 5] },
    });

    await workspace.dispatch({ type: "review.compute" });

    expect(workspace.getSnapshot()).toMatchObject({
      reviewed: explainResult,
      outOfDate: true,
      operation: null,
      failure,
      failureOperation: "explain",
    });
  });

  it("downloads the archive previewed for the exact reviewed request", async () => {
    const archive = buildArchive();
    const core = new CoreStub();
    core.explainResults = [Promise.resolve(explainResult)];
    core.archiveResults = [Promise.resolve(archive)];
    const saveArchive = vi.fn<(download: ArchiveDownload) => void>();
    const workspace = createWorkspace(core, saveArchive);
    await workspace.dispatch({ type: "workspace.start" });
    await workspace.dispatch({ type: "source.open", input: structureInput });
    await workspace.dispatch({ type: "review.compute" });
    await workspace.dispatch({ type: "review.refreshArchive" });
    expect(saveArchive).not.toHaveBeenCalled();

    await workspace.dispatch({ type: "review.download" });

    // Download saves the preview's own archive: one /run in total.
    expect(core.archiveCalls).toEqual([baseRequest()]);
    expect(saveArchive).toHaveBeenCalledOnce();
    expect(saveArchive.mock.calls[0]?.[0]).toBe(archive);
    expect(workspace.getSnapshot()).toMatchObject({
      lastDownload: archive,
      operation: null,
      failure: null,
    });
  });

  it("retries a failed automatic preview without saving the archive", async () => {
    // #189: Retry used to route a failed preview through review.download,
    // which fetched the archive and saved it as a file nobody asked for.
    const unavailable = new CoreFailure(
      "http_error",
      "The Goldilocks Core server rejected the request (503).",
      true,
      { status: 503 },
      503,
    );
    const archive = buildArchive();
    const core = new CoreStub();
    core.explainResults = [Promise.resolve(explainResult)];
    core.archiveResults = [
      Promise.reject(unavailable),
      Promise.resolve(archive),
    ];
    const saveArchive = vi.fn<(download: ArchiveDownload) => void>();
    const workspace = createWorkspace(core, saveArchive);
    await workspace.dispatch({ type: "workspace.start" });
    await workspace.dispatch({ type: "source.open", input: structureInput });
    await workspace.dispatch({ type: "review.compute" });
    await workspace.dispatch({ type: "review.refreshArchive" });
    expect(workspace.getSnapshot()).toMatchObject({
      lastDownload: null,
      failure: unavailable,
      failureOperation: "preview",
    });

    await workspace.dispatch({ type: "failure.retry" });

    expect(saveArchive).not.toHaveBeenCalled();
    expect(core.archiveCalls).toEqual([baseRequest(), baseRequest()]);
    expect(workspace.getSnapshot()).toMatchObject({
      lastDownload: archive,
      operation: null,
      failure: null,
      failureOperation: null,
    });
  });

  it("saves nothing and sends no /run when Download has no current archive", async () => {
    const core = new CoreStub();
    core.explainResults = [Promise.resolve(explainResult)];
    core.archiveResults = [Promise.resolve(buildArchive())];
    const saveArchive = vi.fn<(download: ArchiveDownload) => void>();
    const workspace = createWorkspace(core, saveArchive);
    await workspace.dispatch({ type: "workspace.start" });
    await workspace.dispatch({ type: "source.open", input: structureInput });
    await workspace.dispatch({ type: "review.compute" });

    await workspace.dispatch({ type: "review.download" });

    expect(core.archiveCalls).toEqual([]);
    expect(saveArchive).not.toHaveBeenCalled();
    expect(workspace.getSnapshot()).toMatchObject({
      lastDownload: null,
      operation: null,
      failure: null,
    });
  });

  it("downloads each selected ordering as the listed structure plus its label", async () => {
    // #170: a candidate is generated by its label on the structure that
    // was listed; it carries no structure of its own, and the draft's
    // starting_magnetization (keyed to the uploaded cell's sites) would
    // refuse the candidate's own species or override its seeds. #149: each
    // bundle is named after the listing's own structure.
    const fm = buildArchive("Si.cif-fm-scf_single_point.zip");
    const afm = buildArchive("Si.cif-afm-1-scf_single_point.zip");
    const core = new CoreStub();
    core.magneticOrderingsResults = [
      Promise.resolve(orderingsListing(2, "fm", "afm-1", "afm-2")),
    ];
    core.archiveResults = [Promise.resolve(fm), Promise.resolve(afm)];
    const saveArchive = vi.fn<(download: ArchiveDownload) => void>();
    const workspace = createWorkspace(core, saveArchive);
    await workspace.dispatch({ type: "workspace.start" });
    await workspace.dispatch({ type: "source.open", input: structureInput });
    await workspace.dispatch({
      type: "draft.patch",
      overrides: {
        spin_polarized: true,
        starting_magnetization: { Si0: 0.5 },
        magnetic_ordering: "afm",
      },
    });
    await workspace.dispatch({ type: "magneticOrderings.list" });

    await workspace.dispatch({
      type: "magneticOrderings.downloadSelected",
      labels: ["fm", "afm-1"],
    });

    expect(core.archiveCalls).toEqual([
      {
        ...baseRequest(),
        structure_name: `${structureInput.structure_name}-fm`,
        overrides: { spin_polarized: true, magnetic_ordering: "fm" },
      },
      {
        ...baseRequest(),
        structure_name: `${structureInput.structure_name}-afm-1`,
        overrides: { spin_polarized: true, magnetic_ordering: "afm-1" },
      },
    ]);
    expect(saveArchive.mock.calls).toEqual([[fm], [afm]]);
  });

  it("lists a replacement structure itself and drops the old structure's late listing", async () => {
    // #149: a listing that arrived after another structure opened was shown
    // for that structure, kept it from being listed itself, and Download
    // selected then named the old structure's bundles after the new one.
    const obsoleteListing = deferred<MagneticOrderingsResult>();
    const replacementResult = orderingsListing(1, "fm");
    const core = new CoreStub();
    core.inspectionResults = [
      Promise.resolve(inspection),
      Promise.resolve(inspection),
    ];
    core.magneticOrderingsResults = [
      obsoleteListing.promise,
      Promise.resolve(replacementResult),
    ];
    const workspace = createWorkspace(core, vi.fn());
    await workspace.dispatch({ type: "workspace.start" });
    await workspace.dispatch({ type: "source.open", input: structureInput });
    const listing = workspace.dispatch({ type: "magneticOrderings.list" });
    await workspace.dispatch({ type: "source.open", input: replacementInput });

    // A client that ignores the abort still delivers the old listing.
    obsoleteListing.resolve(orderingsListing(8, "fm", "afm-1"));
    await listing;
    expect(currentOrderings(workspace.getSnapshot())).toBeNull();
    await workspace.dispatch({
      type: "magneticOrderings.downloadSelected",
      labels: ["afm-1"],
    });
    expect(core.archiveCalls).toEqual([]);

    await workspace.dispatch({ type: "magneticOrderings.list" });

    expect(
      core.magneticOrderingsCalls.map((request) => request.structure_content),
    ).toEqual([
      structureInput.structure_content,
      replacementInput.structure_content,
    ]);
    expect(currentOrderings(workspace.getSnapshot())).toEqual({
      input: replacementInput,
      operation: null,
      result: replacementResult,
      error: null,
    });
  });

  it("aborts a listing in flight once another structure opens, and on reset", async () => {
    // #149: nothing could abort a listing, so it ran on after its structure
    // was replaced. An inspection that fails replaces nothing, so the
    // listing of the structure still open runs on.
    const unreadable = deferred<StructureInspection>();
    const obsoleteListing = deferred<MagneticOrderingsResult>();
    const replacementListing = deferred<MagneticOrderingsResult>();
    const core = new CoreStub();
    core.inspectionResults = [
      Promise.resolve(inspection),
      unreadable.promise,
      Promise.resolve(inspection),
    ];
    core.magneticOrderingsResults = [
      obsoleteListing.promise,
      replacementListing.promise,
    ];
    const workspace = createWorkspace(core, vi.fn());
    await workspace.dispatch({ type: "workspace.start" });
    await workspace.dispatch({ type: "source.open", input: structureInput });
    const listing = workspace.dispatch({ type: "magneticOrderings.list" });
    const [signal] = core.magneticOrderingsSignals;
    expect(signal).toBeInstanceOf(AbortSignal);

    const failedOpen = workspace.dispatch({
      type: "source.open",
      input: replacementInput,
    });
    unreadable.reject(
      new CoreFailure("invalid_structure", "Unreadable structure.", false),
    );
    await failedOpen;
    expect(signal?.aborted).toBe(false);
    expect(currentOrderings(workspace.getSnapshot())).toMatchObject({
      input: structureInput,
      operation: "list",
    });

    await workspace.dispatch({ type: "source.open", input: replacementInput });
    expect(signal?.aborted).toBe(true);
    // fetch rejects an aborted request with the signal's reason.
    obsoleteListing.reject(signal?.reason);
    await listing;
    expect(workspace.getSnapshot().magneticOrderings).toBeNull();

    const replacing = workspace.dispatch({ type: "magneticOrderings.list" });
    await workspace.dispatch({ type: "workspace.reset" });
    const replacementSignal = core.magneticOrderingsSignals[1];
    expect(replacementSignal?.aborted).toBe(true);
    replacementListing.reject(replacementSignal?.reason);
    await replacing;
    expect(workspace.getSnapshot().magneticOrderings).toBeNull();
  });

  it.each([
    {
      outcome: "is saved",
      settle: (archive: Deferred<ArchiveDownload>) => {
        archive.resolve(buildArchive("Si.cif-afm-1-scf_single_point.zip"));
      },
      saved: 1,
    },
    {
      outcome: "fails",
      settle: (archive: Deferred<ArchiveDownload>) => {
        archive.reject(
          new CoreFailure("advice_incomplete", "afm-1 is incomplete.", false),
        );
      },
      saved: 0,
    },
  ])(
    "leaves the replacement's listing alone when an old candidate download $outcome",
    async ({ settle, saved }) => {
      // #149: Download selected for the old structure finished by clearing
      // the replacement listing's busy flag (and showing its own error
      // there), so the replacement was listed a second time. The old
      // structure's archives are still saved, named and built from its own
      // listing.
      const candidateArchive = deferred<ArchiveDownload>();
      const replacementListing = deferred<MagneticOrderingsResult>();
      const replacementResult = orderingsListing(1, "fm");
      const core = new CoreStub();
      core.inspectionResults = [
        Promise.resolve(inspection),
        Promise.resolve(inspection),
      ];
      core.magneticOrderingsResults = [
        Promise.resolve(orderingsListing(4, "fm", "afm-1")),
        replacementListing.promise,
      ];
      core.archiveResults = [candidateArchive.promise];
      const saveArchive = vi.fn<(download: ArchiveDownload) => void>();
      const workspace = createWorkspace(core, saveArchive);
      await workspace.dispatch({ type: "workspace.start" });
      await workspace.dispatch({ type: "source.open", input: structureInput });
      await workspace.dispatch({ type: "magneticOrderings.list" });
      const downloading = workspace.dispatch({
        type: "magneticOrderings.downloadSelected",
        labels: ["afm-1"],
      });
      await workspace.dispatch({
        type: "source.open",
        input: replacementInput,
      });
      const listing = workspace.dispatch({ type: "magneticOrderings.list" });

      settle(candidateArchive);
      await downloading;
      // The replacement's listing is still in flight, so it is not re-sent.
      await workspace.dispatch({ type: "magneticOrderings.list" });
      replacementListing.resolve(replacementResult);
      await listing;

      expect(
        core.magneticOrderingsCalls.map((request) => request.structure_name),
      ).toEqual([
        structureInput.structure_name,
        replacementInput.structure_name,
      ]);
      expect(
        core.archiveCalls.map((request) => [
          request.structure_name,
          request.structure_content,
        ]),
      ).toEqual([["Si.cif-afm-1", structureInput.structure_content]]);
      expect(saveArchive).toHaveBeenCalledTimes(saved);
      expect(currentOrderings(workspace.getSnapshot())).toEqual({
        input: replacementInput,
        operation: null,
        result: replacementResult,
        error: null,
      });
    },
  );

  it("keeps the listing on screen while ranking it", async () => {
    const listing = orderingsListing(4, "fm", "afm-1");
    const rankedListing = { ...listing, ranked: true };
    const pendingRanking = deferred<MagneticOrderingsResult>();
    const core = new CoreStub();
    core.magneticOrderingsResults = [
      Promise.resolve(listing),
      pendingRanking.promise,
    ];
    const workspace = createWorkspace(core, vi.fn());
    await workspace.dispatch({ type: "workspace.start" });
    await workspace.dispatch({ type: "source.open", input: structureInput });
    await workspace.dispatch({ type: "magneticOrderings.list" });

    const ranking = workspace.dispatch({ type: "magneticOrderings.rank" });

    expect(core.magneticOrderingsCalls.at(-1)).toMatchObject({
      structure_content: structureInput.structure_content,
      rank_with_mmace: true,
    });
    expect(currentOrderings(workspace.getSnapshot())).toEqual({
      input: structureInput,
      operation: "rank",
      result: listing,
      error: null,
    });
    pendingRanking.resolve(rankedListing);
    await ranking;
    expect(currentOrderings(workspace.getSnapshot())).toEqual({
      input: structureInput,
      operation: null,
      result: rankedListing,
      error: null,
    });
  });

  it("refuses to download while the review is out of date", async () => {
    const archive = buildArchive();
    const core = new CoreStub();
    core.explainResults = [Promise.resolve(explainResult)];
    core.archiveResults = [Promise.resolve(archive)];
    const saveArchive = vi.fn<(download: ArchiveDownload) => void>();
    const workspace = createWorkspace(core, saveArchive);
    await workspace.dispatch({ type: "workspace.start" });
    await workspace.dispatch({ type: "source.open", input: structureInput });
    await workspace.dispatch({ type: "review.compute" });
    await workspace.dispatch({ type: "review.refreshArchive" });
    await workspace.dispatch({
      type: "draft.patch",
      overrides: { k_grid: [5, 5, 5] },
    });

    await workspace.dispatch({ type: "review.download" });

    // The archive in hand belongs to the draft before the edit.
    expect(workspace.getSnapshot()).toMatchObject({
      lastDownload: archive,
      outOfDate: true,
    });
    expect(saveArchive).not.toHaveBeenCalled();
    expect(core.archiveCalls).toHaveLength(1);
  });

  it("does not pair an obsolete explain with a replacement structure", async () => {
    const obsoleteExplain = deferred<ExplainResult>();
    const replacementInput: StructureInput = {
      structure_content: "Silicon B",
      structure_name: "POSCAR",
      structure_format: "poscar",
    };
    const inspectedReplacement: StructureInspection = {
      ...inspection,
      canonical_cif: "replacement",
    };
    const core = new CoreStub();
    core.inspectionResults = [
      Promise.resolve(inspection),
      Promise.resolve(inspectedReplacement),
    ];
    core.explainResults = [
      Promise.resolve(explainResult),
      obsoleteExplain.promise,
    ];
    const workspace = createWorkspace(core, vi.fn());
    await workspace.dispatch({ type: "workspace.start" });
    await workspace.dispatch({ type: "source.open", input: structureInput });
    await workspace.dispatch({ type: "review.compute" });

    const computing = workspace.dispatch({ type: "review.compute" });
    const replacing = workspace.dispatch({
      type: "source.open",
      input: replacementInput,
    });
    obsoleteExplain.resolve({ ...explainResult, warnings: [] });
    await computing;
    await replacing;

    expect(core.explainCalls).toHaveLength(2);
    expect(workspace.getSnapshot()).toMatchObject({
      structureInput: replacementInput,
      reviewed: null,
      operation: null,
    });
  });

  it("reset ignores obsolete source work and retains capabilities", async () => {
    let finishInspection: (value: StructureInspection) => void = () =>
      undefined;
    const pendingInspection = new Promise<StructureInspection>((resolve) => {
      finishInspection = resolve;
    });
    const core = new CoreStub();
    core.inspectionResults = [pendingInspection];
    const workspace = createWorkspace(core);
    await workspace.dispatch({ type: "workspace.start" });

    const opening = workspace.dispatch({
      type: "source.open",
      input: structureInput,
    });
    await workspace.dispatch({ type: "workspace.reset" });
    finishInspection(inspection);
    await opening;

    expect(workspace.getSnapshot()).toMatchObject({
      capabilities,
      structureInput: null,
      attemptedStructureInput: null,
      inspection: null,
      draft: null,
      reviewed: null,
      operation: null,
      failure: null,
    });
  });

  it("does not dismiss a capabilities failure before capabilities are usable", async () => {
    const failure = new CoreFailure(
      "assets_unavailable",
      "Runtime assets are unavailable.",
      true,
    );
    const core = new CoreStub();
    core.capabilitiesResult = Promise.reject(failure);
    const workspace = createWorkspace(core);
    await workspace.dispatch({ type: "workspace.start" });

    await workspace.dispatch({ type: "failure.dismiss" });

    expect(workspace.getSnapshot()).toMatchObject({
      capabilities: null,
      failure,
      failureOperation: "capabilities",
      operation: null,
    });
  });

  it("keeps a typed startup failure retryable", async () => {
    const core = new CoreStub();
    const failure = new CoreFailure(
      "assets_unavailable",
      "Runtime assets are unavailable.",
      true,
    );
    core.capabilitiesResult = Promise.reject(failure);
    const workspace = createWorkspace(core);

    await workspace.dispatch({ type: "workspace.start" });

    expect(workspace.getSnapshot()).toMatchObject({
      capabilities: null,
      operation: null,
      failure,
      failureOperation: "capabilities",
    });

    core.capabilitiesResult = Promise.resolve(capabilities);
    await workspace.dispatch({ type: "failure.retry" });
    expect(workspace.getSnapshot()).toMatchObject({
      capabilities,
      failure: null,
      failureOperation: null,
    });
  });

  it("retains pending capabilities ownership across reset and accepts its result", async () => {
    const pendingCapabilities = deferred<Capabilities>();
    const core = new CoreStub();
    core.capabilitiesResult = pendingCapabilities.promise;
    const workspace = createWorkspace(core);

    const starting = workspace.dispatch({ type: "workspace.start" });
    await workspace.dispatch({ type: "workspace.reset" });

    expect(workspace.getSnapshot()).toMatchObject({
      capabilities: null,
      operation: "capabilities",
      failure: null,
    });
    pendingCapabilities.resolve(capabilities);
    await starting;

    expect(core.capabilitiesCalls).toBe(1);
    expect(workspace.getSnapshot()).toMatchObject({
      capabilities,
      operation: null,
      failure: null,
    });
  });

  it("loads and stores one immutable capabilities snapshot at startup", async () => {
    const core = new CoreStub();
    const workspace = createWorkspace(core);

    await workspace.dispatch({ type: "workspace.start" });
    await workspace.dispatch({ type: "workspace.start" });

    expect(core.capabilitiesCalls).toBe(1);
    expect(workspace.getSnapshot()).toMatchObject({
      capabilities,
      operation: null,
      failure: null,
    });
  });
});

interface Deferred<T> {
  readonly promise: Promise<T>;
  readonly resolve: (value: T) => void;
  readonly reject: (reason: unknown) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve: (value: T) => void = () => undefined;
  let reject: (reason: unknown) => void = () => undefined;
  const promise = new Promise<T>((finish, fail) => {
    resolve = finish;
    reject = fail;
  });
  return { promise, resolve, reject };
}
