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
import { createWorkspace } from "../../src/workspace/workspace";

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

  magneticOrderings(
    request: MagneticOrderingsRequest,
  ): Promise<MagneticOrderingsResult> {
    this.magneticOrderingsCalls.push(request);
    return (
      this.magneticOrderingsResults.shift() ??
      Promise.reject(new Error("magnetic orderings not configured"))
    );
  }
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

  it("downloads the archive for the exact reviewed request", async () => {
    const archive = buildArchive();
    const core = new CoreStub();
    core.explainResults = [Promise.resolve(explainResult)];
    core.archiveResults = [Promise.resolve(archive)];
    const saveArchive = vi.fn<(download: ArchiveDownload) => void>();
    const workspace = createWorkspace(core, saveArchive);
    await workspace.dispatch({ type: "workspace.start" });
    await workspace.dispatch({ type: "source.open", input: structureInput });
    await workspace.dispatch({ type: "review.compute" });

    await workspace.dispatch({ type: "review.download" });

    expect(core.archiveCalls).toEqual([baseRequest()]);
    expect(saveArchive).toHaveBeenCalledWith(archive);
    expect(workspace.getSnapshot()).toMatchObject({
      lastDownload: archive,
      operation: null,
      failure: null,
    });
  });

  it("refuses to download while the review is out of date", async () => {
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

    await workspace.dispatch({ type: "review.download" });

    expect(core.archiveCalls).toHaveLength(0);
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

function deferred<T>(): {
  readonly promise: Promise<T>;
  readonly resolve: (value: T) => void;
  readonly reject: (reason: unknown) => void;
} {
  let resolve: (value: T) => void = () => undefined;
  let reject: (reason: unknown) => void = () => undefined;
  const promise = new Promise<T>((finish, fail) => {
    resolve = finish;
    reject = fail;
  });
  return { promise, resolve, reject };
}
