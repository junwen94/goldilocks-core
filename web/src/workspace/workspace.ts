import { createStore } from "zustand/vanilla";

import type {
  ArchiveDownload,
  CalcTask,
  Capabilities,
  ComputeRequest,
  CoreClient,
  ExplainResult,
  StructureInput,
  StructureInspection,
} from "../api/coreClient";
import { CoreFailure } from "../api/coreClient";

export type WorkspaceOperation =
  "capabilities" | "inspect" | "explain" | "download";

/** The v2 request shape has no `intent`/`hints` vocabulary: every tunable
 * knob is a flat `overrides` entry reflected from `capabilities().settings`/
 * `.facts` (see `capabilities.py`'s own "add an advisor, its settings
 * automatically appear" design goal). A key's absence from `overrides`
 * means "let the heuristic/ml/llm chain decide" -- there is no separate
 * "automatic" sentinel value, matching `--set`'s own semantics. */
export interface CalculationDraft {
  readonly code: string;
  readonly task: CalcTask;
  readonly hpc: string | null;
  readonly overrides: Readonly<Record<string, unknown>>;
  readonly fetchMissing: boolean;
}

export interface WorkspaceSnapshot {
  readonly capabilities: Capabilities | null;
  readonly structureInput: StructureInput | null;
  readonly attemptedStructureInput: StructureInput | null;
  readonly inspection: StructureInspection | null;
  readonly draft: CalculationDraft | null;
  /** The last `/explain` preview -- records + warnings, no files. Never
   * carries an archive: `/explain` never calls `generate()`, so it can
   * succeed even when a `/run` would still refuse with
   * `advice_incomplete` (the tri-state "diagnosis is always available"
   * promise). Downloading real files is a separate `review.download`
   * step against `/run`. */
  readonly reviewed: ExplainResult | null;
  readonly outOfDate: boolean;
  readonly lastDownload: ArchiveDownload | null;
  readonly operation: WorkspaceOperation | null;
  readonly failure: CoreFailure | null;
  readonly failureOperation: WorkspaceOperation | null;
}

export type WorkspaceAction =
  | { readonly type: "workspace.start" }
  | { readonly type: "source.open"; readonly input: StructureInput }
  | {
      readonly type: "draft.patch";
      readonly task?: CalcTask;
      readonly hpc?: string | null;
      /** Shallow-merged into the current overrides. A value of
       * `undefined` clears that key back to "automatic" rather than
       * setting it to `undefined` on the wire. */
      readonly overrides?: Readonly<Record<string, unknown>>;
    }
  | { readonly type: "review.compute" }
  | { readonly type: "review.download" }
  | { readonly type: "failure.retry" }
  | { readonly type: "failure.dismiss" }
  | { readonly type: "workspace.reset" };

export interface Workspace {
  getSnapshot(): WorkspaceSnapshot;
  subscribe(listener: () => void): () => void;
  dispatch(action: WorkspaceAction): Promise<void>;
}

type ArchiveSink = (archive: ArchiveDownload) => void;

interface OperationOwner {
  readonly operation: WorkspaceOperation;
}

const EMPTY_SNAPSHOT: WorkspaceSnapshot = {
  capabilities: null,
  structureInput: null,
  attemptedStructureInput: null,
  inspection: null,
  draft: null,
  reviewed: null,
  outOfDate: false,
  lastDownload: null,
  operation: null,
  failure: null,
  failureOperation: null,
};

function defaultDraft(capabilities: Capabilities): CalculationDraft {
  return {
    code: capabilities.codes[0]?.id ?? "quantum_espresso",
    task:
      (capabilities.tasks[0]?.id as CalcTask | undefined) ?? "scf_single_point",
    hpc: null,
    overrides: {},
    fetchMissing: false,
  };
}

function toComputeRequest(
  structureInput: StructureInput,
  draft: CalculationDraft,
): ComputeRequest {
  return {
    structure_content: structureInput.structure_content,
    structure_name: structureInput.structure_name,
    ...(structureInput.structure_format
      ? { structure_format: structureInput.structure_format }
      : {}),
    code: draft.code,
    task: draft.task,
    hpc: draft.hpc,
    overrides: draft.overrides,
    fetch_missing: draft.fetchMissing,
  };
}

function mergeOverrides(
  current: Readonly<Record<string, unknown>>,
  patch: Readonly<Record<string, unknown>>,
): Readonly<Record<string, unknown>> {
  const merged = { ...current, ...patch };
  return Object.fromEntries(
    Object.entries(merged).filter(([, value]) => value !== undefined),
  );
}

export function createWorkspace(
  core: CoreClient,
  saveArchive: ArchiveSink = saveArchiveToBrowser,
): Workspace {
  const store = createStore<WorkspaceSnapshot>(() => EMPTY_SNAPSHOT);
  let activeOperation: OperationOwner | null = null;
  let startup: {
    readonly owner: OperationOwner;
    readonly promise: Promise<void>;
  } | null = null;
  let draftRevision = 0;

  function beginOperation(operation: WorkspaceOperation): OperationOwner {
    const owner = { operation };
    activeOperation = owner;
    store.setState({
      operation,
      failure: null,
      failureOperation: null,
    });
    return owner;
  }

  function completeOperation(
    owner: OperationOwner,
    update: Partial<WorkspaceSnapshot> = {},
  ): void {
    if (activeOperation !== owner) return;
    activeOperation = null;
    store.setState({ ...update, operation: null });
  }

  function failOperation(owner: OperationOwner, error: unknown): void {
    if (activeOperation !== owner) return;
    if (!(error instanceof CoreFailure)) {
      completeOperation(owner);
      throw error;
    }
    completeOperation(owner, {
      failure: error,
      failureOperation: owner.operation,
    });
  }

  function start(): Promise<void> {
    if (store.getState().capabilities !== null) return Promise.resolve();
    if (startup !== null) return startup.promise;
    const owner = beginOperation("capabilities");
    const promise = core.capabilities().then(
      (capabilities) => {
        if (startup?.owner === owner) startup = null;
        completeOperation(owner, { capabilities });
      },
      (error: unknown) => {
        if (startup?.owner === owner) startup = null;
        failOperation(owner, error);
      },
    );
    startup = { owner, promise };
    return promise;
  }

  async function openSource(input: StructureInput): Promise<void> {
    const capabilities = store.getState().capabilities;
    if (capabilities === null) return;
    const owner = beginOperation("inspect");
    store.setState({ attemptedStructureInput: input });
    try {
      const inspection = await core.inspectStructure(input);
      if (activeOperation !== owner) return;
      draftRevision = 0;
      completeOperation(owner, {
        structureInput: input,
        attemptedStructureInput: null,
        inspection,
        draft: defaultDraft(capabilities),
        reviewed: null,
        outOfDate: false,
        lastDownload: null,
      });
    } catch (error) {
      failOperation(owner, error);
    }
  }

  function patchDraft(
    action: Extract<WorkspaceAction, { type: "draft.patch" }>,
  ): void {
    const snapshot = store.getState();
    const currentDraft = snapshot.draft;
    if (currentDraft === null || snapshot.operation === "inspect") return;
    draftRevision += 1;
    const draft: CalculationDraft = {
      ...currentDraft,
      ...("task" in action ? { task: action.task } : {}),
      ...("hpc" in action ? { hpc: action.hpc } : {}),
      overrides:
        action.overrides === undefined
          ? currentDraft.overrides
          : mergeOverrides(currentDraft.overrides, action.overrides),
    };
    store.setState({
      draft,
      outOfDate: snapshot.reviewed !== null,
      failure: null,
      failureOperation: null,
    });
  }

  async function computeReview(): Promise<void> {
    const snapshot = store.getState();
    if (
      snapshot.structureInput === null ||
      snapshot.draft === null ||
      snapshot.operation !== null
    ) {
      return;
    }
    const revision = draftRevision;
    const request = toComputeRequest(snapshot.structureInput, snapshot.draft);
    const owner = beginOperation("explain");
    try {
      const reviewed = await core.explain(request);
      completeOperation(owner, {
        reviewed,
        outOfDate: revision !== draftRevision,
        lastDownload: null,
      });
    } catch (error) {
      failOperation(owner, error);
    }
  }

  async function downloadReviewed(): Promise<void> {
    const snapshot = store.getState();
    if (
      snapshot.structureInput === null ||
      snapshot.draft === null ||
      snapshot.reviewed === null ||
      snapshot.outOfDate ||
      snapshot.operation !== null
    ) {
      return;
    }
    const request = toComputeRequest(snapshot.structureInput, snapshot.draft);
    const owner = beginOperation("download");
    try {
      const archive = await core.runArchive(request);
      completeOperation(owner, { lastDownload: archive });
      saveArchive(archive);
    } catch (error) {
      failOperation(owner, error);
    }
  }

  async function retryFailure(): Promise<void> {
    const { failureOperation, attemptedStructureInput } = store.getState();
    switch (failureOperation) {
      case "capabilities":
        return start();
      case "inspect":
        if (attemptedStructureInput !== null) {
          await openSource(attemptedStructureInput);
        }
        return;
      case "explain":
        return computeReview();
      case "download":
        return downloadReviewed();
      case null:
        return;
    }
  }

  async function reset(): Promise<void> {
    const capabilities = store.getState().capabilities;
    const pendingStartup =
      startup !== null && activeOperation === startup.owner ? startup : null;
    draftRevision = 0;
    if (pendingStartup === null) {
      activeOperation = null;
      startup = null;
    }
    store.setState(
      {
        ...EMPTY_SNAPSHOT,
        capabilities,
        operation: pendingStartup?.owner.operation ?? null,
      },
      true,
    );
    if (capabilities === null && pendingStartup === null) await start();
  }

  async function dispatch(action: WorkspaceAction): Promise<void> {
    switch (action.type) {
      case "workspace.start":
        await start();
        return;
      case "source.open":
        await openSource(action.input);
        return;
      case "draft.patch":
        patchDraft(action);
        return;
      case "review.compute":
        await computeReview();
        return;
      case "review.download":
        await downloadReviewed();
        return;
      case "failure.retry":
        return retryFailure();
      case "failure.dismiss":
        if (store.getState().capabilities === null) return;
        store.setState({
          attemptedStructureInput: null,
          failure: null,
          failureOperation: null,
        });
        return;
      case "workspace.reset":
        return reset();
    }
  }

  return {
    getSnapshot: store.getState,
    subscribe(listener): () => void {
      return store.subscribe(listener);
    },
    dispatch,
  };
}

export function saveArchiveToBrowser(archive: ArchiveDownload): void {
  const url = URL.createObjectURL(archive.blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = archive.filename;
  anchor.hidden = true;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
