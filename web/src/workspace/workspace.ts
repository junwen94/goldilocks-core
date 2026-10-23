import { createStore } from "zustand/vanilla";

import type {
  ArchiveDownload,
  CalcTask,
  Capabilities,
  ComputeRequest,
  CoreClient,
  ExplainResult,
  MagneticOrderingsRequest,
  MagneticOrderingsResult,
  StructureInput,
  StructureInspection,
} from "../api/coreClient";
import { CoreFailure } from "../api/coreClient";

/** The Core request in flight. `preview` is the automatic `/run` that fetches
 * the archive behind `lastDownload`; saving a file is not an operation, since
 * `review.download` only saves an archive already fetched. */
export type WorkspaceOperation =
  "capabilities" | "inspect" | "explain" | "preview";

/** Independent of `WorkspaceOperation`: listing/ranking/downloading
 * magnetic-ordering candidates is an opt-in side-quest a user runs from
 * the Advisors card, not the main structure -> compute -> download
 * pipeline -- it must not disable the rest of that pipeline's controls
 * (or vice versa) while in flight. One nullable field, same idiom as
 * `WorkspaceOperation` itself, covering its own three concerns. */
export type MagneticOrderingsOperation = "list" | "rank" | "download";

/** A magnetic-ordering listing and the structure it was requested for.
 * `result`, `error` and the request in flight (`operation`) belong to
 * `input` alone: read them through `currentOrderings`, which returns the
 * listing only while `input` is the open structure, and candidate bundles
 * take their structure and name from `input` (#149). */
export interface MagneticOrderingsListing {
  readonly input: StructureInput;
  readonly operation: MagneticOrderingsOperation | null;
  /** The last listing or ranking that arrived; kept while a new request
   * for the same structure runs. */
  readonly result: MagneticOrderingsResult | null;
  /** Shown inline in the Magnetic Orderings card, not the global
   * `FailureBanner` -- this is a side-quest failure, not a reason to
   * block the main pipeline the banner exists for. Not re-sent on its
   * own: the card lists a structure only when it has no listing. */
  readonly error: string | null;
}

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
  /** The last `/explain` preview -- records, warnings and the `blocking`
   * verdict, no files. Never carries an archive: `/explain` never calls
   * `generate()`, so it succeeds even when a `/run` would refuse with
   * `advice_incomplete` (the tri-state "diagnosis is always available"
   * promise), and `blocking` then lists the reasons. The files come from
   * the separate `preview` operation against `/run` (`lastDownload`). */
  readonly reviewed: ExplainResult | null;
  readonly outOfDate: boolean;
  /** The archive the last `preview` fetched; it belongs to the current
   * draft only while `outOfDate` is false (see `currentArchive`). */
  readonly lastDownload: ArchiveDownload | null;
  readonly operation: WorkspaceOperation | null;
  readonly failure: CoreFailure | null;
  readonly failureOperation: WorkspaceOperation | null;
  /** Read through `currentOrderings`. */
  readonly magneticOrderings: MagneticOrderingsListing | null;
}

export type WorkspaceAction =
  | { readonly type: "workspace.start" }
  | { readonly type: "source.open"; readonly input: StructureInput }
  | {
      readonly type: "draft.patch";
      readonly code?: string;
      readonly task?: CalcTask;
      readonly hpc?: string | null;
      /** Shallow-merged into the current overrides. A value of
       * `undefined` clears that key back to "automatic" rather than
       * setting it to `undefined` on the wire. */
      readonly overrides?: Readonly<Record<string, unknown>>;
    }
  | { readonly type: "review.compute" }
  /** Fetches a fresh archive for live preview without saving it anywhere
   * -- the auto-preview effect calls this, not a user action. */
  | { readonly type: "review.refreshArchive" }
  /** Saves the current draft's archive (`currentArchive`) through the
   * archive sink. Sends no request: with no current archive it does
   * nothing, so a file is only ever saved by this explicit action. */
  | { readonly type: "review.download" }
  | { readonly type: "magneticOrderings.list" }
  | { readonly type: "magneticOrderings.rank" }
  | {
      readonly type: "magneticOrderings.downloadSelected";
      readonly labels: readonly string[];
    }
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
  magneticOrderings: null,
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

/** The request for `draft`. An override of a setting that does not apply
 * to the draft's task is left out, since the core refuses it (#154); the
 * draft keeps it, so switching back to a task it applies to sends it again.
 * A key that is not a setting (a fact such as `is_metal`) is always sent. */
function toComputeRequest(
  capabilities: Capabilities,
  structureInput: StructureInput,
  draft: CalculationDraft,
): ComputeRequest {
  const settings = new Map(
    capabilities.settings.map((setting) => [setting.key, setting]),
  );
  const overrides = Object.fromEntries(
    Object.entries(draft.overrides).filter(
      ([key]) => settings.get(key)?.tasks.includes(draft.task) ?? true,
    ),
  );
  return {
    structure_content: structureInput.structure_content,
    structure_name: structureInput.structure_name,
    ...(structureInput.structure_format
      ? { structure_format: structureInput.structure_format }
      : {}),
    code: draft.code,
    task: draft.task,
    hpc: draft.hpc,
    overrides,
    fetch_missing: draft.fetchMissing,
  };
}

function toMagneticOrderingsRequest(
  structureInput: StructureInput,
  rankWithMmace: boolean,
): MagneticOrderingsRequest {
  return {
    structure_content: structureInput.structure_content,
    structure_name: structureInput.structure_name,
    ...(structureInput.structure_format
      ? { structure_format: structureInput.structure_format }
      : {}),
    rank_with_mmace: rankWithMmace,
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

/** The archive fetched for the current draft of the current structure, or
 * null when none has arrived yet (or the draft changed since). The generated
 * inputs preview shows it, and `review.download` saves it and nothing else. */
export function currentArchive(
  snapshot: WorkspaceSnapshot,
): ArchiveDownload | null {
  return snapshot.outOfDate ? null : snapshot.lastDownload;
}

/** The magnetic-ordering listing of the open structure, or null when it
 * has none (nothing requested yet, or the listing belongs to another
 * structure). The card lists the open structure whenever this is null. */
export function currentOrderings(
  snapshot: WorkspaceSnapshot,
): MagneticOrderingsListing | null {
  const listing = snapshot.magneticOrderings;
  return listing?.input === snapshot.structureInput ? listing : null;
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
  // Aborts the listing request in flight; only the open structure's
  // listing can be in flight, since opening another structure aborts it.
  let orderingsRequest: AbortController | null = null;

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
        completeOperation(owner, {
          capabilities,
          draft: defaultDraft(capabilities),
        });
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
      // Only now is the open structure replaced: an inspection that fails
      // leaves it, and the listing request in flight for it, in place.
      abortOrderingsRequest();
      completeOperation(owner, {
        structureInput: input,
        attemptedStructureInput: null,
        inspection,
        draft: defaultDraft(capabilities),
        reviewed: null,
        outOfDate: false,
        lastDownload: null,
        magneticOrderings: null,
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
      ...("code" in action ? { code: action.code } : {}),
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
      snapshot.capabilities === null ||
      snapshot.structureInput === null ||
      snapshot.draft === null ||
      snapshot.operation !== null
    ) {
      return;
    }
    const revision = draftRevision;
    const request = toComputeRequest(
      snapshot.capabilities,
      snapshot.structureInput,
      snapshot.draft,
    );
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

  /** The `preview` operation: fetches the current draft's archive into
   * `lastDownload`, so `GeneratedInputReview` shows a live preview with no
   * separate "generate" step. Never saves it. The auto-preview effect and
   * Retry after a failed preview call this. */
  async function refreshArchive(): Promise<void> {
    const snapshot = store.getState();
    if (
      snapshot.capabilities === null ||
      snapshot.structureInput === null ||
      snapshot.draft === null ||
      snapshot.reviewed === null ||
      snapshot.outOfDate ||
      snapshot.operation !== null
    ) {
      return;
    }
    const revision = draftRevision;
    const request = toComputeRequest(
      snapshot.capabilities,
      snapshot.structureInput,
      snapshot.draft,
    );
    const owner = beginOperation("preview");
    try {
      const archive = await core.runArchive(request);
      completeOperation(owner, {
        lastDownload: archive,
        outOfDate: revision !== draftRevision,
      });
    } catch (error) {
      failOperation(owner, error);
    }
  }

  /** Saves the current archive, the one the preview shows. Fetches nothing:
   * without a current archive there is nothing to save (#189). */
  function downloadReviewed(): void {
    const archive = currentArchive(store.getState());
    if (archive !== null) saveArchive(archive);
  }

  function abortOrderingsRequest(): void {
    orderingsRequest?.abort();
    orderingsRequest = null;
  }

  /** Clears the busy state of the listing `pending` and applies `update`,
   * while `pending` is still the stored listing: the same owner rule as
   * `completeOperation`, so the outcome of a request for a structure that
   * is no longer open (or was reset away) is dropped. */
  function completeOrderings(
    pending: MagneticOrderingsListing,
    update: Partial<Pick<MagneticOrderingsListing, "result" | "error">> = {},
  ): void {
    if (store.getState().magneticOrderings !== pending) return;
    store.setState({
      magneticOrderings: { ...pending, ...update, operation: null },
    });
  }

  /** As `failOperation`: a `CoreFailure` becomes the listing's error, any
   * other error propagates once the listing is no longer busy. */
  function failOrderings(
    pending: MagneticOrderingsListing,
    error: unknown,
  ): void {
    if (store.getState().magneticOrderings !== pending) return;
    if (!(error instanceof CoreFailure)) {
      completeOrderings(pending);
      throw error;
    }
    completeOrderings(pending, { error: error.message });
  }

  /** Lists (or mMACE-ranks) the open structure's candidate orderings.
   * Refused while that structure's listing is busy; a result it already
   * has stays on screen until the new one arrives. */
  async function listMagneticOrderings(rankWithMmace: boolean): Promise<void> {
    const snapshot = store.getState();
    const input = snapshot.structureInput;
    const current = currentOrderings(snapshot);
    if (input === null || (current !== null && current.operation !== null)) {
      return;
    }
    const pending: MagneticOrderingsListing = {
      input,
      operation: rankWithMmace ? "rank" : "list",
      result: current?.result ?? null,
      error: null,
    };
    const request = new AbortController();
    orderingsRequest = request;
    store.setState({ magneticOrderings: pending });
    try {
      const result = await core.magneticOrderings(
        toMagneticOrderingsRequest(input, rankWithMmace),
        request.signal,
      );
      completeOrderings(pending, { result });
    } catch (error) {
      failOrderings(pending, error);
    }
  }

  /** Generates and downloads each selected candidate's DFT input bundle
   * -- one `runArchive` call per candidate, one browser download per
   * archive. A candidate is the listing's own structure (`input`, which
   * also names the bundle) with the draft's overrides plus its own
   * `magnetic_ordering` label (#170), which asks for a spin-polarized
   * run, `fm` included. The draft's `starting_magnetization` is left out
   * of every candidate: it is keyed by the uploaded cell's site labels,
   * which an ordered cell does not have, so each candidate starts from
   * its own ordering's seeds. Archives that arrive after another
   * structure opened are still saved; only the listing's busy state is
   * dropped with it. */
  async function downloadSelectedOrderings(
    labels: readonly string[],
  ): Promise<void> {
    const snapshot = store.getState();
    const { capabilities, draft } = snapshot;
    const listing = currentOrderings(snapshot);
    if (
      capabilities === null ||
      draft === null ||
      listing?.result == null ||
      listing.operation !== null ||
      labels.length === 0
    ) {
      return;
    }
    const { input, result } = listing;
    const pending: MagneticOrderingsListing = {
      ...listing,
      operation: "download",
      error: null,
    };
    store.setState({ magneticOrderings: pending });
    const overrides = Object.fromEntries(
      Object.entries(draft.overrides).filter(
        ([key]) => key !== "starting_magnetization",
      ),
    );
    let archives: readonly ArchiveDownload[];
    try {
      archives = await Promise.all(
        result.candidates
          .filter((candidate) => labels.includes(candidate.label))
          .map((candidate) =>
            core.runArchive({
              ...toComputeRequest(capabilities, input, {
                ...draft,
                overrides: { ...overrides, ...candidate.overrides },
              }),
              structure_name: `${input.structure_name}-${candidate.label}`,
            }),
          ),
      );
    } catch (error) {
      failOrderings(pending, error);
      return;
    }
    completeOrderings(pending);
    for (const archive of archives) saveArchive(archive);
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
      case "preview":
        return refreshArchive();
      case null:
        return;
    }
  }

  async function reset(): Promise<void> {
    const capabilities = store.getState().capabilities;
    const pendingStartup =
      startup !== null && activeOperation === startup.owner ? startup : null;
    draftRevision = 0;
    abortOrderingsRequest();
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
      case "review.refreshArchive":
        await refreshArchive();
        return;
      case "review.download":
        downloadReviewed();
        return;
      case "magneticOrderings.list":
        await listMagneticOrderings(false);
        return;
      case "magneticOrderings.rank":
        await listMagneticOrderings(true);
        return;
      case "magneticOrderings.downloadSelected":
        await downloadSelectedOrderings(action.labels);
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
