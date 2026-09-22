import type { components } from "./schema";

// Request bodies are generated from openapi.json (FastAPI infers a real
// schema from these pydantic models). Response bodies are NOT generated:
// every route in src/goldilocks_core/server/http.py returns a bare
// dict/TypedDict (typed `Any`), so FastAPI has nothing to build an
// OpenAPI response schema from and openapi-typescript resolves every
// 200 response to `unknown`. The response types below are hand-authored
// to mirror the real Python shapes; see the comment above each group for
// its source of truth.
export type StructureInput = components["schemas"]["InlineStructureDocument"];
export type ComputeRequest = components["schemas"]["ComputeRequestDocument"];
export type CalcTask = ComputeRequest["task"];
export type MagneticOrderingsRequest =
  components["schemas"]["MagneticOrderingsRequestDocument"];

export type Source = "human" | "ml" | "llm" | "heuristic";
export type FieldStatus = "resolved" | "unavailable" | "blocked";

/** Mirrors `resolution.ResolvedField`'s wire projection (`from_state`). */
export interface ResolvedField<T = unknown> {
  readonly status: FieldStatus;
  readonly value?: T | null;
  readonly source?: Source | null;
  readonly field_sources?: Readonly<Record<string, Source>> | null;
  readonly reason?: string | null;
  readonly blocked_by?: string | null;
}

/** Mirrors `resolution.Warning`. */
export interface AdvisorWarning {
  readonly code: string;
  readonly level: "info" | "warning" | "error";
  readonly category: string;
  readonly message: string;
}

// ---------------------------------------------------------------------
// GET /capabilities -- mirrors capabilities.py's `Capabilities` TypedDict
// and the Setting/Fact projections `_settings()`/`_facts()` build.
// ---------------------------------------------------------------------

export interface Setting {
  readonly key: string;
  readonly group: string;
  readonly type: string;
  readonly enum?: readonly string[];
  readonly unit: string | null;
  readonly default?: unknown;
  readonly enum_from?: string;
  readonly codes: readonly string[] | null;
  readonly tasks: readonly string[] | null;
  readonly programs: readonly string[] | null;
  readonly scope: "system" | "per_step";
  readonly ml_target: string | null;
  readonly approaches: readonly Source[];
  readonly description: string;
}

export interface Fact {
  readonly key: string;
  readonly type: string;
  readonly values: readonly string[] | null;
  readonly ml_target: string | null;
  readonly approaches: readonly Source[];
  readonly overridable: boolean;
  readonly description: string;
}

export interface CodeInfo {
  readonly id: string;
  readonly name: string;
  readonly tasks: readonly string[];
}

export interface TaskInfo {
  readonly id: string;
  readonly name: string;
  readonly description: string;
  readonly codes: readonly string[];
  readonly step_count: number;
  readonly executables: readonly string[];
}

export interface PseudopotentialTable {
  readonly id: string;
  readonly provider: string;
  readonly version: string;
  readonly functional: string;
  readonly relativistic: string;
  readonly accuracy: string;
  readonly elements: readonly string[];
  readonly licence: string;
  readonly citation: string;
  readonly default: boolean;
}

export interface HpcProfile {
  readonly id: string;
  readonly name: string;
  readonly scheduler: string;
  readonly partitions: readonly string[];
}

export interface Capabilities {
  readonly core_version: string;
  readonly vocabulary_version: string;
  readonly codes: readonly CodeInfo[];
  readonly tasks: readonly TaskInfo[];
  readonly facts: readonly Fact[];
  readonly settings: readonly Setting[];
  readonly pseudopotential_tables: readonly PseudopotentialTable[];
  readonly hpc_profiles: readonly HpcProfile[];
  readonly models: readonly Readonly<Record<string, unknown>>[];
  readonly warnings: readonly AdvisorWarning[];
  readonly sources: readonly Source[];
}

/** `capabilities().vocabulary_version` this client was built against.
 * Bumped in lockstep with `capabilities.VOCABULARY_VERSION` -- guards
 * against exactly the failure mode that made this rewrite necessary:
 * a frontend build silently drifting from the backend contract it
 * talks to, discovered only by runtime breakage deep in a form. */
const SUPPORTED_VOCABULARY_VERSION = "1";

// ---------------------------------------------------------------------
// POST /inspect -- mirrors inputs/structure.py's `StructureInspection`.
// ---------------------------------------------------------------------

export interface SpeciesOccupancy {
  readonly symbol: string;
  readonly label: string;
  readonly occupancy: number;
  readonly oxidation_state: number | null;
}

export interface StructureSite {
  readonly fractional_coordinates: readonly [number, number, number];
  readonly cartesian_coordinates_angstrom: readonly [number, number, number];
  readonly species: readonly SpeciesOccupancy[];
}

export interface LatticeDocument {
  readonly vectors_angstrom: readonly [
    readonly [number, number, number],
    readonly [number, number, number],
    readonly [number, number, number],
  ];
  readonly lengths_angstrom: readonly [number, number, number];
  readonly angles_degrees: readonly [number, number, number];
  readonly volume_angstrom3: number;
}

export interface StructureDocument {
  readonly schema_version: number;
  readonly formula: string;
  readonly reduced_formula: string;
  readonly site_count: number;
  readonly lattice: LatticeDocument;
  readonly periodicity: readonly [boolean, boolean, boolean];
  readonly sites: readonly StructureSite[];
}

export interface StructureSourceDocument {
  readonly origin: "inline" | "path" | "generated";
  readonly name: string;
  readonly format: string;
  readonly content: string | null;
  readonly sha256: string | null;
  readonly size_bytes: number | null;
}

export interface StructureInspection {
  readonly source: StructureSourceDocument;
  readonly structure: StructureDocument;
  readonly canonical_cif: string;
  readonly schema_version: number;
}

// ---------------------------------------------------------------------
// POST /explain, POST /run -- mirror server/_handlers.py's return shapes.
// Neither carries a `schema_version` field (unlike /inspect): only
// `InlineStructureDocument`'s consumer (`normalize_structure(...)
// .inspection`) has ever set one.
// ---------------------------------------------------------------------

export interface ExplainResult {
  readonly records: Readonly<Record<string, ResolvedField>>;
  readonly warnings: readonly AdvisorWarning[];
}

export interface RunResult {
  readonly files: readonly string[];
  readonly records: Readonly<Record<string, ResolvedField>>;
  readonly warnings: readonly AdvisorWarning[];
}

// ---------------------------------------------------------------------
// POST /magnetic-orderings -- mirrors service/_magnetic_orderings.py's
// `report_to_json`/`candidate_to_json`. `structure_content`/`overrides`
// are everything needed to generate *this exact* candidate's own input
// files: feed them straight back as a brand new top-level structure +
// overrides to `run`/`runArchive`, merged on top of the current draft's
// own overrides (see `_bundle_inputs`'s docstring for why an AFM
// candidate needs an explicit `starting_magnetization` override -- CIF
// text cannot round-trip the sign that distinguishes its two magnetic
// sublattices).
// ---------------------------------------------------------------------

export interface MagneticOrderingCandidate {
  readonly label: string;
  readonly formula: string;
  readonly natoms: number;
  readonly energy_per_atom_ev: number | null;
  readonly status: string | null;
  readonly is_recommended: boolean;
  readonly structure_content: string;
  readonly structure_format: "cif";
  readonly overrides: Readonly<Record<string, unknown>>;
}

export interface MagneticOrderingsResult {
  readonly ranked: boolean;
  readonly candidates: readonly MagneticOrderingCandidate[];
  readonly warnings: readonly AdvisorWarning[];
}

export interface ArchiveDownload {
  readonly blob: Blob;
  readonly filename: string;
}

export class CoreFailure extends Error {
  constructor(
    readonly kind: string,
    message: string,
    readonly retryable: boolean,
    readonly details: Readonly<Record<string, unknown>> = {},
    readonly status: number | null = null,
    readonly rawResponse?: unknown,
  ) {
    super(message);
    this.name = "CoreFailure";
  }
}

export interface CoreClient {
  capabilities(): Promise<Capabilities>;
  inspectStructure(input: StructureInput): Promise<StructureInspection>;
  /** Advice-only preview: never generates files, can succeed even when
   * `run` would refuse with `advice_incomplete` (the tri-state
   * "diagnosis is always available" promise). */
  explain(request: ComputeRequest): Promise<ExplainResult>;
  /** Lists candidate magnetic orderings (FM plus any AFM candidates),
   * optionally mMACE-ranked by relaxed energy per atom. */
  magneticOrderings(
    request: MagneticOrderingsRequest,
  ): Promise<MagneticOrderingsResult>;
  /** Executes the pipeline and returns the JSON summary (file list +
   * records + warnings), not the files themselves. */
  run(request: ComputeRequest): Promise<RunResult>;
  /** Executes the pipeline and returns the generated input bundle as a
   * zip. Runs the pipeline a second time server-side -- there is no
   * single call that returns both the summary and the archive (v1's
   * multipart response is gone); callers that need both should call
   * `explain` for the summary/preview and `runArchive` only once the
   * user actually wants the files. */
  runArchive(request: ComputeRequest): Promise<ArchiveDownload>;
}

export class HttpCoreClient implements CoreClient {
  private readonly baseUrl: string;

  constructor(
    baseUrl = "",
    private readonly fetcher: typeof fetch = globalThis.fetch.bind(globalThis),
  ) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  async capabilities(): Promise<Capabilities> {
    const response = await this.request("/capabilities", {
      headers: { Accept: "application/json" },
      method: "GET",
    });
    const capabilities = await parseJson<Capabilities>(response);
    if (capabilities.vocabulary_version !== SUPPORTED_VOCABULARY_VERSION) {
      throw new CoreFailure(
        "incompatible_vocabulary",
        `Goldilocks Core reports vocabulary_version ${capabilities.vocabulary_version}, ` +
          `but this Workbench build expects ${SUPPORTED_VOCABULARY_VERSION}. ` +
          "The frontend and backend were built against different contracts -- " +
          "regenerate the Workbench's API types against this server.",
        false,
        { serverVocabularyVersion: capabilities.vocabulary_version },
      );
    }
    return capabilities;
  }

  async inspectStructure(input: StructureInput): Promise<StructureInspection> {
    const response = await this.request("/inspect", {
      body: JSON.stringify(input),
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      method: "POST",
    });
    return parseVersionedJson<StructureInspection>(response);
  }

  async explain(request: ComputeRequest): Promise<ExplainResult> {
    const response = await this.request("/explain", {
      body: JSON.stringify(request),
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      method: "POST",
    });
    return parseJson<ExplainResult>(response);
  }

  async magneticOrderings(
    request: MagneticOrderingsRequest,
  ): Promise<MagneticOrderingsResult> {
    const response = await this.request("/magnetic-orderings", {
      body: JSON.stringify(request),
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      method: "POST",
    });
    return parseJson<MagneticOrderingsResult>(response);
  }

  async run(request: ComputeRequest): Promise<RunResult> {
    const response = await this.request("/run", {
      body: JSON.stringify({ ...request, respond_with: "json" }),
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      method: "POST",
    });
    return parseJson<RunResult>(response);
  }

  async runArchive(request: ComputeRequest): Promise<ArchiveDownload> {
    const response = await this.request("/run", {
      body: JSON.stringify({ ...request, respond_with: "archive" }),
      headers: {
        Accept: "application/zip",
        "Content-Type": "application/json",
      },
      method: "POST",
    });
    return parseArchive(response, request);
  }

  private async request(path: string, init: RequestInit): Promise<Response> {
    let response: Response;
    try {
      response = await this.fetcher(`${this.baseUrl}${path}`, init);
    } catch (error) {
      throw new CoreFailure(
        "network_error",
        error instanceof Error
          ? `Cannot reach Goldilocks Core: ${error.message}`
          : "Cannot reach Goldilocks Core.",
        true,
      );
    }
    await ensureSuccess(response);
    return response;
  }
}

async function parseJson<T>(response: Response): Promise<T> {
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.toLowerCase().startsWith("application/json")) {
    const rawResponse = await decodeResponse(response);
    throw invalidResponse(
      response.status,
      "returned an invalid JSON response",
      rawResponse,
    );
  }
  try {
    return (await response.json()) as T;
  } catch {
    throw invalidResponse(response.status, "returned unreadable JSON");
  }
}

async function parseVersionedJson<T>(response: Response): Promise<T> {
  const payload = await parseJson<unknown>(response);
  requireSchemaVersion(payload, response.status);
  return payload as T;
}

function requireSchemaVersion(payload: unknown, status: number): void {
  if (
    payload === null ||
    typeof payload !== "object" ||
    !("schema_version" in payload) ||
    payload.schema_version !== 1
  ) {
    throw invalidResponse(
      status,
      "returned an incompatible schema version",
      payload,
    );
  }
}

async function ensureSuccess(response: Response): Promise<void> {
  if (response.ok) return;
  const rawResponse = await decodeResponse(response);
  if (isErrorEnvelope(rawResponse)) {
    const { error } = rawResponse;
    throw new CoreFailure(
      error.kind,
      error.message,
      response.status >= 500,
      error.details ?? {},
      response.status,
      rawResponse,
    );
  }
  throw new CoreFailure(
    "http_error",
    `The Goldilocks Core server rejected the request (${String(response.status)}).`,
    response.status >= 500,
    { status: response.status },
    response.status,
    rawResponse,
  );
}

async function decodeResponse(response: Response): Promise<unknown> {
  const text = await response.text();
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

// The wire error envelope (server/documents.py's `ErrorResponseDocument`)
// no longer carries `retryable` -- v1's error taxonomy did, v2's
// `ExpectedFailure.public_error()` deliberately only returns
// `{kind, message}` (`details` is additive, added ad hoc by validation/
// readiness handlers). `CoreFailure.retryable` above is now always
// derived from the HTTP status/network condition, never read off the
// wire.
function isErrorEnvelope(value: unknown): value is {
  readonly error: {
    readonly kind: string;
    readonly message: string;
    readonly details?: Readonly<Record<string, unknown>> | null;
  };
} {
  if (value === null || typeof value !== "object" || !("error" in value)) {
    return false;
  }
  const error = value.error;
  return (
    error !== null &&
    typeof error === "object" &&
    "kind" in error &&
    typeof error.kind === "string" &&
    "message" in error &&
    typeof error.message === "string"
  );
}

async function parseArchive(
  response: Response,
  request: ComputeRequest,
): Promise<ArchiveDownload> {
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.toLowerCase().startsWith("application/zip")) {
    throw invalidResponse(
      response.status,
      "returned an invalid archive response",
    );
  }
  let blob: Blob;
  try {
    blob = await response.blob();
  } catch {
    throw invalidResponse(response.status, "returned unreadable archive data");
  }
  // No Content-Disposition header is sent (server/http.py's /run archive
  // branch returns a bare `Response(..., media_type="application/zip")`)
  // -- synthesize a reasonably informative name client-side instead.
  return { blob, filename: `${safeArchiveStem(request)}.zip` };
}

function safeArchiveStem(request: ComputeRequest): string {
  const name = request.structure_name.replace(/[^a-zA-Z0-9._-]+/g, "_");
  const task = request.task;
  return name.length > 0 ? `${name}-${task}` : `goldilocks-${task}`;
}

function invalidResponse(
  status: number,
  reason: string,
  rawResponse?: unknown,
): CoreFailure {
  return new CoreFailure(
    "invalid_response",
    `Goldilocks Core ${reason}.`,
    false,
    {},
    status,
    rawResponse,
  );
}
