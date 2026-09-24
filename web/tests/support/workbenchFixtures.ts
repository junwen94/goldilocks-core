import { strToU8, zipSync } from "fflate";

import type {
  ArchiveDownload,
  Capabilities,
  ExplainResult,
  PseudopotentialTable,
  ResolvedField,
  RunResult,
  StructureInput,
  StructureInspection,
} from "../../src/api/coreClient";
import type { CalculationDraft } from "../../src/workspace/workspace";

export const structureInput: StructureInput = {
  structure_content: "data_Si",
  structure_name: "Si.cif",
  structure_format: "cif",
};

const pseudoTable: PseudopotentialTable = {
  id: "pseudodojo-pbesol-efficiency-sr",
  provider: "pseudodojo",
  version: "0.4",
  functional: "PBEsol",
  relativistic: "scalar",
  accuracy: "efficiency",
  elements: ["Si"],
  licence: "CC-BY-4.0",
  citation: "van Setten et al.",
  default: true,
};

const pbeTable: PseudopotentialTable = {
  ...pseudoTable,
  id: "sssp-pbe-precision-sr",
  provider: "sssp",
  version: "1.3.0",
  functional: "PBE",
  accuracy: "precision",
  default: false,
};

export const capabilities: Capabilities = {
  core_version: "1.2.3",
  vocabulary_version: "1",
  codes: [
    {
      id: "quantum_espresso",
      name: "Quantum ESPRESSO",
      tasks: ["scf_single_point", "relax"],
    },
  ],
  tasks: [
    {
      id: "scf_single_point",
      name: "Single-point SCF",
      description: "One self-consistent-field calculation, no relaxation.",
      codes: ["quantum_espresso"],
      step_count: 1,
      executables: ["pw.x"],
    },
    {
      id: "relax",
      name: "Ionic relaxation",
      description: "scf plus BFGS/damped/FIRE ionic-position optimization.",
      codes: ["quantum_espresso"],
      step_count: 1,
      executables: ["pw.x"],
    },
  ],
  facts: [
    {
      key: "needs_soc",
      type: "boolean",
      values: null,
      ml_target: null,
      approaches: ["human", "heuristic"],
      overridable: true,
      description: "Whether spin-orbit coupling is likely relevant.",
    },
  ],
  settings: [
    {
      key: "functional",
      group: "functional",
      type: "string",
      unit: null,
      default: "PBEsol",
      enum_from: "pseudopotential_tables.functional",
      codes: null,
      tasks: null,
      programs: null,
      scope: "system",
      ml_target: null,
      approaches: ["human", "heuristic"],
      description: "Exchange-correlation functional label.",
    },
    {
      key: "pseudo_table_id",
      group: "pseudo_table_id",
      type: "string",
      unit: null,
      codes: null,
      tasks: null,
      programs: null,
      scope: "system",
      ml_target: null,
      approaches: ["human", "heuristic"],
      description: "Pin an explicit pseudopotential table id.",
    },
    {
      key: "spin_polarized",
      group: "magnetic",
      type: "boolean",
      unit: null,
      codes: null,
      tasks: null,
      programs: null,
      scope: "system",
      ml_target: null,
      approaches: ["human", "heuristic"],
      description: "Force spin-polarized vs non-spin-polarized.",
    },
    {
      key: "smearing_type",
      group: "occupations",
      type: "string",
      enum: ["cold", "gaussian", "mp", "fixed"],
      unit: null,
      codes: null,
      tasks: null,
      programs: ["pw.x"],
      scope: "per_step",
      ml_target: null,
      approaches: ["human", "heuristic"],
      description: "Smearing function used when occupations=smearing.",
    },
    {
      key: "degauss",
      group: "occupations",
      type: "number",
      unit: "Ry",
      codes: null,
      tasks: null,
      programs: ["pw.x"],
      scope: "per_step",
      ml_target: null,
      approaches: ["human", "heuristic"],
      description: "Smearing width used when occupations=smearing.",
    },
    {
      key: "k_grid",
      group: "k_sampling",
      type: "array",
      unit: null,
      codes: null,
      tasks: null,
      programs: ["pw.x"],
      scope: "per_step",
      ml_target: null,
      approaches: ["human", "heuristic"],
      description: "Explicit Monkhorst-Pack mesh dimensions.",
    },
    {
      key: "k_distance",
      group: "k_sampling",
      type: "number",
      unit: "Å⁻¹",
      codes: null,
      tasks: null,
      programs: ["pw.x"],
      scope: "per_step",
      ml_target: "k_distance",
      approaches: ["human", "ml", "heuristic"],
      description: "Target reciprocal-space spacing between k-points.",
    },
    {
      key: "nosym",
      group: "n_irr_k",
      type: "boolean",
      unit: null,
      codes: null,
      tasks: null,
      programs: ["pw.x"],
      scope: "per_step",
      ml_target: null,
      approaches: ["human", "heuristic"],
      description: "Disable symmetry reduction when counting k-points.",
    },
    {
      key: "ion_dynamics",
      group: "relax",
      type: "string",
      enum: ["bfgs", "damp", "fire"],
      unit: null,
      default: "bfgs",
      codes: null,
      tasks: null,
      programs: ["pw.x"],
      scope: "per_step",
      ml_target: null,
      approaches: ["human", "heuristic"],
      description: "Ionic relaxation algorithm for relax/vc-relax.",
    },
  ],
  pseudopotential_tables: [pseudoTable, pbeTable],
  hpc_profiles: [
    { id: "scarf", name: "SCARF", scheduler: "slurm", partitions: ["compute"] },
  ],
  models: [],
  warnings: [],
  sources: ["human", "ml", "llm", "heuristic"],
};

export const inspection: StructureInspection = {
  schema_version: 1,
  canonical_cif: "data_Si",
  source: {
    origin: "inline",
    name: "Si.cif",
    format: "cif",
    content: "data_Si",
    sha256: "a".repeat(64),
    size_bytes: 7,
  },
  structure: {
    schema_version: 1,
    formula: "Si1",
    reduced_formula: "Si",
    site_count: 1,
    periodicity: [true, true, true],
    lattice: {
      vectors_angstrom: [
        [4, 0, 0],
        [0, 4, 0],
        [0, 0, 4],
      ],
      lengths_angstrom: [4, 4, 4],
      angles_degrees: [90, 90, 90],
      volume_angstrom3: 64,
    },
    sites: [
      {
        fractional_coordinates: [0, 0, 0],
        cartesian_coordinates_angstrom: [0, 0, 0],
        species: [
          { symbol: "Si", label: "Si", occupancy: 1, oxidation_state: null },
        ],
      },
    ],
  },
};

/** The workspace's own default draft, seeded from `capabilities` above
 * right after a successful `source.open` -- exported so tests can
 * assert against it without re-deriving `defaultDraft`'s logic. */
export const draft: CalculationDraft = {
  code: "quantum_espresso",
  task: "scf_single_point",
  hpc: null,
  overrides: {},
  fetchMissing: false,
};

// Field names confirmed against a real /explain response for Si.cif
// (2026-09-14): {mesh, shift, k_distance, warnings} -- not {grid,
// mesh_type} as an earlier draft of this fixture guessed.
const kSamplingRecord: ResolvedField<{
  mesh: readonly number[];
  shift: readonly number[];
  k_distance: number;
  warnings: unknown[];
}> = {
  status: "resolved",
  value: {
    mesh: [3, 3, 3],
    shift: [0, 0, 0],
    k_distance: 0.15,
    warnings: [],
  },
  source: "heuristic",
};

const cutoffsRecord: ResolvedField<{ ecutwfc_ry: number; ecutrho_ry: number }> =
  {
    status: "resolved",
    value: { ecutwfc_ry: 30, ecutrho_ry: 120 },
    source: "heuristic",
  };

export const scientificRecords: Readonly<Record<string, ResolvedField>> = {
  k_sampling: kSamplingRecord,
  n_irr_k: { status: "resolved", value: 10, source: "heuristic" },
  cutoffs: cutoffsRecord,
  magnetic: {
    status: "resolved",
    value: { spin_polarized: false },
    source: "heuristic",
  },
};

export const explainResult: ExplainResult = {
  records: scientificRecords,
  warnings: [],
};

export const runResult: RunResult = {
  files: ["scf.in", "pseudo/Si.upf", "goldilocks.json"],
  records: scientificRecords,
  warnings: [],
};

const MANIFEST_TEXT = JSON.stringify({
  schema_version: 1,
  records: scientificRecords,
  warnings: [],
  citations: ["van Setten et al."],
  files: {
    "scf.in": { role: "input", sha256: "c".repeat(64), size_bytes: 11 },
    "pseudo/Si.upf": {
      role: "pseudopotential",
      sha256: "d".repeat(64),
      size_bytes: 128,
    },
  },
});

/** jsdom's test realm has its own `Uint8Array` distinct from the one
 * `TextEncoder` (used by fflate's `strToU8`) produces -- `instanceof
 * Uint8Array` then fails inside fflate's own type check, so it treats
 * the bytes as a *nested folder object* instead (a real `Uint8Array`
 * exposes its elements as own enumerable "0","1",... keys) and recurses
 * into each byte as if it were a subdirectory. Re-homing through `new
 * Uint8Array(...)` in the caller's realm avoids that -- real browsers
 * never hit this, `TextEncoder` and `Uint8Array` always share one realm
 * there. */
function u8(text: string): Uint8Array {
  return new Uint8Array(strToU8(text));
}

/** A real zip, matching bundle.py's `bundle_files` shape closely enough
 * for GeneratedInputReview's client-side unzip to exercise for real. */
export function buildArchive(
  filename = "goldilocks-inputs.zip",
): ArchiveDownload {
  const zipped = zipSync({
    "scf.in": u8("&CONTROL\n/\n"),
    "pseudo/Si.upf": u8("UPF content"),
    "goldilocks.json": u8(MANIFEST_TEXT),
  });
  // Blob(...) copies exactly zipped.byteLength bytes; `zipped.buffer` is
  // the wrong thing to pass instead since a Uint8Array's underlying
  // ArrayBuffer can be larger than (or offset within) the view itself.
  // Re-wrapping through `new Uint8Array(zipped)` also gives a plain
  // `Uint8Array<ArrayBuffer>` (never `SharedArrayBuffer`), which is what
  // `BlobPart` requires.
  return {
    blob: new Blob([new Uint8Array(zipped)], { type: "application/zip" }),
    filename,
  };
}
