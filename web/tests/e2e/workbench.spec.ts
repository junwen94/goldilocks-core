import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { posix } from "node:path";
import AxeBuilder from "@axe-core/playwright";
import { strFromU8, unzipSync } from "fflate";

import { expect, test, type Page } from "@playwright/test";
import { fileURLToPath } from "node:url";
import type { InputManifest } from "../../src/review/artifacts";

const SILICON_CIF = fileURLToPath(
  new URL(
    "../../../src/goldilocks_core/examples/structures/Si.cif",
    import.meta.url,
  ),
);
const SILICON_POSCAR = `Silicon
5.431
1 0 0
0 1 0
0 0 1
Si
1
Direct
0 0 0
`;

test("serves concurrent explains through one real Core runtime", async ({
  request,
}) => {
  const structure = await readFile(SILICON_CIF, "utf8");
  const body = {
    structure_content: structure,
    structure_name: "Si.cif",
    structure_format: "cif",
  };

  const responses = await Promise.all(
    Array.from({ length: 8 }, () => request.post("/explain", { data: body })),
  );
  const payloads = await Promise.all(
    responses.map(async (response) => {
      expect(response.status()).toBe(200);
      expect(response.headers()["content-type"]).toContain("application/json");
      return response.json() as Promise<{
        records: { k_sampling?: { value?: { mesh?: number[] } } };
      }>;
    }),
  );
  const meshes = payloads.map((payload) =>
    payload.records.k_sampling?.value?.mesh?.join(","),
  );
  expect(meshes.every((mesh) => mesh !== undefined)).toBe(true);
  expect(new Set(meshes).size).toBe(1);
});

test("prepares and downloads a real Core calculation", async ({ page }) => {
  const runRequests: {
    readonly method: string;
    readonly body: string | null;
  }[] = [];
  page.on("request", (request) => {
    if (new URL(request.url()).pathname === "/run") {
      runRequests.push({ method: request.method(), body: request.postData() });
    }
  });
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "No structure selected" }),
  ).toBeVisible();

  await page.locator('input[type="file"]').setInputFiles(SILICON_CIF);
  await expect(page.getByLabel("Crystal structure viewer")).toBeVisible();
  await expect(page.getByText("8 atomic sites", { exact: true })).toBeVisible();
  await page
    .getByLabel("Pseudopotential table")
    .selectOption("pseudodojo-pbesol-efficiency-sr");

  await page.getByRole("button", { name: "Generate recommendation" }).click();
  const recommendation = page.getByRole("region", {
    name: "Recommendation results",
  });
  await expect(recommendation).toBeVisible();
  await expect(
    page.getByText(/Generate input files to preview them here/),
  ).toBeVisible();

  const downloadStarted = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Generate input files (.zip)" })
    .click();
  const download = await downloadStarted;
  expect(download.suggestedFilename()).toMatch(/\.zip$/);

  // scf.in is the default active tab (review order: .in, submit.sh,
  // README.md, .json, then everything else) -- only the active tab's
  // panel renders a GeneratedInputPreview.
  const generatedInput = page.getByLabel("Generated input scf.in");
  await expect(generatedInput).toBeInViewport();

  const inputResize = page.getByRole("separator", {
    name: "Resize generated input",
  });
  const initialInput = await generatedInput.boundingBox();
  const resizeBox = await inputResize.boundingBox();
  assert(initialInput, "Generated input must have a layout box");
  assert(resizeBox, "Generated input resize handle must have a layout box");
  const resizeX = resizeBox.x + resizeBox.width / 2;
  const resizeY = resizeBox.y + resizeBox.height / 2;
  await page.mouse.move(resizeX, resizeY);
  await page.mouse.down();
  await page.mouse.move(resizeX, resizeY + 64, { steps: 4 });
  await page.mouse.up();
  const resizedInput = await generatedInput.boundingBox();
  assert(resizedInput, "Resizing must retain the generated input");
  expect(resizedInput.height - initialInput.height).toBeGreaterThan(40);

  await inputResize.press("End");
  await expect(inputResize).toHaveAttribute(
    "aria-valuetext",
    "Full input file visible",
  );
  await expectNoAxeViolations(page);

  await page.getByRole("button", { name: "Back to structure" }).click();
  await expect(page.getByLabel("Crystal structure viewer")).toBeVisible();
  await expect(recommendation).toHaveCount(0);
  await page
    .getByRole("button", { name: "Recommendation", exact: true })
    .click();

  await page.getByRole("button", { name: "Switch to dark mode" }).click();
  await expectNoAxeViolations(page);

  const path = await download.path();
  assert(path, "Download must have a local path");
  const entries = unzipSync(new Uint8Array(await readFile(path)));
  verifyArchive(entries, {
    tableId: "pseudodojo-pbesol-efficiency-sr",
    tableVersion: "0.4",
  });
  expect(runRequests).toHaveLength(1);
  expect(runRequests[0]?.method).toBe("POST");
  const runBody = JSON.parse(runRequests[0]?.body ?? "{}") as {
    overrides?: Record<string, unknown>;
  };
  expect(runBody.overrides?.pseudo_table_id).toBe(
    "pseudodojo-pbesol-efficiency-sr",
  );
});

test("opens and closes scientific details with the keyboard", async ({
  page,
}) => {
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles(SILICON_CIF);
  await page.getByRole("button", { name: "Generate recommendation" }).click();
  await expect(
    page.getByRole("region", { name: "Recommendation results" }),
  ).toBeVisible();

  const sampling = page.getByRole("button", { name: /^K Sampling/ });
  await expect(sampling).toHaveAttribute("aria-expanded", "false");
  await sampling.press("Enter");
  await expect(sampling).toHaveAttribute("aria-expanded", "true");
  await expectNoAxeViolations(page);
  await sampling.press("Enter");
  await expect(sampling).toHaveAttribute("aria-expanded", "false");
});

test("applies a paired smearing treatment and width override", async ({
  page,
}) => {
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles(SILICON_CIF);
  await page.getByRole("button", { name: "Occupations" }).click();
  await page.getByLabel("smearing type").selectOption("cold");
  await page.getByLabel("degauss · Ry").fill("0.02");

  await page.getByRole("button", { name: "Generate recommendation" }).click();
  await expect(
    page.getByRole("region", { name: "Recommendation results" }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Generate input files (.zip)" })
    .click();
  await page.getByRole("tab", { name: "scf.in" }).click();

  const input = page.getByLabel("Generated input scf.in");
  await expect(input).toContainText("smearing = 'cold'");
  await expect(input).toContainText("degauss = 0.02");
});

test("keeps an old Result visible until an edited Draft is recomputed", async ({
  page,
}) => {
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles(SILICON_CIF);
  await page.getByRole("button", { name: "Generate recommendation" }).click();
  const recommendation = page.getByRole("region", {
    name: "Recommendation results",
  });
  await expect(recommendation).toBeVisible();

  await page.getByRole("button", { name: "K sampling", exact: true }).click();
  await page.getByRole("checkbox", { name: "Set an explicit grid" }).check();

  await expect(
    page.getByRole("status", { name: "Recommendation notice" }),
  ).toContainText(
    "Your settings changed. Update the recommendation before generating input files.",
  );
  await expect(recommendation).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Generate input files (.zip)" }),
  ).toBeDisabled();
  await expectNoAxeViolations(page);

  await page.getByRole("button", { name: "Update recommendation" }).click();
  await expect(
    page.getByRole("status", { name: "Recommendation notice" }),
  ).toBeHidden();
  await page
    .getByRole("button", { name: "Generate input files (.zip)" })
    .click();
  await page.getByRole("tab", { name: "scf.in" }).click();
  await expect(page.getByLabel("Generated input scf.in")).toContainText(
    "1 1 1",
  );
});

test("has no Axe violations in empty, failure, and viewer fallback states", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "No structure selected" }),
  ).toBeVisible();
  await expectNoAxeViolations(page);

  const themeToggle = page.getByRole("button", {
    name: "Switch to dark mode",
  });
  await themeToggle.click();
  const lightMode = page.getByRole("button", { name: "Switch to light mode" });
  await expect(lightMode).toBeVisible();
  await expectNoAxeViolations(page);
  await lightMode.click();

  await page.locator('input[type="file"]').setInputFiles(SILICON_CIF);
  await page.route(
    "**/explain",
    async (route) => {
      await route.fulfill({
        status: 424,
        contentType: "application/json",
        body: JSON.stringify({
          error: {
            kind: "advice_incomplete",
            message: "Core failed temporarily.",
          },
        }),
      });
    },
    { times: 1 },
  );
  await page.getByRole("button", { name: "Generate recommendation" }).click();
  const alert = page.getByRole("alert");
  await expect(alert).toContainText("Recommendation incomplete");
  await expect(alert).toContainText("Core failed temporarily.");
  const status = page.getByRole("status", { name: "Workbench status" });
  await expect(status).toHaveText("Needs attention");
  await expect(status).not.toContainText("Ready");
  await expect(status).not.toContainText("Another calculation");
  await expectNoAxeViolations(page);

  await page.addInitScript({
    content: `
      for (const prototype of [
        HTMLCanvasElement.prototype,
        globalThis.OffscreenCanvas?.prototype,
      ]) {
        if (!prototype) continue;
        const originalGetContext = prototype.getContext;
        prototype.getContext = function (type, ...args) {
          if (String(type).includes("webgl")) return null;
          return originalGetContext.call(this, type, ...args);
        };
      }
    `,
  });
  await page.reload();
  await page.locator('input[type="file"]').setInputFiles(SILICON_CIF);
  await expect(
    page.getByRole("status", { name: "3D structure preview unavailable" }),
  ).toBeVisible();
  await expectNoAxeViolations(page);
});

test("completes the preparation workflow with keyboard-only activation", async ({
  page,
}) => {
  await page.goto("/");
  const browse = page.getByRole("button", {
    name: "Choose a CIF or POSCAR structure",
  });
  await browse.waitFor();
  await page.keyboard.press("Tab");
  await page.keyboard.press("Tab");
  await expect(browse).toBeFocused();
  const chooserPromise = page.waitForEvent("filechooser");
  await page.keyboard.press("Enter");
  const chooser = await chooserPromise;
  await chooser.setFiles(SILICON_CIF);
  await expect(page.getByLabel("Crystal structure viewer")).toBeVisible();

  const generate = page.getByRole("button", {
    name: "Generate recommendation",
  });
  await generate.press("Enter");
  await expect(
    page.getByRole("region", { name: "Recommendation results" }),
  ).toBeVisible();

  const kSampling = page.getByRole("button", {
    name: "K sampling",
    exact: true,
  });
  await kSampling.press("Enter");
  await expect(kSampling).toHaveAttribute("aria-expanded", "true");
  const explicitGrid = page.getByRole("checkbox", {
    name: "Set an explicit grid",
  });
  await explicitGrid.press("Space");
  await expect(explicitGrid).toBeChecked();
  await expect(
    page.getByRole("status", { name: "Recommendation notice" }),
  ).toContainText(
    "Your settings changed. Update the recommendation before generating input files.",
  );

  const firstRecord = page
    .locator(".record-card")
    .first()
    .getByRole("button")
    .first();
  await firstRecord.press("Enter");
  await expect(firstRecord).toHaveAttribute("aria-expanded", "true");
});

test("keeps keyboard focus visible and primary targets usable", async ({
  page,
}) => {
  await page.goto("/");
  const browse = page.getByRole("button", {
    name: "Choose a CIF or POSCAR structure",
  });
  await browse.waitFor();

  await page.keyboard.press("Tab");
  await expect(
    page.getByRole("button", { name: "Switch to dark mode" }),
  ).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(browse).toBeFocused();
  const box = await browse.boundingBox();
  expect(box).not.toBeNull();
  expect(box?.height).toBeGreaterThanOrEqual(44);
  expect(box?.width).toBeGreaterThanOrEqual(44);
  await expect(browse).toHaveCSS("outline-style", "solid");
  await expect(browse).toHaveCSS("outline-width", "3px");
});

test("resizes the two-panel layout with pointer and keyboard input", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1600, height: 900 });
  await page.goto("/");

  const controls = page.getByRole("region", { name: "Calculation setup" });
  const structure = page.getByRole("region", { name: "Structure workspace" });
  const controlsHandle = page.getByRole("separator", {
    name: "Resize calculation setup",
  });
  const initialControls = await controls.boundingBox();
  const initialStructure = await structure.boundingBox();
  const handle = await controlsHandle.boundingBox();
  assert(initialControls, "Calculation panel must have a layout box");
  assert(initialStructure, "Structure panel must have a layout box");
  assert(handle, "Panel resize handle must have a layout box");
  await page.mouse.move(handle.x + handle.width / 2, handle.y + 200);
  await page.mouse.down();
  await page.mouse.move(handle.x + 120, handle.y + 200);
  await page.mouse.up();

  const resizedControls = await controls.boundingBox();
  const resizedStructure = await structure.boundingBox();
  assert(resizedControls, "Resizing must retain the calculation panel");
  assert(resizedStructure, "Resizing must retain the structure panel");
  expect(resizedControls.width - initialControls.width).toBeGreaterThan(80);
  expect(initialStructure.width - resizedStructure.width).toBeGreaterThan(80);

  expect(await page.getByRole("separator").count()).toBe(1);
  await controlsHandle.press("Home");
  await expect(controlsHandle).toHaveAttribute("aria-valuenow", "24");
});

test("constrains resized desktop panes without clipping", async ({ page }) => {
  await page.setViewportSize({ width: 920, height: 700 });
  await page.goto("/");
  const resize = page.getByRole("separator", {
    name: "Resize calculation setup",
  });
  await resize.press("End");
  await expect(resize).toHaveAttribute("aria-valuenow", "42");

  const dimensions = await page.evaluate<{
    readonly scrollWidth: number;
    readonly viewportWidth: number;
  }>(`({
    scrollWidth: document.documentElement.scrollWidth,
    viewportWidth: window.innerWidth,
  })`);
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.viewportWidth);
});

test("reflows intermediate widths without horizontal clipping", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1050, height: 900 });
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles(SILICON_CIF);
  await page.getByRole("button", { name: "Generate recommendation" }).click();
  const review = page.getByRole("region", { name: "Recommendation results" });
  await expect(review).toBeVisible();

  const workspace = await page.getByRole("main").boundingBox();
  const reviewBox = await review.boundingBox();
  expect(workspace).not.toBeNull();
  expect(reviewBox).not.toBeNull();
  expect((workspace?.x ?? 0) + (workspace?.width ?? 0)).toBeLessThanOrEqual(
    1050,
  );
  expect((reviewBox?.x ?? 0) + (reviewBox?.width ?? 0)).toBeLessThanOrEqual(
    1050,
  );
});

test("uses the document scrollbar for long desktop content", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 700 });
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles(SILICON_CIF);
  await page.getByRole("button", { name: "Generate recommendation" }).click();
  await expect(
    page.getByRole("region", { name: "Recommendation results" }),
  ).toBeVisible();

  const documentScrolls = await page.evaluate<{
    readonly scrollHeight: number;
    readonly viewportHeight: number;
  }>(`({
    scrollHeight: document.documentElement.scrollHeight,
    viewportHeight: window.innerHeight,
  })`);
  expect(documentScrolls.scrollHeight).toBeGreaterThan(
    documentScrolls.viewportHeight,
  );
  await page.evaluate("window.scrollTo(0, document.body.scrollHeight)");
  await expect(page.locator(".record-card").last()).toBeInViewport();
});

test("reflows at effective 200 percent zoom without clipping", async ({
  page,
}) => {
  await page.setViewportSize({ width: 720, height: 500 });
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "No structure selected" }),
  ).toBeVisible();

  const mainBox = await page.getByRole("main").boundingBox();
  expect(mainBox).not.toBeNull();
  expect((mainBox?.x ?? 0) + (mainBox?.width ?? 0)).toBeLessThanOrEqual(720);
});

test("removes nonessential animation when reduced motion is requested", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.route(
    "**/inspect",
    async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 2_000));
      await route.continue();
    },
    { times: 1 },
  );
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles(SILICON_CIF);
  await expect(
    page.getByRole("status", { name: "Workbench status" }),
  ).toHaveText("Inspecting structure");
  const runningAnimations =
    await page.evaluate<number>(`document.getAnimations()
    .filter((animation) => animation.playState === "running").length`);
  expect(runningAnimations).toBe(0);
});

test("prepares a real Core recommendation from POSCAR", async ({ page }) => {
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles({
    name: "POSCAR",
    mimeType: "text/plain",
    buffer: Buffer.from(SILICON_POSCAR),
  });

  await expect(page.getByLabel("Crystal structure viewer")).toBeVisible();
  await expect(page.getByText("1 atomic sites")).toBeVisible();
  await page
    .getByLabel("Pseudopotential table")
    .selectOption("sssp-pbesol-efficiency-sr");
  await page.getByRole("button", { name: "Generate recommendation" }).click();
  await expect(
    page.getByRole("region", { name: "Recommendation results" }),
  ).toBeVisible();
  await expandRecord(page, "Pseudo Table");
  await expect(recordPanel(page, "Pseudo Table")).toContainText(
    "sssp-pbesol-efficiency-sr",
  );

  const downloadStarted = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Generate input files (.zip)" })
    .click();
  const download = await downloadStarted;
  const path = await download.path();
  assert(path, "Download must have a local path");
  const entries = unzipSync(new Uint8Array(await readFile(path)));
  verifyArchive(entries, {
    tableId: "sssp-pbesol-efficiency-sr",
    tableVersion: "1.3.0",
  });
});

test("relax-only overrides stay hidden until the relax task is chosen", async ({
  page,
}) => {
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles(SILICON_CIF);
  await expect(page.getByRole("button", { name: "Relax" })).not.toBeVisible();

  await page.getByLabel("Task").selectOption("relax");
  await expect(page.getByRole("button", { name: "Relax" })).toBeVisible();
});

test("table treatment clears when the functional changes", async ({ page }) => {
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles(SILICON_CIF);
  const table = page.getByLabel("Pseudopotential table");

  await page.getByLabel("Functional").selectOption("PBEsol");
  await table.selectOption("pseudodojo-pbesol-efficiency-fr");
  expect(await table.inputValue()).toBe("pseudodojo-pbesol-efficiency-fr");

  await page.getByLabel("Functional").selectOption("LDA");
  expect(await table.inputValue()).toBe("");
  await expect(
    table.locator('option[value="pseudodojo-pbesol-efficiency-fr"]'),
  ).toHaveCount(0);
});

test("table choices exclude tables that don't cover the structure's elements", async ({
  page,
}) => {
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles(SILICON_CIF);
  await page.getByLabel("Functional").selectOption("PBE");
  const table = page.getByLabel("Pseudopotential table");
  await expect(
    table.locator('option[value="pseudodojo-pbe-lanthanides-sr"]'),
  ).toHaveCount(0);

  await page.locator('input[type="file"]').setInputFiles({
    name: "POSCAR",
    mimeType: "text/plain",
    buffer: Buffer.from(
      "Cerium\n1.0\n5.16 0 0\n0 5.16 0\n0 0 5.16\nCe\n1\nDirect\n0 0 0\n",
    ),
  });
  await expect(page.getByText("Ce1", { exact: true })).toBeVisible();
  await page.getByLabel("Functional").selectOption("PBE");
  // Confirmed against a live /capabilities call: three real tables cover
  // Ce+PBE (pseudodojo's own lanthanides table plus both sssp accuracy
  // levels). The frontend deliberately doesn't replicate the backend's
  // automatic-selection preference for sssp on lanthanides (that
  // heuristic lives in advisors/pseudo_selection.py, not duplicated
  // here) -- picking a table here is an explicit human override, not
  // automatic selection, so every element-eligible table is offered.
  await expect(table.locator("option")).toHaveText([
    "Automatic",
    "pseudodojo · PBE · efficiency · scalar",
    "sssp · PBE · efficiency · scalar",
    "sssp · PBE · precision · scalar",
  ]);
  await table.selectOption("sssp-pbe-efficiency-sr");
  await page.getByRole("button", { name: "Generate recommendation" }).click();
  await expandRecord(page, "Pseudo Table");
  await expect(recordPanel(page, "Pseudo Table")).toContainText(
    "sssp-pbe-efficiency-sr",
  );
});

async function expandRecord(page: Page, name: string): Promise<void> {
  const control = page.getByRole("button", { name });
  if ((await control.getAttribute("aria-expanded")) === "true") return;
  await control.click();
}

function recordPanel(page: Page, name: string) {
  return page
    .getByRole("button", { name })
    .locator("xpath=ancestor::*[contains(@class, 'record-card')]");
}

function verifyArchive(
  entries: Readonly<Record<string, Uint8Array>>,
  expected: {
    readonly tableId: string;
    readonly tableVersion: string;
  },
): void {
  // Confirmed against a real /run call (respond_with: archive, 2026-09-14):
  // bundle_files() only ever emits the generation artifacts themselves
  // (here scf.in + pseudo/<file>), submit.sh, and the three fixed meta
  // files -- v2 never copies the original/canonical structure into the
  // archive the way v1 did (no source/, structure/, or licences/ paths).
  const names = new Set(Object.keys(entries));
  for (const required of [
    "README.md",
    "CITATIONS.md",
    "goldilocks.json",
    "scf.in",
    "submit.sh",
  ]) {
    expect(names).toContain(required);
  }

  const manifest = JSON.parse(
    textEntry(entries, "goldilocks.json"),
  ) as InputManifest & {
    readonly records: {
      readonly pseudo_table?: {
        readonly value?: { readonly id?: string; readonly version?: string };
      };
      readonly pseudopotentials?: {
        readonly value?: readonly { readonly filename: string | null }[];
      };
    };
  };
  expect(manifest.records.pseudo_table?.value?.id).toBe(expected.tableId);
  expect(manifest.records.pseudo_table?.value?.version).toBe(
    expected.tableVersion,
  );

  for (const pseudo of manifest.records.pseudopotentials?.value ?? []) {
    assert(
      pseudo.filename !== null,
      "Published pseudopotential needs a filename",
    );
    const path = `pseudo/${pseudo.filename}`;
    expect(names).toContain(path);
    expect(manifest.files[path]?.role).toBe("pseudopotential");
  }
  expect(new Set(Object.keys(manifest.files))).toEqual(
    new Set([...names].filter((name) => name !== "goldilocks.json")),
  );
  for (const [name, facts] of Object.entries(manifest.files)) {
    const payload = entry(entries, name);
    expect(sha256(payload)).toBe(facts.sha256);
    expect(payload.byteLength).toBe(facts.size_bytes);
  }

  const input = textEntry(entries, "scf.in");
  const pseudoDir = /pseudo_dir\s*=\s*'([^']+)'/.exec(input)?.[1];
  expect(pseudoDir).toBeDefined();
  expect(posix.normalize(pseudoDir ?? "")).toBe("pseudo");
}

function textEntry(
  entries: Readonly<Record<string, Uint8Array>>,
  name: string,
): string {
  return strFromU8(entry(entries, name));
}

function entry(
  entries: Readonly<Record<string, Uint8Array>>,
  name: string,
): Uint8Array {
  const payload = entries[name];
  if (payload === undefined) throw new Error(`archive entry missing: ${name}`);
  return payload;
}

async function expectNoAxeViolations(page: Page): Promise<void> {
  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);
}

function sha256(payload: Uint8Array): string {
  return createHash("sha256").update(payload).digest("hex");
}

test("keeps the crystal title clear of lattice details", async ({ page }) => {
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles(SILICON_CIF);
  const viewer = page.getByRole("region", { name: "Crystal structure viewer" });
  await expect(viewer.locator("canvas")).toBeVisible();
  const title = await viewer
    .getByRole("heading", { name: "Si", exact: true })
    .boundingBox();
  const lattice = await viewer.locator("dl").boundingBox();
  assert(title, "Crystal title must have a layout box");
  assert(lattice, "Lattice details must have a layout box");
  expect(lattice.y).toBeGreaterThan(title.y + title.height);
});
