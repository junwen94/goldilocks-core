import { describe, expect, it, vi } from "vitest";

import {
  CoreFailure,
  HttpCoreClient,
  type ComputeRequest,
} from "../../src/api/coreClient";
import {
  capabilities,
  explainResult,
  inspection,
  runResult,
  structureInput,
} from "../support/workbenchFixtures";

const request: ComputeRequest = {
  structure_content: structureInput.structure_content,
  structure_name: structureInput.structure_name,
  structure_format: "cif",
  code: "quantum_espresso",
  task: "scf_single_point",
  hpc: null,
  overrides: {},
  fetch_missing: false,
};

describe("HttpCoreClient", () => {
  describe("capabilities", () => {
    it("loads Capabilities as generated Core types", async () => {
      const fetcher = vi
        .fn<typeof fetch>()
        .mockResolvedValue(Response.json(capabilities));
      const client = new HttpCoreClient("/core", fetcher);

      await expect(client.capabilities()).resolves.toEqual(capabilities);
      expect(fetcher).toHaveBeenCalledWith("/core/capabilities", {
        headers: { Accept: "application/json" },
        method: "GET",
      });
    });

    it("rejects a vocabulary_version the frontend wasn't built against", async () => {
      const fetcher = vi
        .fn<typeof fetch>()
        .mockResolvedValue(
          Response.json({ ...capabilities, vocabulary_version: "2" }),
        );
      const client = new HttpCoreClient("", fetcher);

      await expect(client.capabilities()).rejects.toMatchObject({
        kind: "incompatible_vocabulary",
        retryable: false,
      });
    });

    it("rejects a response with the wrong content type", async () => {
      const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
        new Response(JSON.stringify(capabilities), {
          headers: { "Content-Type": "text/plain" },
        }),
      );
      const client = new HttpCoreClient("", fetcher);

      await expect(client.capabilities()).rejects.toMatchObject({
        kind: "invalid_response",
        message: "Goldilocks Core returned an invalid JSON response.",
        retryable: false,
      });
    });

    it("reports a retryable network failure", async () => {
      const fetcher = vi
        .fn<typeof fetch>()
        .mockRejectedValue(new TypeError("connection refused"));
      const client = new HttpCoreClient("", fetcher);

      await expect(client.capabilities()).rejects.toMatchObject({
        name: "CoreFailure",
        kind: "network_error",
        message: "Cannot reach Goldilocks Core: connection refused",
        retryable: true,
        status: null,
      });
    });

    it("converts a Core error envelope into one typed failure, deriving retryable from status", async () => {
      const payload = {
        error: {
          kind: "advice_incomplete",
          message:
            "cannot generate a runnable input: pseudopotential unavailable",
          details: { attempt: 2 },
        },
      };
      const fetcher = vi
        .fn<typeof fetch>()
        .mockResolvedValue(Response.json(payload, { status: 424 }));
      const client = new HttpCoreClient("", fetcher);

      const failure = await client
        .capabilities()
        .catch((error: unknown) => error);

      expect(failure).toBeInstanceOf(CoreFailure);
      expect(failure).toMatchObject({
        kind: "advice_incomplete",
        message: payload.error.message,
        retryable: false,
        details: { attempt: 2 },
        status: 424,
        rawResponse: payload,
      });
    });
  });

  describe("inspectStructure", () => {
    it("posts the flattened structure body and requires a schema version", async () => {
      const fetcher = vi
        .fn<typeof fetch>()
        .mockResolvedValue(Response.json(inspection));
      const client = new HttpCoreClient("", fetcher);

      await expect(client.inspectStructure(structureInput)).resolves.toEqual(
        inspection,
      );
      expect(fetcher).toHaveBeenCalledWith("/inspect", {
        body: JSON.stringify(structureInput),
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        method: "POST",
      });
    });

    it("rejects a response missing schema_version", async () => {
      const withoutVersion: Record<string, unknown> = { ...inspection };
      delete withoutVersion.schema_version;
      const fetcher = vi
        .fn<typeof fetch>()
        .mockResolvedValue(Response.json(withoutVersion));
      const client = new HttpCoreClient("", fetcher);

      await expect(
        client.inspectStructure(structureInput),
      ).rejects.toMatchObject({
        kind: "invalid_response",
        message: "Goldilocks Core returned an incompatible schema version.",
      });
    });
  });

  describe("explain", () => {
    it("posts to /explain and returns records + warnings, no schema_version required", async () => {
      const fetcher = vi
        .fn<typeof fetch>()
        .mockResolvedValue(Response.json(explainResult));
      const client = new HttpCoreClient("", fetcher);

      await expect(client.explain(request)).resolves.toEqual(explainResult);
      expect(fetcher).toHaveBeenCalledWith("/explain", {
        body: JSON.stringify(request),
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        method: "POST",
      });
    });

    it("surfaces advice_incomplete without ever having called /run", async () => {
      const fetcher = vi
        .fn<typeof fetch>()
        .mockResolvedValue(
          Response.json(
            { error: { kind: "advice_incomplete", message: "incomplete" } },
            { status: 424 },
          ),
        );
      const client = new HttpCoreClient("", fetcher);

      await expect(client.explain(request)).rejects.toMatchObject({
        kind: "advice_incomplete",
      });
      expect(fetcher).toHaveBeenCalledWith("/explain", expect.anything());
    });
  });

  describe("run", () => {
    it("posts to /run with respond_with: json and returns the file list", async () => {
      const fetcher = vi
        .fn<typeof fetch>()
        .mockResolvedValue(Response.json(runResult));
      const client = new HttpCoreClient("", fetcher);

      await expect(client.run(request)).resolves.toEqual(runResult);
      expect(fetcher).toHaveBeenCalledWith("/run", {
        body: JSON.stringify({ ...request, respond_with: "json" }),
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        method: "POST",
      });
    });
  });

  describe("runArchive", () => {
    it("posts to /run with respond_with: archive and returns a synthesized filename", async () => {
      const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
        new Response("zip bytes", {
          headers: { "Content-Type": "application/zip" },
        }),
      );
      const client = new HttpCoreClient("", fetcher);

      const archive = await client.runArchive(request);

      expect(archive.filename).toBe("Si.cif-scf_single_point.zip");
      await expect(archive.blob.text()).resolves.toBe("zip bytes");
      expect(fetcher).toHaveBeenCalledWith("/run", {
        body: JSON.stringify({ ...request, respond_with: "archive" }),
        headers: {
          Accept: "application/zip",
          "Content-Type": "application/json",
        },
        method: "POST",
      });
    });

    it("rejects an archive response with the wrong content type", async () => {
      const fetcher = vi
        .fn<typeof fetch>()
        .mockResolvedValue(
          Response.json({ files: [], records: {}, warnings: [] }),
        );
      const client = new HttpCoreClient("", fetcher);

      await expect(client.runArchive(request)).rejects.toMatchObject({
        kind: "invalid_response",
        message: "Goldilocks Core returned an invalid archive response.",
      });
    });
  });
});
