import { afterEach, describe, expect, it, vi } from "vitest";
import {
  checkForUpdates,
  fetchConnectUrl,
  getSettings,
  installUpdate,
} from "./api";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("fetchConnectUrl", () => {
  it("requests a desktop handoff after account creation", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            signin_url:
              "https://ollama.com/connect?name=MacBook&key=public-key",
          }),
          { status: 401 },
        ),
      ),
    );

    await expect(fetchConnectUrl()).resolves.toBe(
      "https://ollama.com/connect?name=MacBook&key=public-key&launch=true",
    );
  });
});

describe("desktop update API", () => {
  it("parses the manual-only settings policy", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            settings: {},
            manualUpdatesOnly: true,
            updateReady: true,
            updateVersion: "v0.32.5",
          }),
          { status: 200 },
        ),
      ),
    );

    await expect(getSettings()).resolves.toMatchObject({
      manualUpdatesOnly: true,
      updateReady: true,
      updateVersion: "v0.32.5",
    });
  });

  it("requests installation of a staged update", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response('{"status":"cancelled"}', { status: 200 }),
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(installUpdate()).resolves.toEqual({ status: "cancelled" });
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/v1/update/install"),
      { method: "POST" },
    );
  });

  it.each([
    { status: "up_to_date" as const },
    { status: "ready" as const, version: "v0.32.5" },
  ])("parses $status update results", async (result) => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(JSON.stringify(result), { status: 200 }),
        ),
    );

    await expect(checkForUpdates()).resolves.toEqual(result);
  });

  it("surfaces a useful backend error", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(
            JSON.stringify({ error: "signature verification failed" }),
            { status: 500 },
          ),
        ),
    );

    await expect(checkForUpdates()).rejects.toThrow(
      "signature verification failed",
    );
  });
});
