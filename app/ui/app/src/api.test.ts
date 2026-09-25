import { afterEach, describe, expect, it, vi } from "vitest";
const { listModels } = vi.hoisted(() => ({ listModels: vi.fn() }));
vi.mock("./lib/ollama-client", () => ({
  ollamaClient: { list: listModels },
}));

import {
  fetchConnectUrl,
  getClaudeDesktopAvailableModels,
  getIntegrationStatuses,
  checkForUpdates,
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

describe("getIntegrationStatuses", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns desktop and launcher integration metadata", async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify([
          {
            id: "claude-desktop",
            name: "Claude",
            description: "Use Ollama models in Claude Desktop",
            installed: true,
          },
          {
            id: "opencode",
            name: "OpenCode",
            description: "Open-source coding agent",
            command: "ollama launch opencode",
          },
        ]),
        { status: 200 },
      ),
    );
    vi.stubGlobal("fetch", fetch);

    await expect(getIntegrationStatuses()).resolves.toEqual([
      {
        id: "claude-desktop",
        name: "Claude",
        description: "Use Ollama models in Claude Desktop",
        installed: true,
      },
      {
        id: "opencode",
        name: "OpenCode",
        description: "Open-source coding agent",
        command: "ollama launch opencode",
      },
    ]);
    expect(fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:3001/api/v1/integrations",
    );
  });
});

describe("getClaudeDesktopAvailableModels", () => {
  afterEach(() => {
    listModels.mockReset();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("returns installed local models while pruning remote entries", async () => {
    listModels.mockResolvedValue({
      models: [
        { name: "llama3.2:latest", digest: "local" },
        {
          name: "remote-placeholder",
          digest: "remote",
          remote_host: "https://ollama.com",
        },
      ],
    });
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);

    const models = await getClaudeDesktopAvailableModels();

    expect(models.map((model) => model.model)).toEqual(["llama3.2"]);
    expect(fetch).not.toHaveBeenCalled();
  });

  it("does not request cloud models when they are unavailable to the user", async () => {
    listModels.mockResolvedValue({
      models: [
        { name: "qwen3:8b", digest: "local" },
        { name: "deepseek-v4-flash:cloud", digest: "cached-cloud" },
        { name: "gemma4:31b-cloud", digest: "legacy-cached-cloud" },
      ],
    });
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);

    const models = await getClaudeDesktopAvailableModels();

    expect(models.map((model) => model.model)).toEqual(["qwen3:8b"]);
    expect(fetch).not.toHaveBeenCalled();
  });

  it("loads the account cloud list in parallel when Cloud is available", async () => {
    listModels.mockResolvedValue({
      models: [{ name: "qwen3:8b", digest: "local" }],
    });
    const fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          models: [
            { name: "glm-5.2", digest: "cloud" },
            { name: "gemma4:31b-cloud", digest: "legacy-cloud" },
            { name: "qwen3:8b", digest: "cloud-duplicate" },
          ],
        }),
      ),
    );
    vi.stubGlobal("fetch", fetch);

    const models = await getClaudeDesktopAvailableModels(true);

    expect(models.map((model) => model.model)).toEqual([
      "qwen3:8b",
      "glm-5.2:cloud",
      "gemma4:31b-cloud",
    ]);
    expect(fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:3001/api/v1/models/cloud",
    );
  });

  it("keeps local models when the account cloud list fails", async () => {
    listModels.mockResolvedValue({
      models: [{ name: "qwen3:8b", digest: "local" }],
    });
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));

    const models = await getClaudeDesktopAvailableModels(true);

    expect(models.map((model) => model.model)).toEqual(["qwen3:8b"]);
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
            updateBuildVersion: "32.5.1",
            updateSource: "mlx-preview",
            updateChannel: "preview",
            updateReleaseUrl: "https://github.com/pd95/ollama/releases",
          }),
          { status: 200 },
        ),
      ),
    );

    await expect(getSettings()).resolves.toMatchObject({
      manualUpdatesOnly: true,
      updateReady: true,
      updateVersion: "v0.32.5",
      updateBuildVersion: "32.5.1",
      updateSource: "mlx-preview",
      updateChannel: "preview",
      updateReleaseUrl: "https://github.com/pd95/ollama/releases",
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

  it("qualifies update checks and installation by source", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response('{"status":"up_to_date","source":"official"}', {
          status: 200,
        }),
      )
      .mockResolvedValueOnce(
        new Response('{"status":"started"}', { status: 200 }),
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(checkForUpdates("official")).resolves.toMatchObject({
      status: "up_to_date",
      source: "official",
    });
    await expect(installUpdate("official")).resolves.toEqual({
      status: "started",
    });
    for (const [, request] of fetchMock.mock.calls) {
      expect(request).toMatchObject({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: "official" }),
      });
    }
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
