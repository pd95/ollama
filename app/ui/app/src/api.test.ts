import { afterEach, describe, expect, it, vi } from "vitest";
const { listModels, showModel } = vi.hoisted(() => ({
  listModels: vi.fn(),
  showModel: vi.fn(),
}));
vi.mock("./lib/ollama-client", () => ({
  ollamaClient: { list: listModels, show: showModel },
}));

import {
  fetchConnectUrl,
  getModelCapabilities,
  getModels,
  sendMessage,
  getClaudeDesktopAvailableModels,
  getClaudeDesktopModelsSettings,
  getCodexDesktopModelsSettings,
  getIntegrationStatuses,
} from "./api";

describe("desktop model settings", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("requests summaries and catalogs through the configured API server", async () => {
    const response = { settings: { selected: ["saved-model"] } };
    const fetch = vi
      .fn()
      .mockResolvedValue(new Response(JSON.stringify(response)));
    vi.stubGlobal("fetch", fetch);
    await expect(getCodexDesktopModelsSettings(false)).resolves.toEqual(
      response,
    );
    expect(fetch).toHaveBeenLastCalledWith(
      "http://127.0.0.1:3001/api/v1/integrations/chatgpt/models?catalog=false",
      { signal: undefined },
    );
    fetch.mockResolvedValue(new Response(JSON.stringify({ installed: true })));
    const controller = new AbortController();
    await expect(
      getClaudeDesktopModelsSettings(true, controller.signal),
    ).resolves.toEqual({ installed: true });
    expect(fetch).toHaveBeenLastCalledWith(
      "http://127.0.0.1:3001/api/v1/integrations/claude-desktop/models?catalog=true",
      { signal: controller.signal },
    );
  });

  it("rejects failed discovery responses", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("timed out", { status: 504 })),
    );
    await expect(getCodexDesktopModelsSettings(true)).rejects.toThrow("504");
  });
});

describe("picker capability discovery", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("preserves model size and runtime details from list and show metadata", async () => {
    const details = {
      format: "gguf",
      runner: "ggml",
      parameter_size: "8B",
      quantization_level: "Q4_K_M",
      context_length: 32768,
    };
    listModels.mockResolvedValue({
      models: [
        { name: "my-alias", digest: "one", size: 4_900_000_000, details },
      ],
    });
    const models = await getModels();
    expect(models[0].size).toBe(4_900_000_000);
    expect(models[0].metadata).toMatchObject({
      format: "gguf",
      runner: "ggml",
      parameterSize: "8B",
      quantization: "Q4_K_M",
      contextLength: 32768,
    });
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(
            JSON.stringify({
              details: { ...details, context_length: undefined },
              model_info: {
                "general.architecture": "llama",
                "llama.context_length": 65536,
              },
            }),
          ),
        ),
    );
    expect((await getModelCapabilities("my-alias")).metadata).toMatchObject({
      format: "gguf",
      runner: "ggml",
      contextLength: 65536,
    });
  });

  it("preserves exact-tag capability metadata from the model list", async () => {
    listModels.mockResolvedValue({
      models: [
        {
          name: "my-alias:latest",
          digest: "one",
          capabilities: ["completion", "vision", "audio"],
          details: {},
        },
        {
          name: "my-alias:text",
          digest: "two",
          capabilities: ["completion"],
          details: {},
        },
        { name: "unknown:latest", digest: "three", details: {} },
      ],
    });
    const models = await getModels();
    expect(models.map((model) => model.capabilities)).toEqual([
      ["completion", "vision", "audio"],
      ["completion"],
      undefined,
    ]);
    expect(models.map((model) => model.model)).toEqual([
      "my-alias",
      "my-alias:text",
      "unknown",
    ]);
  });

  it("rejects failed discovery rather than claiming no capabilities", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(new Response("not downloaded", { status: 404 })),
    );
    vi.spyOn(console, "error").mockImplementation(() => {});
    await expect(getModelCapabilities("unknown")).rejects.toThrow();
    vi.restoreAllMocks();
  });
});

describe("fetchConnectUrl", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

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

describe("model thinking discovery and transport", () => {
  afterEach(() => {
    showModel.mockReset();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("preserves backend thinking values for a custom model name", async () => {
    const thinking = { values: [false, "low", "xhigh"], default: "xhigh" };
    const fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          capabilities: ["thinking"],
          thinking,
          renderer: "apertus1p5",
        }),
      ),
    );
    vi.stubGlobal("fetch", fetch);
    const metadata = await getModelCapabilities("my-alias");
    expect(metadata.thinking).toEqual(thinking);
    expect(metadata.renderer).toBe("apertus1p5");
    expect(fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:3001/api/show",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ model: "my-alias" }),
        signal: expect.any(AbortSignal),
      }),
    );
  });

  it("does not invent metadata when discovery fails", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(new Response("not downloaded", { status: 404 })),
    );
    await expect(getModelCapabilities("unknown")).rejects.toThrow("404");
  });

  it.each([false, true, "xhigh", "none", undefined])(
    "preserves %s in the desktop chat request",
    async (think) => {
      const fetch = vi.fn().mockResolvedValue(new Response(""));
      vi.stubGlobal("fetch", fetch);
      const stream = sendMessage(
        "new",
        "Hello",
        { model: "my-alias" } as never,
        undefined,
        undefined,
        undefined,
        false,
        false,
        false,
        think,
      );
      for await (const _event of stream) {
        /* consume the request */
      }
      const body = JSON.parse(fetch.mock.calls[0][1].body);
      expect(body.think).toBe(think);
      expect(Object.hasOwn(body, "think")).toBe(think !== undefined);
    },
  );
});
