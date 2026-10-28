import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import { expect, it, vi } from "vitest";

it.each([
  undefined,
  { images: true },
  { audio: true },
  { images: true, audio: true },
])(
  "forwards native media permissions from the HTML bridge: %j",
  async (options) => {
    const window: any = { selectFiles: vi.fn(), addEventListener: vi.fn() };
    const html = readFileSync(
      new URL("../../index.html", import.meta.url),
      "utf8",
    );
    const script = html.match(/<script>([\s\S]*?)<\/script>/)![1];
    runInNewContext(script, {
      window,
      document: { addEventListener: vi.fn() },
    });
    const promise = window.webview.selectMultipleFiles(options);
    expect(window.selectFiles).toHaveBeenCalledWith(
      options?.images === true,
      options?.audio === true,
    );
    window.__selectFilesCallback([]);
    await expect(promise).resolves.toEqual([]);
  },
);
