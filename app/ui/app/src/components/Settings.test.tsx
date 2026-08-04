import { renderToStaticMarkup } from "react-dom/server";
import type React from "react";
import { describe, expect, it, vi } from "vitest";

import { UpdateSettingsControl } from "./Settings";

function render(
  props: Partial<React.ComponentProps<typeof UpdateSettingsControl>> = {},
) {
  return renderToStaticMarkup(
    <UpdateSettingsControl
      manualUpdatesOnly
      autoUpdateEnabled
      isPending={false}
      isInstalling={false}
      onCheck={vi.fn()}
      onInstall={vi.fn()}
      onToggle={vi.fn()}
      {...props}
    />,
  );
}

describe("UpdateSettingsControl", () => {
  it("renders the manual preview policy without an auto-update switch", () => {
    const html = render();

    expect(html).toContain("Official Ollama updates");
    expect(html).toContain(
      "This MLX preview does not check for or download official updates automatically.",
    );
    expect(html).not.toContain('role="switch"');
  });

  it("renders pending, current, ready, and error outcomes", () => {
    expect(render({ isPending: true })).toContain(
      "Ollama is checking and may be downloading the official release.",
    );
    expect(render({ result: { status: "up_to_date" } })).toContain(
      "No newer official Ollama release is available.",
    );

    const ready = render({ result: { status: "ready", version: "v0.32.5" } });
    expect(ready).toContain("v0.32.5 is downloaded and ready.");
    expect(ready).toContain("Update now");
    expect(ready).toContain(
      "Installing this update replaces the MLX preview with official Ollama. Some preview-only features will no longer be available.",
    );
    expect(
      render({ error: new Error("signature verification failed") }),
    ).toContain("signature verification failed");
  });

  it("retains the official-build auto-update switch", () => {
    const html = render({
      manualUpdatesOnly: false,
      result: { status: "ready", version: "v0.32.5" },
    });

    expect(html).toContain("Auto-download updates");
    expect(html).toContain('role="switch"');
    expect(html).toContain("Check now");
    expect(html).not.toContain("Update now");
  });
});
