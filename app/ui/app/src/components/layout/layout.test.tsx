import { act, create } from "react-test-renderer";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SidebarLayout } from "./layout";

const mocks = vi.hoisted(() => ({
  settingsData: { SidebarOpen: true },
  setSettings: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("@/hooks/useSettings", () => ({
  useSettings: () => mocks,
}));

describe("SidebarLayout", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.settingsData.SidebarOpen = true;
    vi.stubGlobal("window", { OLLAMA_PLATFORM: "darwin" });
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  });

  it("restores and persists the sidebar without rendering the wrong state", async () => {
    let renderer;
    await act(async () => {
      renderer = create(
        <SidebarLayout title="Connect your apps" sidebar={<nav />}>
          <div />
        </SidebarLayout>,
      );
      await Promise.resolve();
    });

    const heading = renderer!.root.findByType("h1");
    expect(heading.props.className).toContain("pl-6");
    expect(heading.props.className).toContain("transition-[padding-left]");

    const toggle = renderer!.root.findByProps({ "aria-label": "Hide sidebar" });
    await act(async () => {
      toggle.props.onClick();
      await Promise.resolve();
    });

    expect(mocks.setSettings).toHaveBeenCalledWith({ SidebarOpen: false });
    expect(
      renderer!.root.findByProps({ "aria-label": "Show sidebar" }),
    ).toBeTruthy();
  });
});
