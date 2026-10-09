import { createRef } from "react";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ThinkButton } from "./ThinkButton";

let renderer: ReactTestRenderer;
beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("document", {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  });
});
afterEach(async () => {
  await act(async () => renderer?.unmount());
  vi.unstubAllGlobals();
});

it("offers exactly the backend's mixed values and sends false as a boolean", async () => {
  const onChange = vi.fn();
  await act(async () => {
    renderer = create(
      <ThinkButton
        values={[false, "low", "medium", "xhigh"]}
        value="xhigh"
        onChange={onChange}
      />,
    );
  });
  await act(async () => renderer.root.findByType("button").props.onClick());
  const options = renderer.root.findAllByProps({ role: "menuitemradio" });
  expect(options.map((option) => option.children.join(""))).toEqual([
    "Off",
    "Low",
    "Medium",
    "Xhigh",
  ]);
  await act(async () => options[0].props.onClick());
  expect(onChange).toHaveBeenCalledWith(false);
  expect(renderer.root.findAllByProps({ role: "menuitemradio" })).toHaveLength(
    0,
  );
});

it("toggles models whose default is on without adding named levels", async () => {
  const onChange = vi.fn();
  await act(async () => {
    renderer = create(
      <ThinkButton values={[false, true]} value={true} onChange={onChange} />,
    );
  });
  const button = renderer.root.findByType("button");
  expect(button.props["aria-pressed"]).toBe(true);
  await act(async () => button.props.onClick());
  expect(onChange).toHaveBeenCalledWith(false);
});

it("does not invent an off option for GPT-OSS", async () => {
  await act(async () => {
    renderer = create(
      <ThinkButton
        values={["low", "medium", "high"]}
        value="medium"
        onChange={vi.fn()}
      />,
    );
  });
  await act(async () => renderer.root.findByType("button").props.onClick());
  expect(
    renderer.root
      .findAllByProps({ role: "menuitemradio" })
      .map((option) => option.children.join("")),
  ).toEqual(["Low", "Medium", "High"]);
});

it("indicates a single thinking value as fixed and hides non-thinking models", async () => {
  await act(async () => {
    renderer = create(
      <ThinkButton values={[true]} value={true} onChange={vi.fn()} />,
    );
  });
  expect(renderer.root.findByType("button").props.disabled).toBe(true);
  expect(renderer.root.findByType("button").props.title).toContain("fixed");
  await act(async () =>
    renderer.update(
      <ThinkButton values={[false]} value={false} onChange={vi.fn()} />,
    ),
  );
  expect(renderer.toJSON()).toBeNull();
});

it("keeps dropdown coordination working after switching from a toggle model", async () => {
  const ref = createRef<HTMLButtonElement & { closeDropdown?: () => void }>();
  await act(async () => {
    renderer = create(
      <ThinkButton
        ref={ref}
        values={[false, true]}
        value={true}
        onChange={vi.fn()}
      />,
      { createNodeMock: () => ({}) },
    );
  });
  await act(async () =>
    renderer.update(
      <ThinkButton
        ref={ref}
        values={["low", "medium", "high"]}
        value="medium"
        onChange={vi.fn()}
      />,
    ),
  );
  await act(async () => renderer.root.findByType("button").props.onClick());
  expect(renderer.root.findAllByProps({ role: "menuitemradio" })).toHaveLength(
    3,
  );
  await act(async () => ref.current?.closeDropdown?.());
  expect(renderer.root.findAllByProps({ role: "menuitemradio" })).toHaveLength(
    0,
  );
});
