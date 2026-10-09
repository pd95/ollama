import {
  QueryClient,
  QueryClientProvider,
  notifyManager,
  defaultScheduler,
} from "@tanstack/react-query";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { Model } from "@/gotypes";
import { ModelList, ModelPicker } from "./ModelPicker";
import { getModelCapabilities } from "@/api";
import { createRef } from "react";
import { CheckIcon } from "@heroicons/react/24/outline";

const state = vi.hoisted(() => ({ models: [] as any[], setSettings: vi.fn() }));
vi.mock("@/hooks/useModels", () => ({
  useModels: () => ({ data: state.models }),
}));
vi.mock("@/hooks/useSelectedModel", () => ({
  useSelectedModel: () => ({
    models: state.models,
    selectedModel: state.models[0],
    setSettings: state.setSettings,
    loading: false,
  }),
}));
vi.mock("@/hooks/useCloudStatus", () => ({
  useCloudStatus: () => ({ cloudDisabled: false }),
}));
vi.mock("@/api", () => ({
  getModelUpstreamInfo: vi.fn().mockResolvedValue({ stale: false }),
  getModelCapabilities: vi.fn(),
}));

let renderer: ReactTestRenderer;
let client: QueryClient;
let hostNodes: Set<any>;
const documentState = {
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  activeElement: null as any,
};
function mockNode(element: any) {
  const node = {
    props: element.props,
    children: [],
    scrollTop: 0,
    style: { paddingBottom: "" },
    clientHeight: 0,
    getBoundingClientRect: () => ({ top: 0, bottom: 0 }),
    contains: (target: any) => hostNodes.has(target),
    focus: vi.fn(() => {
      documentState.activeElement = node;
    }),
  };
  hostNodes.add(node);
  return node;
}
async function mount(element: React.ReactNode) {
  await act(async () => {
    renderer = create(
      <QueryClientProvider client={client}>{element}</QueryClientProvider>,
      { createNodeMock: mockNode },
    );
  });
}
function key(key: string) {
  return { key, preventDefault: vi.fn(), stopPropagation: vi.fn() };
}
function content() {
  return JSON.stringify(renderer.toJSON(), (key, value) =>
    key === "ref" ? undefined : value,
  );
}
beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  notifyManager.setScheduler(queueMicrotask);
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  hostNodes = new Set();
  documentState.activeElement = null;
  documentState.addEventListener.mockReset();
  documentState.removeEventListener.mockReset();
  vi.stubGlobal("document", documentState);
  vi.mocked(getModelCapabilities).mockReset();
  vi.mocked(getModelCapabilities).mockResolvedValue({
    capabilities: ["completion"],
  });
  state.models = [
    Object.assign(new Model({ model: "my-alias", digest: "one" }), {
      capabilities: ["completion", "vision", "audio"],
      isCloud: () => false,
    }),
  ];
});
afterEach(async () => {
  await act(async () => renderer?.unmount());
  client.clear();
  notifyManager.setScheduler(defaultScheduler);
  vi.unstubAllGlobals();
});

it("shows advertised image and audio input before a model is selected", async () => {
  await act(async () => {
    renderer = create(
      <QueryClientProvider client={client}>
        <ModelList
          models={state.models}
          selectedModel={null}
          onModelSelect={vi.fn()}
          cloudDisabled={false}
          isOpen
        />
      </QueryClientProvider>,
    );
  });
  const content = JSON.stringify(renderer.toJSON());
  expect(content).toContain("Image input");
  expect(content).toContain("Audio input");
});

it("distinguishes image generation from image input and does not infer by name", async () => {
  state.models = [
    Object.assign(
      new Model({ model: "vision-audio-thinking", digest: "one" }),
      { capabilities: ["image"], isCloud: () => false },
    ),
  ];
  await mount(
    <ModelList
      models={state.models}
      selectedModel={null}
      onModelSelect={vi.fn()}
      cloudDisabled={false}
      isOpen
    />,
  );
  expect(content()).toContain("Image generation");
  expect(content()).not.toContain("Image input");
  expect(content()).not.toContain("Audio input");
  expect(content()).not.toContain("Thinking");
  expect(getModelCapabilities).not.toHaveBeenCalled();
});

it("keeps selection usable during discovery and shows failures as unknown", async () => {
  state.models[0].capabilities = undefined;
  let finish!: (value: {}) => void;
  vi.mocked(getModelCapabilities).mockReturnValue(
    new Promise((resolve) => {
      finish = resolve;
    }),
  );
  const select = vi.fn();
  await mount(
    <ModelList
      models={state.models}
      selectedModel={null}
      onModelSelect={select}
      cloudDisabled={false}
      isOpen
    />,
  );
  expect(content()).toContain("Loading capabilities");
  const row = renderer.root.findByProps({ role: "option" });
  await act(async () => row.props.onClick());
  expect(select).toHaveBeenCalledWith(state.models[0]);
  await act(async () => finish({}));
  expect(content()).toContain("Capabilities unknown");
  expect(content()).not.toContain("No advertised capabilities");
});

it("discovers visible entries, skips offscreen entries, and keeps cloud labels separate", async () => {
  const intersections: ((entries: { isIntersecting: boolean }[]) => void)[] =
    [];
  vi.stubGlobal(
    "IntersectionObserver",
    class {
      constructor(callback: (typeof intersections)[number]) {
        intersections.push(callback);
      }
      observe() {}
      disconnect() {}
    },
  );
  state.models = ["first", "second:cloud", "offscreen"].map((model) =>
    Object.assign(new Model({ model }), {
      isCloud: () => model.endsWith(":cloud"),
    }),
  );
  await mount(
    <ModelList
      models={state.models}
      selectedModel={null}
      onModelSelect={vi.fn()}
      cloudDisabled={false}
      isOpen
    />,
  );
  expect(getModelCapabilities).toHaveBeenCalledTimes(1);
  await act(async () => intersections[1]([{ isIntersecting: true }]));
  expect(getModelCapabilities).toHaveBeenCalledTimes(2);
  expect(
    vi.mocked(getModelCapabilities).mock.calls.map(([name]) => name),
  ).toEqual(["first", "second:cloud"]);
  expect(content()).toContain("Cloud model");
});

it("exposes option selection and supports keyboard navigation and Enter", async () => {
  state.models.push(
    Object.assign(new Model({ model: "text", digest: "two" }), {
      capabilities: ["completion"],
      isCloud: () => false,
    }),
  );
  const select = vi.fn();
  const active = vi.fn();
  await mount(
    <ModelList
      models={state.models}
      selectedModel={state.models[0]}
      onModelSelect={select}
      cloudDisabled={false}
      isOpen
      onActiveChange={active}
      listId="choices"
    />,
  );
  const list = renderer.root.findByProps({ role: "listbox" });
  expect(
    renderer.root
      .findAllByProps({ role: "option" })
      .map((row) => row.props["aria-selected"]),
  ).toEqual([true, false]);
  await act(async () => list.props.onKeyDown(key("ArrowDown")));
  expect(active).toHaveBeenLastCalledWith("choices-option-1");
  await act(async () => list.props.onKeyDown(key("Enter")));
  expect(select).toHaveBeenLastCalledWith(state.models[1]);
  await act(async () => list.props.onKeyDown(key("ArrowDown")));
  expect(active).toHaveBeenLastCalledWith("choices-option-0");
});

it("opens labeled details, restores focus on Escape, and keeps the panels exclusive", async () => {
  const detailsRef = createRef<HTMLButtonElement>();
  await mount(<ModelPicker detailsButtonRef={detailsRef} />);
  const detailsButton = renderer.root.findByProps({
    "aria-label": "Model information for my-alias",
  });
  await act(async () => detailsButton.props.onClick());
  expect(
    renderer.root.findByProps({ role: "dialog" }).props["aria-label"],
  ).toBe("Model information for my-alias");
  expect(content()).toContain("Not reported");
  const dialogNode = [...hostNodes].find(
    (node) => node.props.role === "dialog",
  );
  expect(documentState.activeElement).toBe(dialogNode);
  const escape = key("Escape");
  await act(async () =>
    renderer.root
      .findAllByType("div")
      .find(
        (node) =>
          node.props.onKeyDown && node.props.className.includes("relative"),
      )!
      .props.onKeyDown(escape),
  );
  expect(escape.stopPropagation).toHaveBeenCalled();
  expect(documentState.activeElement).toBe(detailsRef.current);
  expect(renderer.root.findAllByProps({ role: "dialog" })).toHaveLength(0);
  const selector = renderer.root.findByProps({ title: "Select model" });
  await act(async () => selector.props.onClick());
  expect(
    renderer.root.findByProps({ role: "combobox" }).props[
      "aria-activedescendant"
    ],
  ).toBeDefined();
  await act(async () => detailsButton.props.onClick());
  expect(renderer.root.findAllByProps({ role: "listbox" })).toHaveLength(0);
  expect(renderer.root.findAllByProps({ role: "dialog" })).toHaveLength(1);
});

it("shows an explicit empty list and closes details on an outside click", async () => {
  state.models[0].capabilities = [];
  await mount(<ModelPicker />);
  await act(async () =>
    renderer.root.findByProps({ title: "Select model" }).props.onClick(),
  );
  expect(content()).toContain("No advertised capabilities");
  expect(content()).not.toContain("Capabilities unknown");
  await act(async () =>
    renderer.root
      .findByProps({ "aria-label": "Model information for my-alias" })
      .props.onClick(),
  );
  const outside = documentState.addEventListener.mock.calls.find(
    ([name]) => name === "mousedown",
  )![1];
  await act(async () => outside({ target: {} }));
  expect(renderer.root.findAllByProps({ role: "dialog" })).toHaveLength(0);
});

it("keeps search and selection working with capability rows", async () => {
  await mount(<ModelPicker />);
  await act(async () =>
    renderer.root.findByProps({ title: "Select model" }).props.onClick(),
  );
  const search = renderer.root.findByProps({ role: "combobox" });
  await act(async () => search.props.onChange({ target: { value: "alias" } }));
  expect(renderer.root.findByProps({ role: "combobox" }).props.value).toBe(
    "alias",
  );
  await act(async () =>
    renderer.root.findByProps({ role: "option" }).props.onClick(),
  );
  expect(state.setSettings).toHaveBeenCalledWith({ SelectedModel: "my-alias" });
  expect(renderer.root.findAllByProps({ role: "listbox" })).toHaveLength(0);
});

it("does not label unfamiliar advertised capabilities as an empty list", async () => {
  state.models[0].capabilities = ["embedding", "future-capability"];
  await mount(
    <ModelList
      models={state.models}
      selectedModel={null}
      onModelSelect={vi.fn()}
      cloudDisabled={false}
      isOpen
    />,
  );
  expect(content()).toContain("Other capabilities advertised");
  expect(content()).not.toContain("No advertised capabilities");
});

it("reveals whole rows without moving visible rows or including the search header", async () => {
  state.models.push(
    Object.assign(new Model({ model: "second", digest: "two" }), {
      capabilities: ["completion"],
      isCloud: () => false,
    }),
  );
  const ref = createRef<React.ComponentRef<typeof ModelList>>();
  await mount(
    <ModelList
      ref={ref}
      models={state.models}
      selectedModel={state.models[0]}
      onModelSelect={vi.fn()}
      cloudDisabled={false}
      isOpen
    />,
  );
  const container = [...hostNodes].find(
    (node) => node.props.role === "listbox",
  );
  const rows = [...hostNodes].filter((node) => node.props.role === "option");
  container.children = rows;
  container.clientHeight = 120;
  container.getBoundingClientRect = () => ({ top: 80, bottom: 200 });
  // The list starts below a search header; offsetTop belongs to that ancestor.
  rows.forEach((row, index) => {
    row.offsetTop = 80 + index * 60;
    row.clientHeight = 60;
    row.getBoundingClientRect = () => ({
      top: 80 + index * 60 - container.scrollTop,
      bottom: 140 + index * 60 - container.scrollTop,
    });
  });
  await act(async () => ref.current!.scrollToSelectedModel());
  expect(container.scrollTop).toBe(0);
  container.scrollTop = 30;
  await act(async () => ref.current!.scrollToSelectedModel());
  expect(container.scrollTop).toBe(0);
  // A shorter viewport clips the second row's capability line.
  container.clientHeight = 100;
  container.getBoundingClientRect = () => ({ top: 80, bottom: 180 });
  await act(async () =>
    renderer.root
      .findByProps({ role: "listbox" })
      .props.onKeyDown(key("ArrowDown")),
  );
  expect(container.scrollTop).toBe(60);
  await act(async () => ref.current!.scrollToSelectedModel());
  expect(container.scrollTop).toBe(60);
});

it("shows a continuation cue only while earlier rows are hidden", async () => {
  await mount(
    <ModelList
      models={state.models}
      selectedModel={state.models[0]}
      onModelSelect={vi.fn()}
      cloudDisabled={false}
      isOpen
    />,
  );
  const cue = () =>
    renderer.root
      .findAllByType("div")
      .filter((node) => node.props["aria-hidden"] === true);
  expect(cue()).toHaveLength(0);
  await act(async () =>
    renderer.root.findByProps({ role: "listbox" }).props.onScroll({
      currentTarget: { scrollTop: 30 },
    }),
  );
  expect(cue()).toHaveLength(1);
  expect(cue()[0].props["aria-hidden"]).toBe(true);
  await act(async () =>
    renderer.root.findByProps({ role: "listbox" }).props.onScroll({
      currentTarget: { scrollTop: 0 },
    }),
  );
  expect(cue()).toHaveLength(0);
});

it("keeps the selected checkmark on its model while another row is highlighted", async () => {
  state.models.push(
    Object.assign(new Model({ model: "second", digest: "two" }), {
      capabilities: ["completion"],
      isCloud: () => false,
    }),
  );
  await mount(
    <ModelList
      models={state.models}
      selectedModel={state.models[0]}
      onModelSelect={vi.fn()}
      cloudDisabled={false}
      isOpen
    />,
  );
  const rows = () => renderer.root.findAllByProps({ role: "option" });
  expect(rows()[0].findAllByType(CheckIcon)).toHaveLength(1);
  expect(rows()[1].findAllByType(CheckIcon)).toHaveLength(0);
  await act(async () => rows()[1].props.onMouseEnter());
  expect(rows()[0].findAllByType(CheckIcon)).toHaveLength(1);
  expect(rows()[1].findAllByType(CheckIcon)).toHaveLength(0);
  expect(rows().map((row) => row.props["aria-selected"])).toEqual([
    true,
    false,
  ]);
});

it("opens near the end with a complete leading row instead of a detached status", async () => {
  let resize!: () => void;
  vi.stubGlobal(
    "ResizeObserver",
    class {
      constructor(callback: () => void) {
        resize = callback;
      }
      observe() {}
      disconnect() {}
    },
  );
  state.models = Array.from({ length: 7 }, (_, index) =>
    Object.assign(new Model({ model: `model-${index}`, digest: `${index}` }), {
      capabilities: ["completion"],
      isCloud: () => false,
    }),
  );
  const ref = createRef<React.ComponentRef<typeof ModelList>>();
  await mount(
    <ModelList
      ref={ref}
      models={state.models}
      selectedModel={state.models[6]}
      onModelSelect={vi.fn()}
      cloudDisabled={false}
      isOpen
    />,
  );
  const container = [...hostNodes].find(
    (node) => node.props.role === "listbox",
  );
  const rows = [...hostNodes].filter((node) => node.props.role === "option");
  container.children = rows;
  container.clientHeight = 320;
  container.getBoundingClientRect = () => ({ top: 80, bottom: 400 });
  let scrollTop = 0;
  const heights = Array(7).fill(60);
  Object.defineProperty(container, "scrollTop", {
    get: () => scrollTop,
    set: (value) => {
      const padding = parseFloat(container.style.paddingBottom) || 0;
      const contentHeight = heights.reduce((sum, height) => sum + height, 0);
      scrollTop = Math.max(0, Math.min(value, contentHeight + padding - 320));
    },
  });
  rows.forEach((row, index) => {
    row.getBoundingClientRect = () => ({
      top:
        80 +
        heights.slice(0, index).reduce((sum, height) => sum + height, 0) -
        container.scrollTop,
      bottom:
        80 +
        heights.slice(0, index + 1).reduce((sum, height) => sum + height, 0) -
        container.scrollTop,
      height: heights[index],
    });
  });
  await act(async () => ref.current!.scrollToSelectedModel());
  expect(container.scrollTop).toBe(120);
  expect(parseFloat(container.style.paddingBottom)).toBe(20);
  expect(rows[2].getBoundingClientRect().top).toBe(80);
  expect(rows[6].getBoundingClientRect().bottom).toBeLessThanOrEqual(400);
  await act(async () => resize());
  heights[0] = 100;
  heights[2] = 80;
  await act(async () => resize());
  expect(container.scrollTop).toBe(160);
  expect(parseFloat(container.style.paddingBottom)).toBe(0);
  expect(rows[2].getBoundingClientRect().top).toBe(80);
  expect(rows[6].getBoundingClientRect().bottom).toBeLessThanOrEqual(400);
  // Wheel scrolling remains continuous, including positions between row starts.
  container.scrollTop = 110;
  await act(async () =>
    renderer.root
      .findByProps({ role: "listbox" })
      .props.onScroll({ currentTarget: container }),
  );
  expect(container.scrollTop).toBe(110);
  await act(async () => ref.current!.scrollToTop());
  expect(container.scrollTop).toBe(0);
  expect(parseFloat(container.style.paddingBottom) || 0).toBe(0);
});

it("shows useful model information including its local runtime backend", async () => {
  Object.assign(state.models[0], {
    size: 8_100_000_000,
    metadata: {
      format: "safetensors",
      runner: "mlx",
      parameterSize: "8.1B",
      quantization: "NVFP4",
      contextLength: 65536,
    },
  });
  await mount(<ModelPicker />);
  await act(async () =>
    renderer.root.findByProps({ title: "Model information" }).props.onClick(),
  );
  expect(content()).toContain("MLX");
  expect(content()).toContain("8.1 GB");
  expect(content()).toContain("8.1B");
  expect(content()).toContain("NVFP4");
  expect(content()).toContain("65,536 tokens");
  expect(content()).toContain("Local");
  expect(content()).toContain("Reported");
});

it("does not present a remote model's metadata as a local backend or file size", async () => {
  Object.assign(state.models[0], {
    size: 8_100_000_000,
    metadata: { format: "gguf" },
  });
  vi.mocked(getModelCapabilities).mockResolvedValue({
    capabilities: ["completion"],
    remoteHost: "https://example.test",
  });
  await mount(<ModelPicker />);
  await act(async () =>
    renderer.root.findByProps({ title: "Model information" }).props.onClick(),
  );
  expect(content()).toContain("Remote server");
  expect(content()).not.toContain("llama.cpp");
  expect(content()).not.toContain("8.1 GB");
  expect(content()).toContain("Not reported");
});
