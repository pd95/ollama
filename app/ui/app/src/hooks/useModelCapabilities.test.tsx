import {
  QueryClient,
  QueryClientProvider,
  notifyManager,
  defaultScheduler,
} from "@tanstack/react-query";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { getModelCapabilities } from "@/api";
import type { Model } from "@/gotypes";
import { modelCapabilityKey } from "@/lib/modelCapabilities";
import {
  useModelCapabilities,
  useModelCapabilitySummary,
} from "./useModelCapabilities";

const state = vi.hoisted(() => ({ models: [] as Model[] }));
vi.mock("./useModels", () => ({ useModels: () => ({ data: state.models }) }));
vi.mock("@/api", () => ({ getModelCapabilities: vi.fn() }));
let renderer: ReactTestRenderer;
let client: QueryClient;
let summary: ReturnType<typeof useModelCapabilitySummary>;
let details: ReturnType<typeof useModelCapabilities>;
function Probe({
  model,
  enabled = true,
  selected = false,
}: {
  model: Model;
  enabled?: boolean;
  selected?: boolean;
}) {
  summary = useModelCapabilitySummary(model, enabled);
  if (selected) return <Selected name={model.model} />;
  return null;
}
function Selected({ name }: { name: string }) {
  details = useModelCapabilities(name);
  return null;
}
async function render(model: Model, enabled = true, selected = false) {
  state.models = [model];
  await act(async () => {
    const element = (
      <QueryClientProvider client={client}>
        <Probe model={model} enabled={enabled} selected={selected} />
      </QueryClientProvider>
    );
    if (renderer) renderer.update(element);
    else renderer = create(element);
  });
}
beforeEach(() => {
  renderer = undefined as any;
  client = new QueryClient();
  notifyManager.setScheduler(queueMicrotask);
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.mocked(getModelCapabilities).mockReset();
  vi.mocked(getModelCapabilities).mockResolvedValue({
    capabilities: ["vision"],
    thinking: { values: [false, true], default: false },
  });
});
afterEach(async () => {
  await act(async () => renderer?.unmount());
  client.clear();
  notifyManager.setScheduler(defaultScheduler);
  vi.unstubAllGlobals();
});

it("uses list metadata immediately without starting discovery", async () => {
  await render({
    model: "alias",
    digest: "one",
    capabilities: ["completion", "audio"],
  } as Model);
  expect(summary.capabilities).toEqual(["completion", "audio"]);
  expect(summary.isLoading).toBe(false);
  expect(getModelCapabilities).not.toHaveBeenCalled();
});

it("does not discover offscreen entries until they become visible", async () => {
  const model = { model: "alias", digest: "one" } as Model;
  await render(model, false);
  expect(getModelCapabilities).not.toHaveBeenCalled();
  await render(model);
  expect(summary.capabilities).toEqual(["vision"]);
  expect(getModelCapabilities).toHaveBeenCalledTimes(1);
});

it("shares discovery with selected-model controls and reuses cache on reopening", async () => {
  const model = { model: "alias", digest: "one" } as Model;
  await render(model, true, true);
  expect(details.data?.thinking?.default).toBe(false);
  expect(getModelCapabilities).toHaveBeenCalledTimes(1);
  await render(model, false);
  await render(model);
  expect(getModelCapabilities).toHaveBeenCalledTimes(1);
});

it("never carries a replaced tag's capabilities into the new digest", async () => {
  const oldModel = { model: "alias", digest: "one" } as Model;
  await render(oldModel);
  expect(summary.capabilities).toEqual(["vision"]);
  let finish!: (value: { capabilities: string[] }) => void;
  vi.mocked(getModelCapabilities).mockReturnValue(
    new Promise((resolve) => {
      finish = resolve;
    }),
  );
  await render({ model: "alias", digest: "two" } as Model);
  expect(summary.capabilities).toBeUndefined();
  expect(summary.isLoading).toBe(true);
  await act(async () => finish({ capabilities: ["completion"] }));
  expect(summary.capabilities).toEqual(["completion"]);
  expect(client.getQueryData(modelCapabilityKey("alias", "one"))).toEqual(
    expect.objectContaining({ capabilities: ["vision"] }),
  );
});

it("keeps unavailable metadata unknown and retries explicitly", async () => {
  const { CapabilityDiscoveryError } = await import("@/lib/capabilityRequests");
  vi.mocked(getModelCapabilities).mockRejectedValue(
    new CapabilityDiscoveryError(404),
  );
  await render({ model: "alias:cloud" } as Model);
  expect(summary.capabilities).toBeUndefined();
  expect(summary.isLoading).toBe(false);
  expect(getModelCapabilities).toHaveBeenCalledTimes(1);
  vi.mocked(getModelCapabilities).mockResolvedValue({
    capabilities: ["audio"],
  });
  await act(async () => {
    await summary.refetch();
  });
  expect(summary.capabilities).toEqual(["audio"]);
});
