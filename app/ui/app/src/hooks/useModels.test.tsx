import {
  QueryClient,
  QueryClientProvider,
  notifyManager,
  defaultScheduler,
} from "@tanstack/react-query";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { Model } from "@/gotypes";
import { getModels } from "@/api";
import { useModels } from "./useModels";
const state = vi.hoisted(() => ({
  recommendations: [] as { model: string; capabilities?: string[] }[],
}));
vi.mock("./useFeaturedModels", () => ({
  useFeaturedModels: () => ({ data: state.recommendations, isLoading: false }),
}));
vi.mock("./useCloudStatus", () => ({
  useCloudStatus: () => ({ cloudDisabled: false }),
}));
vi.mock("@/api", () => ({ getModels: vi.fn() }));
let renderer: ReactTestRenderer;
let client: QueryClient;
let models: Model[];
function Probe({ search = "" }) {
  models = useModels(search).data;
  return null;
}
beforeEach(() => {
  client = new QueryClient();
  notifyManager.setScheduler(queueMicrotask);
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  state.recommendations = [
    { model: "alias", capabilities: ["completion"] },
    { model: "remote:cloud", capabilities: ["completion", "vision"] },
    { model: "unknown" },
  ];
  vi.mocked(getModels).mockResolvedValue([
    {
      model: "alias",
      digest: "one",
      capabilities: ["completion", "audio"],
      isCloud: () => false,
    } as Model,
  ]);
});
afterEach(async () => {
  await act(async () => renderer?.unmount());
  client.clear();
  notifyManager.setScheduler(defaultScheduler);
  vi.unstubAllGlobals();
});
it("preserves exact local metadata over recommendations and keeps missing metadata unknown", async () => {
  await act(async () => {
    renderer = create(
      <QueryClientProvider client={client}>
        <Probe />
      </QueryClientProvider>,
    );
  });
  expect(models.map((model) => [model.model, model.capabilities])).toEqual([
    ["alias", ["completion", "audio"]],
    ["remote:cloud", ["completion", "vision"]],
    ["unknown", undefined],
  ]);
});
it("retains capability metadata when searching recommended entries", async () => {
  await act(async () => {
    renderer = create(
      <QueryClientProvider client={client}>
        <Probe search="remote" />
      </QueryClientProvider>,
    );
  });
  expect(models).toHaveLength(1);
  expect(models[0].capabilities).toEqual(["completion", "vision"]);
});
