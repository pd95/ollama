import {
  QueryClient,
  QueryClientProvider,
  notifyManager,
  defaultScheduler,
} from "@tanstack/react-query";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useThinking } from "./useThinking";
import { useModelCapabilities } from "./useModelCapabilities";

vi.mock("./useModelCapabilities", () => ({ useModelCapabilities: vi.fn() }));
let renderer: ReactTestRenderer;
let queryClient: QueryClient;
let thinking: ReturnType<typeof useThinking>;
let metadata: any;
let isPending = false;
function Control({ model = "my-alias", chat: _chat = "new" }) {
  return (
    <QueryClientProvider client={queryClient}>
      <Inner model={model} />
    </QueryClientProvider>
  );
}
function Inner({ model }: { model: string }) {
  thinking = useThinking(model, false, "high");
  return null;
}
beforeEach(() => {
  queryClient = new QueryClient();
  notifyManager.setScheduler(queueMicrotask);
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  metadata = undefined;
  isPending = false;
  vi.mocked(useModelCapabilities).mockImplementation(
    () => ({ data: metadata, isPending }) as never,
  );
});
afterEach(async () => {
  await act(async () => renderer?.unmount());
  queryClient.clear();
  notifyManager.setScheduler(defaultScheduler);
  vi.unstubAllGlobals();
});

it("uses the advertised default instead of global legacy settings for aliases", async () => {
  metadata = { thinking: { values: [false, true], default: true } };
  await act(async () => {
    renderer = create(<Control />);
  });
  expect(thinking.value).toBe(true);
  await act(async () => thinking.setValue(false));
  expect(thinking.value).toBe(false);
  expect(thinking.controls?.values).toEqual([false, true]);
});

it("retains supported choices and resets unsupported choices when switching models", async () => {
  metadata = {
    thinking: { values: [false, "low", "xhigh"], default: "xhigh" },
  };
  await act(async () => {
    renderer = create(<Control />);
  });
  expect(thinking.value).toBe("xhigh");
  await act(async () => thinking.setValue("low"));
  metadata = {
    thinking: { values: ["low", "medium", "high"], default: "medium" },
  };
  await act(async () => renderer.update(<Control model="another-alias" />));
  expect(thinking.value).toBe("low");
  metadata = { thinking: { values: [false, true], default: true } };
  await act(async () => renderer.update(<Control model="boolean-model" />));
  expect(thinking.value).toBe(true);
  metadata = {
    thinking: { values: ["low", "medium", "high"], default: "medium" },
  };
  await act(async () => renderer.update(<Control model="another-alias" />));
  expect(thinking.value).toBe("medium");
});

it("retains an explicit choice when a new chat is assigned its saved ID or the form remounts", async () => {
  metadata = { thinking: { values: [false, true], default: true } };
  await act(async () => {
    renderer = create(<Control chat="new" />);
  });
  await act(async () => thinking.setValue(false));
  await act(async () => renderer.unmount());
  await act(async () => {
    renderer = create(<Control chat="saved-id" />);
  });
  expect(thinking.value).toBe(false);
});

it("only uses legacy controls when metadata is absent", async () => {
  await act(async () => {
    renderer = create(<Control model="gpt-oss:20b" />);
  });
  expect(thinking.value).toBe("high");
  metadata = { thinking: { values: [false], default: false } };
  await act(async () => renderer.update(<Control model="gpt-oss:20b" />));
  expect(thinking.controls?.values).toEqual([false]);
  expect(thinking.value).toBe(false);
  metadata = undefined;
  await act(async () => renderer.update(<Control />));
  expect(thinking.controls).toBeUndefined();
  expect(thinking.value).toBeUndefined();
});

it("recognizes existing thinking/web-search restrictions from renderer metadata for aliases", async () => {
  metadata = {
    renderer: "apertus1p5",
    thinking: { values: [false, true], default: false },
  };
  await act(async () => {
    renderer = create(<Control />);
  });
  expect(thinking.conflictsWithWebSearch).toBe(true);
  metadata = {
    renderer: "qwen3.5",
    thinking: { values: [false, true], default: true },
  };
  await act(async () => renderer.update(<Control />));
  expect(thinking.conflictsWithWebSearch).toBe(false);
});

it("retains a compatible choice while the next model's discovery is pending", async () => {
  metadata = { thinking: { values: [false, true], default: true } };
  await act(async () => {
    renderer = create(<Control />);
  });
  await act(async () => thinking.setValue(false));
  metadata = undefined;
  isPending = true;
  await act(async () => renderer.update(<Control model="loading-alias" />));
  expect(thinking.value).toBeUndefined();
  metadata = { thinking: { values: [false, true], default: true } };
  isPending = false;
  await act(async () => renderer.update(<Control model="loading-alias" />));
  expect(thinking.value).toBe(false);
});
