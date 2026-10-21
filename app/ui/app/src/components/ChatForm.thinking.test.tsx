import {
  QueryClient,
  QueryClientProvider,
  notifyManager,
  defaultScheduler,
} from "@tanstack/react-query";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import ChatForm from "./ChatForm";
import { WebSearchButton } from "./WebSearchButton";

const state = vi.hoisted(() => ({
  model: "my-alias",
  metadata: {} as any,
  settings: {
    webSearchEnabled: false,
    thinkEnabled: false,
    thinkLevel: "high",
  },
  setSettings: vi.fn(),
  send: vi.fn(),
}));
vi.mock("@/hooks/useSelectedModel", () => ({
  useSelectedModel: () => ({
    selectedModel: { model: state.model, isCloud: () => false },
  }),
}));
vi.mock("@/hooks/useModelCapabilities", () => ({
  useModelCapabilities: () => ({ data: state.metadata }),
  useHasToolsCapability: () => true,
  useHasVisionCapability: () => false,
}));
vi.mock("@/hooks/useSettings", () => ({
  useSettings: () => ({
    settings: state.settings,
    setSettings: state.setSettings,
  }),
}));
vi.mock("@/hooks/useChats", () => ({
  useSendMessage: () => ({ mutate: state.send }),
  useIsStreaming: () => false,
  useCancelMessage: () => vi.fn(),
}));
vi.mock("@/hooks/useUser", () => ({
  useUser: () => ({ isAuthenticated: true, isLoading: false }),
}));
vi.mock("@/hooks/useCloudStatus", () => ({
  useCloudStatus: () => ({ cloudDisabled: false }),
}));
vi.mock("@tanstack/react-router", () => ({ useNavigate: () => vi.fn() }));
vi.mock("./ModelPicker", () => ({ ModelPicker: () => null }));
vi.mock("./Logo", () => ({ default: () => null }));
vi.mock("./DisplayLogin", () => ({ DisplayLogin: () => null }));
vi.mock("./WebSearchButton", () => ({ WebSearchButton: () => null }));

let renderer: ReactTestRenderer;
let queryClient: QueryClient;
function Form(props: React.ComponentProps<typeof ChatForm>) {
  return (
    <QueryClientProvider client={queryClient}>
      <ChatForm {...props} />
    </QueryClientProvider>
  );
}
beforeEach(() => {
  queryClient = new QueryClient();
  notifyManager.setScheduler(queueMicrotask);
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("window", {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  });
  vi.stubGlobal("document", {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  });
  state.model = "my-alias";
  state.metadata = { thinking: { values: [false, true], default: true } };
  state.settings = {
    webSearchEnabled: false,
    thinkEnabled: false,
    thinkLevel: "high",
  };
  state.setSettings.mockReset();
  state.send.mockReset();
});
afterEach(async () => {
  await act(async () => renderer?.unmount());
  queryClient.clear();
  notifyManager.setScheduler(defaultScheduler);
  vi.unstubAllGlobals();
});

async function submitHello() {
  await act(async () =>
    renderer.root.findByType("textarea").props.onChange({
      target: { value: "Hello", style: {}, scrollHeight: 24 },
    }),
  );
  await act(async () =>
    renderer.root.findByType("textarea").props.onKeyDown({
      key: "Enter",
      preventDefault: vi.fn(),
      shiftKey: false,
      nativeEvent: { isComposing: false },
    }),
  );
}

it("sends the backend default then explicit false through normal chat submission", async () => {
  await act(async () => {
    renderer = create(<Form hasMessages={false} />);
  });
  await submitHello();
  expect(state.send).toHaveBeenLastCalledWith(
    expect.objectContaining({ think: true }),
  );
  await act(async () =>
    renderer.root.findByProps({ "aria-label": "Thinking" }).props.onClick(),
  );
  await submitHello();
  expect(state.send).toHaveBeenLastCalledWith(
    expect.objectContaining({ think: false }),
  );
});

it("passes arbitrary named levels through the edit/submit callback", async () => {
  state.metadata = {
    thinking: { values: [false, "low", "xhigh"], default: "xhigh" },
  };
  const onSubmit = vi.fn();
  await act(async () => {
    renderer = create(<Form hasMessages={true} onSubmit={onSubmit} />);
  });
  await submitHello();
  expect(onSubmit).toHaveBeenLastCalledWith(
    "Hello",
    expect.objectContaining({ think: "xhigh" }),
  );
});

it("turns thinking off for restricted aliases when web search is enabled", async () => {
  state.metadata = {
    renderer: "apertus1p5",
    thinking: { values: [false, true], default: true },
  };
  await act(async () => {
    renderer = create(<Form hasMessages={false} />);
  });
  await act(async () =>
    renderer.root.findByType(WebSearchButton).props.onToggle(),
  );
  expect(state.setSettings).toHaveBeenLastCalledWith({
    WebSearchEnabled: true,
  });
  await submitHello();
  expect(state.send).toHaveBeenLastCalledWith(
    expect.objectContaining({ think: false }),
  );
});

it("allows thinking and web search together for models without that restriction", async () => {
  state.settings.webSearchEnabled = true;
  state.metadata = {
    renderer: "qwen3.5",
    thinking: { values: [false, true], default: true },
  };
  await act(async () => {
    renderer = create(<Form hasMessages={false} />);
  });
  expect(state.setSettings).not.toHaveBeenCalled();
  await submitHello();
  expect(state.send).toHaveBeenLastCalledWith(
    expect.objectContaining({ think: true, webSearch: true }),
  );
});
