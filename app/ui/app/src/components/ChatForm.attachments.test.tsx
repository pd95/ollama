import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import ChatForm from "./ChatForm";
import { ImageThumbnail } from "./ImageThumbnail";

const state = vi.hoisted(() => ({
  model: "gptoss-mlx:20b",
  metadata: { capabilities: ["completion"] } as any,
  messages: [] as any[],
  editingMessage: null as any,
}));
vi.mock("@/hooks/useSelectedModel", () => ({
  useSelectedModel: () => ({
    selectedModel: { model: state.model, isCloud: () => false },
  }),
}));
vi.mock("@/hooks/useModelCapabilities", () => ({
  useModelCapabilities: () => ({ data: state.metadata }),
  useHasVisionCapability: () =>
    state.metadata?.capabilities?.includes("vision") ?? false,
  useHasToolsCapability: () => false,
}));
vi.mock("@/hooks/useSettings", () => ({
  useSettings: () => ({ settings: {}, setSettings: vi.fn() }),
}));
vi.mock("@/hooks/useChats", () => ({
  useSendMessage: () => ({ mutate: vi.fn() }),
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
vi.mock("./ImageThumbnail", () => ({ ImageThumbnail: () => null }));

let renderer: ReactTestRenderer;
let client: QueryClient;
let receive: (
  files: { filename: string; data: Uint8Array }[],
  errors: [],
) => void;
const submit = vi.fn();
const nativePicker = vi.fn();
function Form() {
  return (
    <QueryClientProvider client={client}>
      <ChatForm
        hasMessages={false}
        messages={state.messages}
        editingMessage={state.editingMessage}
        onSubmit={submit}
        onFilesReceived={(callback) => {
          receive = callback;
        }}
      />
    </QueryClientProvider>
  );
}
const files = ["photo.png", "speech.mp3", "notes.txt"].map((filename) => ({
  filename,
  data: new Uint8Array([1, 2, 3]),
}));
beforeEach(() => {
  client = new QueryClient();
  state.model = "gptoss-mlx:20b";
  state.metadata = { capabilities: ["completion"] };
  state.messages = [];
  state.editingMessage = null;
  submit.mockReset();
  nativePicker.mockReset();
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("window", {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    webview: { selectMultipleFiles: nativePicker },
  });
  vi.stubGlobal("document", {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  });
  vi.stubGlobal(
    "FileReader",
    class {
      result: ArrayBuffer | null = null;
      onload?: () => void;
      async readAsArrayBuffer(file: File) {
        this.result = await file.arrayBuffer();
        this.onload?.();
      }
    },
  );
});
afterEach(async () => {
  await act(async () => renderer?.unmount());
  client.clear();
  vi.unstubAllGlobals();
});
async function render() {
  await act(async () => {
    renderer = create(<Form />);
  });
}
async function enterHello() {
  await act(async () =>
    renderer.root.findByType("textarea").props.onChange({
      target: { value: "Hello", style: {}, scrollHeight: 24 },
    }),
  );
}
async function send() {
  await act(async () =>
    renderer.root.findByType("textarea").props.onKeyDown({
      key: "Enter",
      preventDefault: vi.fn(),
      shiftKey: false,
      nativeEvent: { isComposing: false },
    }),
  );
}
async function dismissMediaErrors() {
  for (const filename of ["photo.png", "speech.mp3"]) {
    await act(async () =>
      renderer.root
        .findByProps({ "aria-label": `Remove ${filename}` })
        .props.onClick(),
    );
  }
}

it("rejects incoming image/audio for GPT-OSS and still sends a text document", async () => {
  await render();
  await act(async () => receive(files, []));
  expect(renderer.root.findAllByType(ImageThumbnail)).toHaveLength(0);
  await enterHello();
  await send();
  expect(submit).not.toHaveBeenCalled();
  await dismissMediaErrors();
  await enterHello();
  await send();
  expect(submit).toHaveBeenCalledWith(
    "Hello",
    expect.objectContaining({ attachments: [files[2]] }),
  );
});

it.each([
  [false, false],
  [true, false],
  [false, true],
  [true, true],
])(
  "passes selected media capabilities to the native picker (%s, %s)",
  async (images, audio) => {
    state.metadata = {
      capabilities: [
        "completion",
        ...(images ? ["vision"] : []),
        ...(audio ? ["audio"] : []),
      ],
    };
    nativePicker.mockResolvedValue(null);
    await render();
    await act(async () =>
      renderer.root
        .findByProps({ title: "Upload multiple files" })
        .props.onClick(),
    );
    expect(nativePicker).toHaveBeenCalledWith({ images, audio });
  },
);

it("blocks sending incompatible queued attachments after model switch and preserves them for switching back", async () => {
  state.metadata = { capabilities: ["vision", "audio"] };
  await render();
  await act(async () => receive(files.slice(0, 2), []));
  await enterHello();
  state.metadata = { capabilities: ["completion"] };
  await act(async () => renderer.update(<Form />));
  await send();
  expect(submit).not.toHaveBeenCalled();
  state.metadata = { capabilities: ["vision", "audio"] };
  await act(async () => renderer.update(<Form />));
  await send();
  expect(submit).toHaveBeenCalledWith(
    "Hello",
    expect.objectContaining({ attachments: files.slice(0, 2) }),
  );
});

it("rechecks the current model when an already-open native picker returns", async () => {
  let resolvePicker!: (value: unknown) => void;
  nativePicker.mockReturnValue(
    new Promise((resolve) => {
      resolvePicker = resolve;
    }),
  );
  state.metadata = { capabilities: ["vision", "audio"] };
  await render();
  let selection: Promise<void>;
  await act(async () => {
    selection = renderer.root
      .findByProps({ title: "Upload multiple files" })
      .props.onClick();
  });
  state.metadata = { capabilities: ["completion"] };
  await act(async () => renderer.update(<Form />));
  await act(async () => {
    resolvePicker(
      files.map((f) => ({
        filename: f.filename,
        dataURL: "data:application/octet-stream;base64,AQID",
      })),
    );
    await selection!;
  });
  await dismissMediaErrors();
  await enterHello();
  await send();
  expect(submit).toHaveBeenCalledWith(
    "Hello",
    expect.objectContaining({ attachments: [files[2]] }),
  );
});

it("filters the browser file input and rejects unsupported media from its change event", async () => {
  await render();
  const input = renderer.root.findByProps({ type: "file" });
  expect(input.props.accept.split(",")).toContain(".txt");
  expect(input.props.accept.split(",")).not.toContain(".png");
  expect(input.props.accept.split(",")).not.toContain(".mp3");
  await act(async () =>
    input.props.onChange({
      target: {
        value: "selection",
        files: [
          new File([new Uint8Array([1, 2, 3])], "photo.png"),
          new File([new Uint8Array([1, 2, 3])], "speech.mp3"),
          new File([new Uint8Array([1, 2, 3])], "notes.txt"),
        ],
      },
    }),
  );
  expect(renderer.root.findAllByType(ImageThumbnail)).toHaveLength(0);
  await dismissMediaErrors();
  await enterHello();
  await send();
  expect(submit).toHaveBeenCalledWith(
    "Hello",
    expect.objectContaining({ attachments: [files[2]] }),
  );
});

it.each(["speech.mp3", "photo.png"])(
  "blocks sending a Markdown follow-up when history contains unsupported %s",
  async (filename) => {
    state.messages = [
      { role: "user", attachments: [{ filename }] },
      { role: "assistant", content: "" },
    ];
    await render();
    await act(async () =>
      receive(
        [{ filename: "notes.md", data: new Uint8Array([35, 32, 65]) }],
        [],
      ),
    );
    await enterHello();
    await send();
    expect(submit).not.toHaveBeenCalled();
    expect(
      renderer.root.findByProps({ role: "alert" }).children.join(""),
    ).toContain(filename);
    state.metadata = { capabilities: ["vision", "audio"] };
    await act(async () => renderer.update(<Form />));
    await send();
    expect(submit).toHaveBeenCalledWith(
      "Hello",
      expect.objectContaining({
        attachments: [
          { filename: "notes.md", data: new Uint8Array([35, 32, 65]) },
        ],
      }),
    );
  },
);

it("allows Markdown in a new chat and a text-only history", async () => {
  state.messages = [
    { role: "user", attachments: [{ filename: "earlier.md" }] },
  ];
  await render();
  const markdown = { filename: "notes.md", data: new Uint8Array([35, 32, 65]) };
  await act(async () => receive([markdown], []));
  await enterHello();
  await send();
  expect(submit).toHaveBeenCalledWith(
    "Hello",
    expect.objectContaining({ attachments: [markdown] }),
  );
});

it("lets editing remove incompatible media and truncates later history", async () => {
  state.messages = [
    {
      role: "user",
      content: "Remove this audio",
      attachments: [{ filename: "speech.mp3" }],
    },
    { role: "user", attachments: [{ filename: "later.png" }] },
  ];
  await render();
  state.editingMessage = {
    content: "Hello",
    index: 0,
    originalMessage: state.messages[0],
  };
  await act(async () => renderer.update(<Form />));
  await act(async () =>
    renderer.root
      .findByProps({ "aria-label": "Remove speech.mp3" })
      .props.onClick(),
  );
  await send();
  expect(submit).toHaveBeenCalledWith(
    "Hello",
    expect.objectContaining({ attachments: [] }),
  );
});
