import { act, create, type ReactTestRenderer } from "react-test-renderer";
import { beforeEach, afterEach, expect, it, vi } from "vitest";
import { FileUpload } from "./FileUpload";

let renderer: ReactTestRenderer;
const added = vi.fn();
let paste: (event: any) => Promise<void>;
beforeEach(() => {
  added.mockReset();
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("document", {
    addEventListener: (event: string, handler: any) => {
      if (event === "paste") paste = handler;
    },
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
  vi.unstubAllGlobals();
});

it("rejects dropped image/audio for text-only models and preserves document bytes", async () => {
  await act(async () => {
    renderer = create(<FileUpload onFilesAdded={added}>Chat</FileUpload>);
  });
  await act(async () =>
    renderer.root.findAllByType("div")[0].props.onDrop({
      preventDefault: vi.fn(),
      stopPropagation: vi.fn(),
      dataTransfer: {
        types: ["Files"],
        files: [
          new File(["png"], "photo.png"),
          new File(["wav"], "speech.wav"),
          new File(["notes"], "notes.txt"),
        ],
      },
    }),
  );
  expect(added).toHaveBeenCalledWith(
    [
      {
        filename: "notes.txt",
        data: new TextEncoder().encode("notes"),
        type: undefined,
      },
    ],
    [
      { filename: "photo.png", error: "This model does not support images" },
      { filename: "speech.wav", error: "This model does not support audio" },
    ],
  );
});

it.each([false, true])(
  "uses independent audio permission for pasted files (%s)",
  async (audio) => {
    await act(async () => {
      renderer = create(
        <FileUpload
          onFilesAdded={added}
          hasVisionCapability
          hasAudioCapability={audio}
        >
          Chat
        </FileUpload>,
      );
    });
    const file = new File([new Uint8Array([1, 2, 3])], "speech.mp3", {
      type: "audio/mpeg",
    });
    await act(async () =>
      paste({
        preventDefault: vi.fn(),
        clipboardData: {
          files: [file],
          types: ["Files"],
          items: [{ kind: "file", getAsFile: () => file }],
          getData: () => "",
        },
      }),
    );
    if (audio) {
      expect(added).toHaveBeenCalledWith(
        [
          {
            filename: "speech.mp3",
            data: new Uint8Array([1, 2, 3]),
            type: "audio/mpeg",
          },
        ],
        [],
      );
    } else {
      expect(added).toHaveBeenCalledWith(
        [],
        [
          {
            filename: "speech.mp3",
            error: "This model does not support audio",
          },
        ],
      );
    }
  },
);
