import { expect, it } from "vitest";
import { modelRuntimeBackend, parseModelMetadata } from "./modelDetails";

it("uses the scheduler's format rules even for legacy runner labels", () => {
  expect(modelRuntimeBackend({ format: "safetensors", runner: "mlx" })).toBe(
    "MLX",
  );
  expect(modelRuntimeBackend({ format: "gguf", runner: "ggml" })).toBe(
    "llama.cpp",
  );
  expect(modelRuntimeBackend({ runner: "llamacpp" })).toBe("llama.cpp");
  expect(modelRuntimeBackend({ format: "gguf" }, true)).toBeUndefined();
  expect(modelRuntimeBackend({ runner: "future-runner" })).toBeUndefined();
  expect(
    modelRuntimeBackend({ format: "future-format", runner: "mlx" }),
  ).toBeUndefined();
});

it("does not invent details from malformed metadata or a different architecture", () => {
  expect(
    parseModelMetadata(
      { parameter_size: 8, quantization_level: null, context_length: -1 },
      { "other.context_length": 65536 },
    ),
  ).toBeUndefined();
  expect(
    parseModelMetadata(null, {
      "general.architecture": "llama",
      "llama.context_length": "65536",
    }),
  ).toEqual({ family: "llama" });
});
