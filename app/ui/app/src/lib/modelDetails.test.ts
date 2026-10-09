import { expect, it } from "vitest";
import {
  formatModelParameterSize,
  modelRuntimeBackend,
  parseModelMetadata,
} from "./modelDetails";

it("formats raw parameter counts in compact decimal units and preserves model labels", () => {
  expect(formatModelParameterSize("32682372656")).toBe("32.7B");
  expect(formatModelParameterSize("31000000000")).toBe("31B");
  expect(formatModelParameterSize("1500000000")).toBe("1.5B");
  expect(formatModelParameterSize("500000000")).toBe("500M");
  expect(formatModelParameterSize("1200000000000")).toBe("1.2T");
  expect(formatModelParameterSize("8.1B")).toBe("8.1B");
  expect(formatModelParameterSize("8x7B")).toBe("8x7B");
  expect(formatModelParameterSize(undefined)).toBeUndefined();
  expect(formatModelParameterSize("0")).toBeUndefined();
});

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
