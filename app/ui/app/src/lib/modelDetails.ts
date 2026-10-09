export type ModelMetadata = {
  format?: string;
  runner?: string;
  family?: string;
  parameterSize?: string;
  quantization?: string;
  contextLength?: number;
};

export function positiveNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value
    : undefined;
}

function object(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

export function metadataText(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

export function parseModelMetadata(
  details: unknown,
  modelInfo?: unknown,
): ModelMetadata | undefined {
  const source = object(details);
  const info = object(modelInfo);
  const architecture = metadataText(info["general.architecture"]);
  const values: ModelMetadata = {
    format: metadataText(source.format),
    runner: metadataText(source.runner),
    family: metadataText(source.family) ?? architecture,
    parameterSize: metadataText(source.parameter_size),
    quantization: metadataText(source.quantization_level),
    contextLength:
      positiveNumber(source.context_length) ??
      (architecture
        ? positiveNumber(info[`${architecture}.context_length`])
        : undefined),
  };
  const known = Object.entries(values).filter(
    ([, value]) => value !== undefined,
  );
  return known.length ? Object.fromEntries(known) : undefined;
}

export function modelRuntimeBackend(
  metadata?: ModelMetadata,
  remote = false,
): string | undefined {
  if (remote || !metadata) return undefined;
  // The local scheduler selects MLX for safetensors and llama-server for GGUF.
  // Legacy GGUF manifests can say "ggml" even when llama.cpp serves them.
  switch (metadata.format?.toLowerCase()) {
    case "safetensors":
      return "MLX";
    case "gguf":
      return "llama.cpp";
    case undefined:
    case "":
      if (metadata.runner?.toLowerCase() === "mlx") return "MLX";
      if (metadata.runner?.toLowerCase() === "llamacpp") return "llama.cpp";
  }
  return undefined;
}

export function formatModelFileSize(
  bytes: number | undefined,
): string | undefined {
  if (!positiveNumber(bytes)) return undefined;
  const units = ["B", "KB", "MB", "GB", "TB"];
  const unit = Math.max(
    0,
    Math.min(Math.floor(Math.log10(bytes!) / 3), units.length - 1),
  );
  return `${Number((bytes! / 1000 ** unit).toFixed(1))} ${units[unit]}`;
}

export function formatModelParameterSize(
  value: string | undefined,
): string | undefined {
  const text = metadataText(value);
  if (!text || !/^\d+(?:\.\d+)?$/.test(text)) return text;
  const count = positiveNumber(Number(text));
  return count === undefined
    ? undefined
    : new Intl.NumberFormat("en-US", {
        notation: "compact",
        maximumFractionDigits: 1,
      }).format(count);
}
