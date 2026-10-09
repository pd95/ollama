// Public backend labels describe inputs separately from image generation.
export const capabilityLabels = [
  { capability: "completion", label: "Text input" },
  { capability: "vision", label: "Image input" },
  { capability: "audio", label: "Audio input" },
  { capability: "tools", label: "Tools" },
  { capability: "thinking", label: "Thinking" },
  { capability: "image", label: "Image generation" },
] as const;

export function parseCapabilities(value: unknown): string[] | undefined {
  return Array.isArray(value) && value.every((item) => typeof item === "string")
    ? value
    : undefined;
}

export function modelCapabilityKey(
  modelName: string | undefined,
  digest?: string,
) {
  return ["modelCapabilities", modelName, digest || null] as const;
}

export function capabilityCacheTime(
  modelName: string | undefined,
  digest?: string,
) {
  return digest && !modelName?.endsWith("cloud")
    ? 60 * 60 * 1000
    : 5 * 60 * 1000;
}
