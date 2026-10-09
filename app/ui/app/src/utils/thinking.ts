export function supportsThinkingLevels(modelName: string | undefined): boolean {
  return modelName?.toLowerCase().startsWith("gpt-oss") ?? false;
}

export function supportsThinkingToggle(modelName: string | undefined): boolean {
  const name = modelName?.toLowerCase() ?? "";

  return name.startsWith("deepseek-v3.1") || name.startsWith("apertus-1.5");
}

export type ThinkingValue = boolean | string;
export interface ThinkingControls {
  values: ThinkingValue[];
  default: ThinkingValue;
}

// Treat absent or malformed discovery as unknown, preserving legacy handling.
export function parseThinkingControls(
  value: unknown,
): ThinkingControls | undefined {
  if (!value || typeof value !== "object") return undefined;
  const controls = value as Partial<ThinkingControls>;
  if (
    !Array.isArray(controls.values) ||
    controls.values.length === 0 ||
    !controls.values.every(
      (v) => typeof v === "boolean" || (typeof v === "string" && v.length > 0),
    ) ||
    !controls.values.includes(controls.default as ThinkingValue)
  )
    return undefined;
  return {
    values: [...new Set(controls.values)],
    default: controls.default as ThinkingValue,
  };
}

export function legacyThinkingControls(
  modelName: string | undefined,
  enabled: boolean,
  level: string,
): ThinkingControls | undefined {
  if (supportsThinkingLevels(modelName)) {
    const values = ["low", "medium", "high"];
    return { values, default: values.includes(level) ? level : "medium" };
  }
  if (supportsThinkingToggle(modelName))
    return { values: [false, true], default: enabled };
  return undefined;
}

export function thinkingLabel(value: ThinkingValue): string {
  if (typeof value === "boolean") return value ? "On" : "Off";
  return value.charAt(0).toUpperCase() + value.slice(1);
}

export function thinkingConflictsWithWebSearch(
  modelName: string | undefined,
  renderer?: string,
): boolean {
  return (
    supportsThinkingToggle(modelName) ||
    renderer === "deepseek3.1" ||
    renderer === "apertus1p5"
  );
}
