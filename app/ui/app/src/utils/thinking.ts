export function supportsThinkingLevels(modelName: string | undefined): boolean {
  return modelName?.toLowerCase().startsWith("gpt-oss") ?? false;
}

export function supportsThinkingToggle(
  modelName: string | undefined,
): boolean {
  const name = modelName?.toLowerCase() ?? "";

  return name.startsWith("deepseek-v3.1") || name.startsWith("apertus-1.5");
}

export function resolveThinkingSetting(
  modelName: string | undefined,
  thinkingEnabled: boolean,
  thinkingLevel: string,
): boolean | string | undefined {
  if (supportsThinkingLevels(modelName)) return thinkingLevel;
  if (supportsThinkingToggle(modelName)) return thinkingEnabled;
  return undefined;
}
