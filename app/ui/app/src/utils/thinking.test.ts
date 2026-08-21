import { describe, expect, it } from "vitest";
import {
  resolveThinkingSetting,
  supportsThinkingLevels,
  supportsThinkingToggle,
} from "./thinking";

describe("thinking controls", () => {
  it("uses level selection for GPT-OSS", () => {
    expect(supportsThinkingLevels("gpt-oss:20b")).toBe(true);
    expect(supportsThinkingToggle("gpt-oss:20b")).toBe(false);
  });

  it("keeps the existing DeepSeek thinking toggle", () => {
    expect(supportsThinkingToggle("deepseek-v3.1:671b")).toBe(true);
  });

  it("shows the toggle for Apertus 1.5 model names", () => {
    expect(supportsThinkingToggle("apertus-1.5-mlx:8b-nvfp4")).toBe(true);
    expect(
      supportsThinkingToggle("apertus-1.5-mlx:8b-nvfp4-media"),
    ).toBe(true);
    expect(supportsThinkingToggle("Apertus-1.5:8b")).toBe(true);
  });

  it("does not enable the toggle for legacy Apertus", () => {
    expect(supportsThinkingToggle("apertus-mlx:8b-nvfp4")).toBe(false);
  });

  it("sends an explicit false value when Apertus 1.5 thinking is disabled", () => {
    expect(
      resolveThinkingSetting("apertus-1.5-mlx:8b-nvfp4", false, "medium"),
    ).toBe(false);
  });
});
