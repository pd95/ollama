import { describe, expect, it } from "vitest";
import {
  legacyThinkingControls,
  parseThinkingControls,
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
    expect(supportsThinkingToggle("apertus-1.5-mlx:8b-nvfp4-media")).toBe(true);
    expect(supportsThinkingToggle("Apertus-1.5:8b")).toBe(true);
  });

  it("does not enable the toggle for legacy Apertus", () => {
    expect(supportsThinkingToggle("apertus-mlx:8b-nvfp4")).toBe(false);
  });

  it("sends an explicit false value when Apertus 1.5 thinking is disabled", () => {
    expect(
      legacyThinkingControls("apertus-1.5-mlx:8b-nvfp4", false, "medium")
        ?.default,
    ).toBe(false);
  });
});

it.each([
  undefined,
  null,
  {},
  { values: [], default: false },
  { values: [false, 1], default: false },
  { values: [false, true], default: "high" },
  { values: [""], default: "" },
])("rejects malformed or absent thinking metadata %j", (value) => {
  expect(parseThinkingControls(value)).toBeUndefined();
});

it("accepts mixed values and boolean false defaults without coercion", () => {
  expect(
    parseThinkingControls({ values: [false, "none", "xhigh"], default: false }),
  ).toEqual({ values: [false, "none", "xhigh"], default: false });
});
