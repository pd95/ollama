import { expect, it } from "vitest";
import { planModelPickerScroll } from "./modelPickerScroll";

it("keeps whole leading rows and the active model visible across wrapped row heights", () => {
  const heights = [58, 58, 78, 58, 94, 58, 78];
  let top = 0;
  const rows = heights.map((height) => {
    const row = { top, bottom: top + height };
    top += height;
    return row;
  });
  for (const viewportHeight of [120, 240, 320]) {
    for (let active = 0; active < rows.length; active++) {
      for (const scrollTop of [0, 30, 100, 150]) {
        const plan = planModelPickerScroll(
          rows,
          active,
          scrollTop,
          viewportHeight,
        );
        expect(rows.some((row) => row.top === plan.scrollTop)).toBe(true);
        expect(rows[active].top).toBeGreaterThanOrEqual(plan.scrollTop);
        expect(rows[active].bottom).toBeLessThanOrEqual(
          plan.scrollTop + viewportHeight,
        );
        expect(plan.scrollTop + viewportHeight).toBeLessThanOrEqual(
          top + plan.bottomPadding,
        );
        expect(plan.bottomPadding).toBeLessThan(Math.max(...heights));
        expect(
          planModelPickerScroll(rows, active, plan.scrollTop, viewportHeight),
        ).toEqual(plan);
      }
    }
  }
});

it("needs no trailing space for a short list and prioritizes the name of an oversized row", () => {
  expect(planModelPickerScroll([{ top: 0, bottom: 58 }], 0, 0, 320)).toEqual({
    scrollTop: 0,
    bottomPadding: 0,
  });
  expect(planModelPickerScroll([{ top: 0, bottom: 150 }], 0, 30, 100)).toEqual({
    scrollTop: 0,
    bottomPadding: 0,
  });
});
