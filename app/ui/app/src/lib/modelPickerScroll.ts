export type ModelRowBounds = { top: number; bottom: number };

// Align programmatic scrolling to a model name, including at the list's end.
// Padding is derived from the rows alone so repeated measurements cannot grow it.
export function planModelPickerScroll(
  rows: ModelRowBounds[],
  activeIndex: number,
  scrollTop: number,
  viewportHeight: number,
) {
  const active = rows[activeIndex];
  if (!active || viewportHeight <= 0) return { scrollTop: 0, bottomPadding: 0 };

  const tolerance = 0.5;
  const lastBottom = rows[rows.length - 1].bottom;
  const finalStart = Math.max(0, lastBottom - viewportHeight);
  const finalRow = rows.find((row) => row.top >= finalStart - tolerance);
  const bottomPadding =
    lastBottom <= viewportHeight
      ? 0
      : Math.max(
          0,
          (finalRow?.top ?? finalStart) + viewportHeight - lastBottom,
        );

  let target = scrollTop;
  if (active.top < target || active.bottom - active.top > viewportHeight)
    target = active.top;
  else if (active.bottom > target + viewportHeight)
    target = active.bottom - viewportHeight;

  const leadingRow = rows.find((row) => row.top >= target - tolerance);
  return {
    scrollTop: Math.max(0, Math.min(leadingRow?.top ?? target, active.top)),
    bottomPadding,
  };
}
