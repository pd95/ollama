import { afterEach, expect, it, vi } from "vitest";
import {
  withCapabilityDiscovery,
  CapabilityDiscoveryError,
  retryCapabilityDiscovery,
} from "./capabilityRequests";
import {
  parseCapabilities,
  modelCapabilityKey,
  capabilityCacheTime,
} from "./modelCapabilities";

afterEach(() => vi.useRealTimers());

it("distinguishes missing/malformed metadata from an advertised empty list", () => {
  expect(parseCapabilities(undefined)).toBeUndefined();
  expect(parseCapabilities("vision")).toBeUndefined();
  expect(parseCapabilities(["vision", false])).toBeUndefined();
  expect(parseCapabilities([])).toEqual([]);
  expect(parseCapabilities(["vision", "audio", "future"])).toEqual([
    "vision",
    "audio",
    "future",
  ]);
});

it("keeps aliases, local/cloud tags, and replacement digests separate", () => {
  const keys = [
    modelCapabilityKey("alias", "one"),
    modelCapabilityKey("alias", "two"),
    modelCapabilityKey("alias:cloud", "one"),
    modelCapabilityKey("alias"),
  ];
  expect(new Set(keys.map((key) => JSON.stringify(key))).size).toBe(4);
  expect(capabilityCacheTime("alias", "one")).toBe(3_600_000);
  expect(capabilityCacheTime("alias:cloud", "one")).toBe(300_000);
  expect(capabilityCacheTime("alias")).toBe(300_000);
});

it("limits active requests to four and starts queued work as slots finish", async () => {
  const finishes: (() => void)[] = [];
  let active = 0;
  let peak = 0;
  const requests = Array.from({ length: 9 }, () =>
    withCapabilityDiscovery(async () => {
      active++;
      peak = Math.max(peak, active);
      await new Promise<void>((resolve) => finishes.push(resolve));
      active--;
    }),
  );
  await vi.waitFor(() => expect(finishes).toHaveLength(4));
  while (finishes.length) {
    finishes.shift()!();
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  await Promise.all(requests);
  expect(peak).toBe(4);
  expect(active).toBe(0);
});

it("aborts a timed-out request and releases its slot", async () => {
  vi.useFakeTimers();
  const request = withCapabilityDiscovery(
    (signal) =>
      new Promise((_, reject) =>
        signal.addEventListener("abort", () => reject(signal.reason)),
      ),
  );
  const result = expect(request).rejects.toMatchObject({
    name: "TimeoutError",
  });
  await vi.advanceTimersByTimeAsync(10_000);
  await result;
  await expect(withCapabilityDiscovery(async () => "next")).resolves.toBe(
    "next",
  );
});

it("removes a cancelled queued request without starting it", async () => {
  const finishes: (() => void)[] = [];
  const running = Array.from({ length: 4 }, () =>
    withCapabilityDiscovery(
      async () => new Promise<void>((resolve) => finishes.push(resolve)),
    ),
  );
  await vi.waitFor(() => expect(finishes).toHaveLength(4));
  const controller = new AbortController();
  const task = vi.fn();
  const queued = withCapabilityDiscovery(task, controller.signal);
  const result = expect(queued).rejects.toMatchObject({ name: "AbortError" });
  controller.abort();
  await result;
  finishes.forEach((finish) => finish());
  await Promise.all(running);
  expect(task).not.toHaveBeenCalled();
});

it("retries transient failures once and never retries auth/not-found errors", () => {
  for (const status of [400, 401, 403, 404])
    expect(
      retryCapabilityDiscovery(0, new CapabilityDiscoveryError(status)),
    ).toBe(false);
  for (const status of [408, 429, 500, 503]) {
    expect(
      retryCapabilityDiscovery(0, new CapabilityDiscoveryError(status)),
    ).toBe(true);
    expect(
      retryCapabilityDiscovery(1, new CapabilityDiscoveryError(status)),
    ).toBe(false);
  }
  expect(retryCapabilityDiscovery(0, new TypeError("offline"))).toBe(true);
  expect(
    retryCapabilityDiscovery(0, new DOMException("cancelled", "AbortError")),
  ).toBe(false);
});
