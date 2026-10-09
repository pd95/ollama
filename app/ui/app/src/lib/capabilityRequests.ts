type WaitingRequest = { start: () => void };
const waiting: WaitingRequest[] = [];
let active = 0;

function drain() {
  while (active < 4 && waiting.length) waiting.shift()!.start();
}

function acquire(signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    signal?.throwIfAborted();
    const onAbort = () => {
      const index = waiting.indexOf(request);
      if (index !== -1) waiting.splice(index, 1);
      reject(signal!.reason);
    };
    const request = {
      start: () => {
        signal?.removeEventListener("abort", onAbort);
        active++;
        resolve();
      },
    };
    signal?.addEventListener("abort", onAbort, { once: true });
    waiting.push(request);
    drain();
  });
}

export class CapabilityDiscoveryError extends Error {
  status: number;
  constructor(status: number) {
    super(`Failed to fetch model capabilities: ${status}`);
    this.status = status;
  }
}

// A separate controller per request avoids aborting other SDK requests. The
// deadline starts when a slot is acquired, so queued rows do not time out early.
export async function withCapabilityDiscovery<T>(
  request: (signal: AbortSignal) => Promise<T>,
  signal?: AbortSignal,
): Promise<T> {
  await acquire(signal);
  const controller = new AbortController();
  const onAbort = () => controller.abort(signal!.reason);
  signal?.addEventListener("abort", onAbort, { once: true });
  if (signal?.aborted) onAbort();
  const timeout = setTimeout(
    () =>
      controller.abort(
        new DOMException("Capability discovery timed out", "TimeoutError"),
      ),
    10_000,
  );
  try {
    return await request(controller.signal);
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener("abort", onAbort);
    active--;
    drain();
  }
}

export function retryCapabilityDiscovery(failureCount: number, error: Error) {
  if (error.name === "AbortError") return false;
  if (
    error instanceof CapabilityDiscoveryError &&
    error.status < 500 &&
    error.status !== 408 &&
    error.status !== 429
  )
    return false;
  return failureCount < 1;
}
