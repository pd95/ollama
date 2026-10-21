import {
  useState,
  useCallback,
  useRef,
  useEffect,
  useId,
  forwardRef,
  type JSX,
  type Ref,
  type KeyboardEvent,
  useImperativeHandle,
} from "react";
import { Model } from "@/gotypes";
import { useSelectedModel } from "@/hooks/useSelectedModel";
import { useCloudStatus } from "@/hooks/useCloudStatus";
import {
  useModelCapabilities,
  useModelCapabilitySummary,
} from "@/hooks/useModelCapabilities";
import { useQueryClient } from "@tanstack/react-query";
import { getModelUpstreamInfo } from "@/api";
import { capabilityLabels } from "@/lib/modelCapabilities";
import { planModelPickerScroll } from "@/lib/modelPickerScroll";
import {
  modelRuntimeBackend,
  formatModelFileSize,
  formatModelParameterSize,
} from "@/lib/modelDetails";
import {
  ArrowDownTrayIcon,
  CheckIcon,
  CloudIcon,
  InformationCircleIcon,
} from "@heroicons/react/24/outline";

const stalenessCheckCache = new Map<string, number>();
const panelClass =
  "absolute right-0 text-[15px] bottom-full mb-2 z-50 w-[360px] max-w-[calc(100vw-2rem)] rounded-2xl overflow-hidden bg-white border border-neutral-100 text-neutral-800 shadow-xl shadow-black/5 dark:border-neutral-600/40 dark:bg-neutral-800 dark:text-white";

type ModelListHandle = {
  scrollToSelectedModel: () => void;
  scrollToTop: () => void;
  handleKeyDown: (event: KeyboardEvent) => void;
};

export const ModelPicker = forwardRef<
  HTMLButtonElement,
  {
    chatId?: string;
    onModelSelect?: () => void;
    onEscape?: () => void;
    onDropdownToggle?: (isOpen: boolean) => void;
    isDisabled?: boolean;
    detailsButtonRef?: Ref<HTMLButtonElement>;
  }
>(function ModelPicker(
  {
    chatId,
    onModelSelect,
    onEscape,
    onDropdownToggle,
    isDisabled,
    detailsButtonRef,
  },
  ref,
): JSX.Element {
  const [panel, setPanel] = useState<"models" | "details" | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeOptionId, setActiveOptionId] = useState<string>();
  const isOpen = panel === "models";
  const { selectedModel, setSettings, models, loading } = useSelectedModel(
    chatId,
    searchQuery,
  );
  const { cloudDisabled } = useCloudStatus();
  const dropdownRef = useRef<HTMLDivElement>(null);
  const selectorRef = useRef<HTMLButtonElement>(null);
  const detailsRef = useRef<HTMLButtonElement>(null);
  const detailsPanelRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const modelListRef = useRef<ModelListHandle>(null);
  const queryClient = useQueryClient();
  const id = useId();
  const listId = `${id}-models`;
  const detailsId = `${id}-details`;

  useImperativeHandle(
    ref,
    () =>
      Object.assign(selectorRef.current!, {
        closeDropdown: () => setPanel(null),
      }),
    [],
  );
  useImperativeHandle(detailsButtonRef, () => detailsRef.current!, [
    selectedModel,
  ]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      )
        setPanel(null);
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    if (isOpen) {
      searchInputRef.current?.focus();
      modelListRef.current?.scrollToSelectedModel();
    } else {
      setSearchQuery("");
      setActiveOptionId(undefined);
    }
  }, [isOpen]);

  useEffect(() => {
    if (panel === "details") detailsPanelRef.current?.focus();
  }, [panel]);

  useEffect(() => {
    if (searchQuery) modelListRef.current?.scrollToTop();
  }, [searchQuery]);

  useEffect(() => {
    if (!selectedModel?.digest || loading) return;
    const model = selectedModel;
    const now = Date.now();
    const lastChecked = stalenessCheckCache.get(model.model);
    if (lastChecked && now - lastChecked < 5 * 60 * 1000) return;
    stalenessCheckCache.set(model.model, now);
    getModelUpstreamInfo(model)
      .then((upstreamInfo) => {
        if (upstreamInfo.stale) {
          const staleModels = new Map(
            queryClient.getQueryData<Map<string, boolean>>(["staleModels"]),
          );
          staleModels.set(model.model, true);
          queryClient.setQueryData(["staleModels"], staleModels);
        }
      })
      .catch((error) =>
        console.error("Failed to check model staleness:", error),
      );
  }, [selectedModel?.model, selectedModel?.digest, loading, queryClient]);

  const togglePanel = (next: "models" | "details") => {
    const opening = panel !== next;
    setPanel(opening ? next : null);
    onDropdownToggle?.(opening);
  };

  const handleModelSelect = (model: Model) => {
    setSettings({ SelectedModel: model.model });
    setPanel(null);
    onModelSelect?.();
  };

  return (
    <div
      className="relative flex min-w-0 items-center gap-1"
      ref={dropdownRef}
      onKeyDown={(event) => {
        if (event.key !== "Escape" || !panel) return;
        event.preventDefault();
        event.stopPropagation();
        setPanel(null);
        if (panel === "details") detailsRef.current?.focus();
        else if (onEscape) onEscape();
        else selectorRef.current?.focus();
      }}
    >
      <button
        ref={selectorRef}
        type="button"
        title="Select model"
        aria-label={`Select model${selectedModel ? `: ${selectedModel.model}` : ""}`}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-controls={isOpen ? listId : undefined}
        disabled={isDisabled}
        onClick={() => togglePanel("models")}
        onMouseDown={(event) => event.stopPropagation()}
        onDoubleClick={(event) => event.stopPropagation()}
        className="flex min-w-0 items-center select-none gap-1.5 rounded-full px-3.5 py-1.5 bg-white dark:bg-neutral-700 text-neutral-800 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:text-neutral-100 cursor-pointer disabled:cursor-default"
      >
        <span className="truncate">
          {isDisabled ? "Loading..." : selectedModel?.model || "Select a model"}
        </span>
        <svg
          className="h-3 w-3 shrink-0 opacity-70"
          aria-hidden="true"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>
      {selectedModel && (
        <button
          ref={detailsRef}
          type="button"
          aria-label={`Model information for ${selectedModel.model}`}
          aria-haspopup="dialog"
          aria-expanded={panel === "details"}
          aria-controls={panel === "details" ? detailsId : undefined}
          title="Model information"
          disabled={isDisabled}
          onClick={() => togglePanel("details")}
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-neutral-500 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer disabled:cursor-default"
        >
          <InformationCircleIcon className="h-4 w-4" aria-hidden="true" />
        </button>
      )}
      {panel === "details" && selectedModel && (
        <div
          id={detailsId}
          ref={detailsPanelRef}
          role="dialog"
          tabIndex={-1}
          aria-modal="false"
          aria-label={`Model information for ${selectedModel.model}`}
          className={`${panelClass} max-h-[min(32rem,calc(100vh-8rem))] overflow-y-auto p-4`}
        >
          <ModelDetails model={selectedModel} />
        </div>
      )}
      {isOpen && (
        <div className={panelClass}>
          <div className="px-1 py-2 border-b border-neutral-100 dark:border-neutral-700">
            <input
              ref={searchInputRef}
              type="text"
              role="combobox"
              aria-label="Find model"
              aria-autocomplete="list"
              aria-expanded="true"
              aria-controls={listId}
              aria-activedescendant={activeOptionId}
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              onKeyDown={(event) => modelListRef.current?.handleKeyDown(event)}
              placeholder="Find model..."
              autoCorrect="off"
              className="w-full px-2 py-0.5 bg-transparent border-none rounded-md outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <ModelList
            ref={modelListRef}
            models={models}
            selectedModel={selectedModel}
            onModelSelect={handleModelSelect}
            cloudDisabled={cloudDisabled}
            isOpen
            listId={listId}
            onActiveChange={setActiveOptionId}
          />
        </div>
      )}
    </div>
  );
});

function ModelInfoStatus({ reported = false }: { reported?: boolean }) {
  return (
    <>
      {reported ? (
        <CheckIcon className="h-4 w-4" aria-hidden="true" />
      ) : (
        <span aria-hidden="true">–</span>
      )}
      <span className="sr-only">{reported ? "Reported" : "Not reported"}</span>
    </>
  );
}

function ModelDetails({ model }: { model: Model }) {
  const query = useModelCapabilities(model.model);
  const capabilities = model.capabilities ?? query.data?.capabilities;
  const metadata = { ...model.metadata, ...query.data?.metadata };
  const remoteHost = model.remoteHost ?? query.data?.remoteHost;
  const remote = model.isCloud() || !!remoteHost;
  const contextLimit = metadata.contextLength?.toLocaleString("en-US");
  const details = [
    { label: "Runtime backend", value: modelRuntimeBackend(metadata, remote) },
    {
      label: "Location",
      value: model.isCloud()
        ? "Cloud"
        : remoteHost
          ? "Remote server"
          : model.digest
            ? "Local"
            : "Not installed",
    },
    {
      label: "File size",
      value: remote ? undefined : formatModelFileSize(model.size),
    },
    {
      label: "Parameters",
      value: formatModelParameterSize(metadata.parameterSize),
    },
    { label: "Quantization", value: metadata.quantization },
    {
      label: "Context limit",
      value: contextLimit ? `${contextLimit} tokens` : undefined,
    },
  ];
  return (
    <>
      <p className="mb-3 break-all font-medium">{model.model}</p>
      <dl className="space-y-1 text-sm">
        {details.map(({ label, value }) => (
          <div key={label} className="flex justify-between gap-4">
            <dt className="shrink-0">{label}</dt>
            <dd className="text-right text-neutral-500 dark:text-neutral-400">
              {value ?? (query.isFetching ? "Loading…" : <ModelInfoStatus />)}
            </dd>
          </div>
        ))}
      </dl>
      {contextLimit && (
        <p className="mt-2 text-xs text-neutral-500 dark:text-neutral-400">
          Model context limit; the chat may use less.
        </p>
      )}
      <h2 className="mt-4 mb-2 text-sm font-medium">
        Capabilities reported by the model
      </h2>
      {capabilities === undefined && (
        <p
          role="status"
          className="mb-2 text-sm text-neutral-500 dark:text-neutral-400"
        >
          {query.isFetching ? "Loading capabilities…" : "Capabilities unknown"}
        </p>
      )}
      <dl className="space-y-1 text-sm">
        {capabilityLabels.map(({ capability, label }) => (
          <div key={capability} className="flex justify-between gap-4">
            <dt>{label}</dt>
            <dd className="flex items-center text-neutral-500 dark:text-neutral-400">
              {capabilities === undefined ? (
                "Unknown"
              ) : (
                <ModelInfoStatus reported={capabilities.includes(capability)} />
              )}
            </dd>
          </div>
        ))}
      </dl>
      {(query.isError || (capabilities === undefined && !query.isFetching)) && (
        <button
          type="button"
          onClick={() => void query.refetch()}
          className="mt-3 text-sm text-blue-600 dark:text-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          Retry discovery
        </button>
      )}
    </>
  );
}

export const ModelList = forwardRef<
  ModelListHandle,
  {
    models: Model[];
    selectedModel: Model | null;
    onModelSelect: (model: Model) => void;
    cloudDisabled: boolean;
    isOpen: boolean;
    listId?: string;
    onActiveChange?: (id: string | undefined) => void;
  }
>(function ModelList(
  {
    models,
    selectedModel,
    onModelSelect,
    cloudDisabled,
    isOpen,
    listId: providedListId,
    onActiveChange,
  },
  ref,
): JSX.Element {
  const id = useId();
  const listId = providedListId || `${id}-models`;
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [scrolledFromTop, setScrolledFromTop] = useState(false);
  const [highlightedName, setHighlightedName] = useState<string>();
  const highlightedIndex = Math.max(
    0,
    models.findIndex(
      (model) => model.model === (highlightedName ?? selectedModel?.model),
    ),
  );
  const activeIndexRef = useRef(highlightedIndex);

  const scrollToItem = useCallback((index: number) => {
    const container = scrollContainerRef.current;
    if (!container) return;
    const containerTop = container.getBoundingClientRect().top;
    const rows = Array.from(container.children).map((item) => {
      const bounds = item.getBoundingClientRect();
      return {
        top: bounds.top - containerTop + container.scrollTop,
        bottom: bounds.bottom - containerTop + container.scrollTop,
      };
    });
    const plan = planModelPickerScroll(
      rows,
      index,
      container.scrollTop,
      container.clientHeight,
    );
    container.style.paddingBottom = `${plan.bottomPadding}px`;
    container.scrollTop = plan.scrollTop;
    setScrolledFromTop(container.scrollTop > 0);
  }, []);

  useEffect(() => {
    activeIndexRef.current = highlightedIndex;
    onActiveChange?.(
      models.length ? `${listId}-option-${highlightedIndex}` : undefined,
    );
  }, [models.length, highlightedIndex, listId, onActiveChange]);

  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container || !isOpen || typeof ResizeObserver === "undefined") return;
    const sizeKey = () =>
      [
        container.clientWidth,
        container.clientHeight,
        ...Array.from(container.children).map(
          (item) => item.getBoundingClientRect().height,
        ),
      ].join(",");
    let previousSize = sizeKey();
    const observer = new ResizeObserver(() => {
      const size = sizeKey();
      if (size === previousSize) return;
      previousSize = size;
      const top = container.getBoundingClientRect().top;
      const items = Array.from(container.children);
      const active = items[activeIndexRef.current]?.getBoundingClientRect();
      // Preserve the user's browsing position if the active model is offscreen.
      const visibleIndex =
        active &&
        active.bottom > top &&
        active.top < top + container.clientHeight
          ? activeIndexRef.current
          : items.findIndex(
              (item) => item.getBoundingClientRect().bottom > top,
            );
      scrollToItem(Math.max(0, visibleIndex));
    });
    observer.observe(container);
    Array.from(container.children).forEach((item) => observer.observe(item));
    return () => observer.disconnect();
  }, [models, isOpen, scrollToItem]);

  const handleKeyDown = (event: KeyboardEvent) => {
    if (!isOpen || !models.length) return;
    if (event.key === "Enter") {
      event.preventDefault();
      event.stopPropagation();
      onModelSelect(models[highlightedIndex]);
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const offset = event.key === "ArrowDown" ? 1 : -1;
      const next = (highlightedIndex + offset + models.length) % models.length;
      activeIndexRef.current = next;
      setHighlightedName(models[next].model);
      scrollToItem(next);
    }
  };

  useImperativeHandle(ref, () => ({
    scrollToSelectedModel: () => scrollToItem(highlightedIndex),
    scrollToTop: () => {
      if (scrollContainerRef.current) {
        scrollContainerRef.current.style.paddingBottom = "";
        scrollContainerRef.current.scrollTop = 0;
      }
      setScrolledFromTop(false);
    },
    handleKeyDown,
  }));

  return (
    <div className="relative">
      <div
        ref={scrollContainerRef}
        id={listId}
        role="listbox"
        aria-label="Models"
        onKeyDown={handleKeyDown}
        onScroll={(event) =>
          setScrolledFromTop(event.currentTarget.scrollTop > 0)
        }
        className="max-h-80 overflow-y-auto overflow-x-hidden"
      >
        {models.length === 0 ? (
          <div className="px-3 py-2 text-neutral-500 dark:text-neutral-400">
            No models found
          </div>
        ) : (
          models.map((model, index) => (
            <ModelOption
              key={`${model.model}-${model.digest || "no-digest"}`}
              id={`${listId}-option-${index}`}
              model={model}
              selected={selectedModel?.model === model.model}
              highlighted={highlightedIndex === index}
              cloudDisabled={cloudDisabled}
              isOpen={isOpen}
              scrollRoot={scrollContainerRef}
              onSelect={() => onModelSelect(model)}
              onHighlight={() => setHighlightedName(model.model)}
            />
          ))
        )}
      </div>
      {scrolledFromTop && (
        <div
          aria-hidden={true}
          className="pointer-events-none absolute inset-x-0 top-0 h-2 bg-gradient-to-b from-black/10 to-transparent dark:from-black/30"
        />
      )}
    </div>
  );
});

function ModelOption({
  id,
  model,
  selected,
  highlighted,
  cloudDisabled,
  isOpen,
  scrollRoot,
  onSelect,
  onHighlight,
}: {
  id: string;
  model: Model;
  selected: boolean;
  highlighted: boolean;
  cloudDisabled: boolean;
  isOpen: boolean;
  scrollRoot: { current: HTMLDivElement | null };
  onSelect: () => void;
  onHighlight: () => void;
}) {
  const rowRef = useRef<HTMLButtonElement>(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    if (!rowRef.current || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      ([entry]) => setVisible(entry.isIntersecting),
      { root: scrollRoot.current },
    );
    observer.observe(rowRef.current);
    return () => observer.disconnect();
  }, [scrollRoot]);
  const { capabilities, isLoading } = useModelCapabilitySummary(
    model,
    isOpen && (visible || highlighted),
  );
  const badges = capabilityLabels.filter(({ capability }) =>
    capabilities?.includes(capability),
  );
  return (
    <button
      ref={rowRef}
      id={id}
      type="button"
      role="option"
      aria-selected={selected}
      tabIndex={-1}
      onClick={onSelect}
      onMouseEnter={onHighlight}
      className={`block w-full px-3 py-2 text-left hover:bg-neutral-100 dark:hover:bg-neutral-700/60 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500 cursor-pointer ${highlighted ? "bg-neutral-100 dark:bg-neutral-700/60" : ""}`}
    >
      <span className="flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate" title={model.model}>
          {model.model}
        </span>
        {model.isCloud() && (
          <CloudIcon
            className="h-4 w-4 shrink-0 text-neutral-500 dark:text-neutral-400"
            aria-label="Cloud model"
            aria-hidden={false}
            role="img"
          />
        )}
        {model.digest === undefined && (cloudDisabled || !model.isCloud()) && (
          <ArrowDownTrayIcon
            className="h-4 w-4 shrink-0 text-neutral-500 dark:text-neutral-400"
            aria-label="Download required"
            aria-hidden={false}
            role="img"
          />
        )}
        <span className="h-4 w-4 shrink-0">
          {selected && <CheckIcon className="h-4 w-4" aria-hidden="true" />}
        </span>
      </span>
      <span className="mt-1 flex flex-wrap gap-x-2 gap-y-1 text-[11px] leading-4 text-neutral-600 dark:text-neutral-300">
        {capabilities === undefined ? (
          <span>
            {isLoading ? "Loading capabilities…" : "Capabilities unknown"}
          </span>
        ) : badges.length ? (
          badges.map(({ capability, label }, index) => (
            <span
              key={capability}
              className="inline-flex items-center gap-2 whitespace-nowrap"
            >
              {index > 0 && <span aria-hidden="true">·</span>}
              {label}
            </span>
          ))
        ) : (
          <span>
            {capabilities.length
              ? "Other capabilities advertised"
              : "No advertised capabilities"}
          </span>
        )}
      </span>
    </button>
  );
}
