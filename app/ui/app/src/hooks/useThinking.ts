import { useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useModelCapabilities } from "./useModelCapabilities";
import {
  legacyThinkingControls,
  thinkingConflictsWithWebSearch,
  type ThinkingValue,
} from "@/utils/thinking";

export function useThinking(
  modelName: string | undefined,
  legacyEnabled: boolean,
  legacyLevel: string,
) {
  const { data, isPending } = useModelCapabilities(modelName);
  const controls =
    data?.thinking ??
    legacyThinkingControls(modelName, legacyEnabled, legacyLevel);
  const queryClient = useQueryClient();
  // Keep explicit choices across navigation and new-chat creation. A fresh app
  // session starts from backend defaults, rather than legacy global settings.
  const { data: selection } = useQuery({
    queryKey: ["thinkingSelection"],
    queryFn: () => ({ value: undefined as ThinkingValue | undefined }),
    initialData: { value: undefined as ThinkingValue | undefined },
    enabled: false,
    gcTime: Infinity,
  });
  const selectedValue = selection.value;
  const value = controls?.values.includes(selectedValue as ThinkingValue)
    ? selectedValue
    : controls?.default;

  // Discard incompatible choices so revisiting a model uses its default.
  useEffect(() => {
    if (
      selectedValue !== undefined &&
      !isPending &&
      !controls?.values.includes(selectedValue)
    ) {
      queryClient.setQueryData(["thinkingSelection"], { value: undefined });
    }
  }, [controls, isPending, selectedValue, queryClient]);

  return {
    controls,
    value,
    setValue: (value: ThinkingValue) =>
      queryClient.setQueryData(["thinkingSelection"], { value }),
    usesMetadata: data?.thinking !== undefined,
    conflictsWithWebSearch: thinkingConflictsWithWebSearch(
      modelName,
      data?.renderer,
    ),
  };
}
