import { useQuery } from "@tanstack/react-query";
import { getModelCapabilities } from "@/api";
import type { ModelCapabilityDetails } from "@/api";
import type { Model } from "@/gotypes";
import {
  modelCapabilityKey,
  capabilityCacheTime,
} from "@/lib/modelCapabilities";
import { retryCapabilityDiscovery } from "@/lib/capabilityRequests";
import { useModels } from "./useModels";

function discoveryOptions(modelName: string | undefined, digest?: string) {
  const cacheTime = capabilityCacheTime(modelName, digest);
  return {
    queryKey: modelCapabilityKey(modelName, digest),
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getModelCapabilities(modelName!, signal),
    gcTime: cacheTime,
    staleTime: cacheTime,
    retry: retryCapabilityDiscovery,
    retryDelay: 1000,
  };
}

export function useModelCapabilities(modelName: string | undefined) {
  const { data: models } = useModels();
  const model = models?.find((item) => item.model === modelName);
  return useQuery<ModelCapabilityDetails, Error>({
    ...discoveryOptions(modelName, model?.digest),
    enabled: !!modelName,
  });
}

export function useModelCapabilitySummary(
  model: Model | null,
  enabled: boolean,
) {
  const query = useQuery<ModelCapabilityDetails, Error>({
    ...discoveryOptions(model?.model, model?.digest),
    enabled: !!model && enabled && model.capabilities === undefined,
  });
  const capabilities = model?.capabilities ?? query.data?.capabilities;
  return {
    capabilities,
    isLoading: capabilities === undefined && query.isFetching,
    refetch: query.refetch,
  };
}

export function useHasVisionCapability(modelName: string | undefined) {
  const { data: capabilitiesResponse } = useModelCapabilities(modelName);
  return capabilitiesResponse?.capabilities?.includes("vision") ?? false;
}

export function useHasToolsCapability(modelName: string | undefined) {
  const { data: capabilitiesResponse } = useModelCapabilities(modelName);
  return capabilitiesResponse?.capabilities?.includes("tools") ?? false;
}
