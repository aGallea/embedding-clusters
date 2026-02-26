import type { IndexStartResponse, PlotRequest, PlotResponse, SearchRequest, SearchResponse, SuggestClustersRequest, SuggestClustersStatusResponse } from "../types";
import { apiFetch, apiPost } from "./client";

export async function startPlotCompute(
  request: PlotRequest,
): Promise<IndexStartResponse> {
  return apiPost<IndexStartResponse>("/plot/compute", request);
}

export async function getPlotData(
  jobId: string,
): Promise<PlotResponse & { status: string; ready: boolean }> {
  return apiFetch<PlotResponse & { status: string; ready: boolean }>(
    `/plot/data/${jobId}`,
  );
}

export async function searchCollection(
  request: SearchRequest,
): Promise<SearchResponse> {
  return apiPost<SearchResponse>("/search", request);
}

export async function suggestClusters(
  request: SuggestClustersRequest,
): Promise<IndexStartResponse> {
  return apiPost<IndexStartResponse>("/plot/suggest-clusters", request);
}

export async function getSuggestClustersStatus(
  jobId: string,
): Promise<SuggestClustersStatusResponse> {
  return apiFetch<SuggestClustersStatusResponse>(
    `/plot/suggest-clusters/${jobId}`,
  );
}
