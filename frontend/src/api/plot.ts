import type {
  AnnotationUpdate,
  AnnotationsResponse,
  ClusterDetailResponse,
  IndexStartResponse,
  MessageResponse,
  PlotRequest,
  PlotResponse,
  SearchRequest,
  SearchResponse,
  SubClusterRequest,
  SubClusterResponse,
  SuggestClustersRequest,
  SuggestClustersStatusResponse,
  SuggestKRequest,
  SuggestKResponse,
} from "../types";
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

export async function getClusterDetail(
  jobId: string,
  clusterIndex: number,
  page = 1,
  pageSize = 50,
): Promise<ClusterDetailResponse> {
  return apiFetch<ClusterDetailResponse>(
    `/plot/${jobId}/cluster/${clusterIndex}?page=${page}&page_size=${pageSize}`,
  );
}

export async function subCluster(
  jobId: string,
  clusterIndex: number,
  request: SubClusterRequest,
): Promise<SubClusterResponse> {
  return apiPost<SubClusterResponse>(
    `/plot/${jobId}/cluster/${clusterIndex}/sub-cluster`,
    request,
  );
}

export async function subClusterByPointIds(
  jobId: string,
  request: SubClusterRequest,
): Promise<SubClusterResponse> {
  return apiPost<SubClusterResponse>(
    `/plot/${jobId}/sub-cluster`,
    request,
  );
}

export async function suggestK(
  jobId: string,
  request: SuggestKRequest,
): Promise<SuggestKResponse> {
  return apiPost<SuggestKResponse>(
    `/plot/${jobId}/suggest-k`,
    request,
  );
}

export async function getAnnotations(
  jobId: string,
): Promise<AnnotationsResponse> {
  return apiFetch<AnnotationsResponse>(`/annotations/${jobId}`);
}

export async function updateAnnotation(
  jobId: string,
  clusterIndex: number,
  body: AnnotationUpdate,
): Promise<AnnotationsResponse> {
  return apiFetch<AnnotationsResponse>(
    `/annotations/${jobId}/cluster/${clusterIndex}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
}

export async function deleteAnnotations(
  jobId: string,
): Promise<MessageResponse> {
  return apiFetch<MessageResponse>(`/annotations/${jobId}`, {
    method: "DELETE",
  });
}
