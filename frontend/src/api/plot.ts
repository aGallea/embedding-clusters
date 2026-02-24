import type { IndexStartResponse, PlotRequest, PlotResponse } from "../types";
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
