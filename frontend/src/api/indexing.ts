import {
  IndexRequest,
  IndexStartResponse,
  IndexStatusResponse,
  MessageResponse,
} from "../types";
import { apiFetch, apiPost } from "./client";

export async function startIndex(
  request: IndexRequest,
): Promise<IndexStartResponse> {
  return apiPost<IndexStartResponse>("/index/start", request);
}

export async function getIndexStatus(
  jobId: string,
): Promise<IndexStatusResponse> {
  return apiFetch<IndexStatusResponse>(`/index/status/${jobId}`);
}

export async function cancelIndex(jobId: string): Promise<MessageResponse> {
  return apiPost<MessageResponse>(`/index/cancel/${jobId}`, {});
}

export function createIndexWebSocket(jobId: string): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  return new WebSocket(`${protocol}//${host}/api/index/ws/${jobId}`);
}
