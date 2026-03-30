import type { AiNamingRequest, AiNamingResponse, AiTestConnectionRequest, AiTestConnectionResponse } from "../types";
import { apiPost } from "./client";

const AI_SETTINGS_KEY = "ai-cluster-naming-settings";

export interface StoredAiSettings {
  provider: string;
  model: string;
  apiKey: string;
  baseUrl: string;
  temperature: number;
}

export const DEFAULT_AI_SETTINGS: StoredAiSettings = {
  provider: "openai",
  model: "gpt-4o-mini",
  apiKey: "",
  baseUrl: "",
  temperature: 0.5,
};

export function loadAiSettings(): StoredAiSettings {
  try {
    const raw = localStorage.getItem(AI_SETTINGS_KEY);
    if (!raw) return { ...DEFAULT_AI_SETTINGS };
    return { ...DEFAULT_AI_SETTINGS, ...JSON.parse(raw) } as StoredAiSettings;
  } catch {
    return { ...DEFAULT_AI_SETTINGS };
  }
}

export function saveAiSettings(settings: StoredAiSettings): void {
  localStorage.setItem(AI_SETTINGS_KEY, JSON.stringify(settings));
}

export async function testAiConnection(
  request: AiTestConnectionRequest,
): Promise<AiTestConnectionResponse> {
  return apiPost<AiTestConnectionResponse>("/ai/test-connection", request);
}

export async function nameAiClusters(
  request: AiNamingRequest,
): Promise<AiNamingResponse> {
  return apiPost<AiNamingResponse>("/ai/name-clusters", request);
}

export async function nameAiSubClusters(
  request: AiNamingRequest,
): Promise<AiNamingResponse> {
  return apiPost<AiNamingResponse>("/ai/name-sub-clusters", request);
}
