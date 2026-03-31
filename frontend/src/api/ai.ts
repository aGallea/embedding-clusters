import type {
  AiNamingRequest,
  AiNamingResponse,
  AiSubClusterNamingRequest,
  AiTestConnectionRequest,
  AiTestConnectionResponse,
  OllamaModelsResponse,
} from "../types";
import { apiPost } from "./client";

const AI_SETTINGS_KEY = "ai-cluster-naming-settings";

export const AI_PROVIDERS = [
  { value: "openai", label: "OpenAI", defaultBaseUrl: "" },
  { value: "google", label: "Google", defaultBaseUrl: "" },
  { value: "anthropic", label: "Anthropic", defaultBaseUrl: "" },
  { value: "ollama", label: "Ollama", defaultBaseUrl: "http://localhost:11434" },
] as const;

export type AiProvider = (typeof AI_PROVIDERS)[number]["value"];

export interface StoredAiSettings {
  provider: AiProvider;
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
  request: AiSubClusterNamingRequest,
): Promise<AiNamingResponse> {
  return apiPost<AiNamingResponse>("/ai/name-sub-clusters", request);
}

export async function fetchOllamaModels(
  baseUrl: string = "http://localhost:11434",
): Promise<OllamaModelsResponse> {
  return apiPost<OllamaModelsResponse>("/ai/ollama/models", {
    base_url: baseUrl,
  });
}
