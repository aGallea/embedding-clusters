import type { CsvPreviewResponse, CsvUploadResponse } from "../types";
import { apiFetch, apiPost } from "./client";

export async function uploadCsv(file: File): Promise<CsvUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return apiFetch<CsvUploadResponse>("/csv/upload", {
    method: "POST",
    body: formData,
  });
}

export async function previewCsv(
  filename: string,
  limit?: number,
): Promise<CsvPreviewResponse> {
  return apiPost<CsvPreviewResponse>("/csv/preview", {
    filename,
    limit: limit ?? 10,
  });
}
