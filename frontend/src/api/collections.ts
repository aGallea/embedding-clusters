import { CollectionDetail, CollectionInfo, MessageResponse } from "../types";
import { apiFetch } from "./client";

export async function fetchCollections(): Promise<CollectionInfo[]> {
  return apiFetch<CollectionInfo[]>("/collections");
}

export async function fetchCollection(name: string): Promise<CollectionDetail> {
  return apiFetch<CollectionDetail>(`/collections/${name}`);
}

export async function deleteCollection(name: string): Promise<MessageResponse> {
  return apiFetch<MessageResponse>(`/collections/${name}`, {
    method: "DELETE",
  });
}
