// Collections
export interface CollectionInfo {
  name: string;
  count: number;
}

export interface CollectionDetail {
  name: string;
  count: number;
  metadata_fields: string[];
}

// CSV
export interface CsvUploadResponse {
  filename: string;
  rows: number;
  columns: string[];
}

export interface CsvPreviewResponse {
  columns: string[];
  rows: Record<string, string>[];
  total_rows: number;
}

// Indexing
export interface IndexRequest {
  csv_filename: string;
  id_field?: string;
  image_embedding_fields?: string[];
  text_embedding_fields?: string[];
  image_model_name?: string;
  text_model_name?: string;
  chromadb_collection_prefix?: string;
  number_of_async_tasks?: number;
  index_bulk_size?: number;
  index_start_line?: number;
  index_end_line?: number;
  process_unit_device?: string;
  embedding_fields_prefix?: string;
  total_rows?: number;
}

export interface IndexStartResponse {
  job_id: string;
  status: string;
}

export interface IndexStatusResponse {
  job_id: string;
  status: string;
  rows_indexed: number;
  total_rows: number | null;
  errors: number;
  error: string | null;
}

// Plot
export interface PlotRequest {
  chromadb_collection_name: string;
  num_clusters?: number;
  text_display_fields?: string[];
  image_field?: string;
  gpt_generate_cluster_name?: boolean;
  gpt_default_model?: string;
  gpt_default_temperature?: number;
}

export interface PlotPoint {
  x: number;
  y: number;
  z: number;
  cluster: number;
  metadata: Record<string, unknown>;
  id: string;
}

export interface PlotCluster {
  index: number;
  name: string;
  color: string;
  count: number;
}

export interface PlotResponse {
  points: PlotPoint[];
  clusters: PlotCluster[];
  total_points: number;
}

// Messages
export interface MessageResponse {
  message: string;
}

// Search
export interface SearchResult {
  id: string;
  distance: number;
  metadata: Record<string, unknown>;
}

export interface SearchRequest {
  collection_name: string;
  query_text?: string;
  query_image_url?: string;
  n_results?: number;
  model_type?: string;
  image_model_name?: string;
  text_model_name?: string;
}

export interface SearchResponse {
  results: SearchResult[];
}
