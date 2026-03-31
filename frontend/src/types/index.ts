// Collections
export interface CollectionInfo {
  name: string;
  count: number;
  model_name: string | null;
  model_type: string | null;
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
export type ReductionAlgorithm = 'tsne' | 'umap' | 'pca'

export interface PlotRequest {
  chromadb_collection_name: string
  num_clusters?: number
  text_display_fields?: string[]
  image_field?: string
  reduction_algorithm?: ReductionAlgorithm
  tsne_perplexity?: number
  tsne_learning_rate?: string
  umap_n_neighbors?: number
  umap_min_dist?: number
  umap_metric?: string
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
  job_id?: string;
}

// Cluster Suggestion
export interface SuggestClustersRequest {
  collection_name: string;
  k_min?: number;
  k_max?: number;
}

export interface SuggestClustersResponse {
  k_values: number[];
  inertias: number[];
  silhouette_scores: number[];
  suggested_k: number;
}

export interface SuggestClustersStatusResponse {
  status: string;
  ready: boolean;
  phase?: string;
  current_k?: number;
  total_k?: number;
  result?: SuggestClustersResponse;
  error?: string;
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

// Cluster Detail
export interface ClusterItemResponse {
  id: string;
  metadata: Record<string, unknown>;
  distance_to_centroid: number;
}

export interface ClusterDetailResponse {
  cluster_index: number;
  cluster_name: string;
  total_items: number;
  page: number;
  page_size: number;
  items: ClusterItemResponse[];
}

// Sub-Cluster
export interface SubClusterRequest {
  num_sub_clusters: number;
  point_ids?: string[];
}

export interface SubClusterPoint {
  id: string;
  x: number;
  y: number;
  z: number;
  sub_cluster: number;
  metadata: Record<string, unknown>;
}

export interface SubClusterInfo {
  index: number;
  count: number;
  color: string;
  name?: string;
}

export interface SubClusterResponse {
  parent_cluster_index: number;
  points: SubClusterPoint[];
  sub_clusters: SubClusterInfo[];
  total_points: number;
}

// Drill-down
export interface DrillLevel {
  label: string;
  pointIds: string[];
  subClusterData: SubClusterResponse;
}

// Suggest K
export interface SuggestKRequest {
  point_ids?: string[];
  cluster_index?: number;
  max_k?: number;
}

export interface SuggestKScoreEntry {
  k: number;
  score: number;
}

export interface SuggestKResponse {
  suggested_k: number;
  scores: SuggestKScoreEntry[];
}

// Annotations
export interface AnnotationUpdate {
  name?: string;
  notes?: string;
  tags?: string[];
}

export interface ClusterAnnotation {
  name?: string;
  notes?: string;
  tags?: string[];
  updated_at?: string;
}

export interface AnnotationsResponse {
  job_id: string;
  clusters: Record<string, ClusterAnnotation>;
}

// AI Naming
export interface AiSettings {
  provider: string;
  model: string;
  apiKey: string;
  baseUrl: string;
  temperature: number;
}

export interface AiNamingRequest {
  job_id: string;
  cluster_indices: number[];
  api_key: string;
  model: string;
  base_url?: string;
  temperature?: number;
}

export interface AiNamingResponse {
  names: Record<string, string>;
}

export interface AiSubClusterNamingRequest {
  job_id: string;
  point_ids: string[];
  sub_cluster_labels: number[];
  api_key: string;
  model: string;
  base_url?: string;
  temperature?: number;
  parent_cluster_name?: string;
}

export interface AiTestConnectionRequest {
  api_key: string;
  model: string;
  base_url?: string;
}

export interface AiTestConnectionResponse {
  success: boolean;
  error: string | null;
}

export interface OllamaModel {
  name: string;
  size: number | null;
  parameter_size: string | null;
  family: string | null;
}

export interface OllamaModelsResponse {
  models: OllamaModel[];
}
