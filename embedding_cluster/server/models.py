from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator


class CollectionInfo(BaseModel):
    name: str
    count: int
    model_name: str | None = None
    model_type: str | None = None


class CollectionDetail(BaseModel):
    name: str
    count: int
    metadata_fields: list[str]


class MessageResponse(BaseModel):
    message: str


class CsvUploadResponse(BaseModel):
    filename: str
    rows: int
    columns: list[str]


class CsvPreviewResponse(BaseModel):
    columns: list[str]
    rows: list[dict[str, str]]
    total_rows: int


class IndexRequest(BaseModel):
    csv_filename: str
    id_field: str | None = None
    image_embedding_fields: list[str] | None = None
    text_embedding_fields: list[str] | None = None
    image_model_name: str = "openai/clip-vit-base-patch32"
    text_model_name: str = "BAAI/bge-small-en-v1.5"
    chromadb_collection_prefix: str = ""
    number_of_async_tasks: int = 1
    index_bulk_size: int = 100
    index_start_line: int | None = None
    index_end_line: int | None = None
    process_unit_device: str = "cpu"
    embedding_fields_prefix: str = "embedding_"
    total_rows: int | None = None


class IndexStartResponse(BaseModel):
    job_id: str
    status: str


class IndexStatusResponse(BaseModel):
    job_id: str
    status: str
    rows_indexed: int
    total_rows: int | None
    errors: int
    error: str | None = None


class PlotRequest(BaseModel):
    chromadb_collection_name: str
    num_clusters: int = 10
    text_display_fields: list[str] | None = None
    image_field: str | None = None
    reduction_algorithm: Literal["tsne", "umap", "pca"] = "tsne"
    tsne_perplexity: float = 30.0
    tsne_learning_rate: str = "auto"
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"

    @field_validator("reduction_algorithm")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        allowed = {"tsne", "umap", "pca"}
        if v not in allowed:
            msg = (
                f"Invalid reduction algorithm: '{v}'. "
                f"Must be one of: {', '.join(sorted(allowed))}"
            )
            raise ValueError(msg)
        return v


class SuggestClustersRequest(BaseModel):
    collection_name: str
    k_min: int = 2
    k_max: int = 30


class SuggestClustersResponse(BaseModel):
    k_values: list[int]
    inertias: list[float]
    silhouette_scores: list[float]
    suggested_k: int


class SuggestClustersStatusResponse(BaseModel):
    status: str
    ready: bool
    phase: str | None = None
    current_k: int | None = None
    total_k: int | None = None
    result: SuggestClustersResponse | None = None
    error: str | None = None


class PlotPoint(BaseModel):
    x: float
    y: float
    z: float
    cluster: int
    metadata: dict[str, object]
    id: str


class PlotCluster(BaseModel):
    index: int
    name: str
    color: str
    count: int


class PlotResponse(BaseModel):
    points: list[PlotPoint]
    clusters: list[PlotCluster]
    total_points: int


class SearchResult(BaseModel):
    id: str
    distance: float
    metadata: dict[str, object]


class SearchRequest(BaseModel):
    collection_name: str
    query_text: str | None = None
    query_image_url: str | None = None
    n_results: int = 10
    model_type: str = "text"
    image_model_name: str = "openai/clip-vit-base-patch32"
    text_model_name: str = "BAAI/bge-small-en-v1.5"


class SearchResponse(BaseModel):
    results: list[SearchResult]


class ClusterItemResponse(BaseModel):
    id: str
    metadata: dict[str, object]
    distance_to_centroid: float


class ClusterDetailResponse(BaseModel):
    cluster_index: int
    cluster_name: str
    total_items: int
    page: int
    page_size: int
    items: list[ClusterItemResponse]


class SubClusterRequest(BaseModel):
    num_sub_clusters: int = 3
    point_ids: list[str] | None = None

    @field_validator("num_sub_clusters")
    @classmethod
    def validate_num_sub_clusters(cls, v: int) -> int:
        if v < 2:
            msg = "num_sub_clusters must be at least 2"
            raise ValueError(msg)
        return v


class SubClusterPoint(BaseModel):
    id: str
    x: float
    y: float
    z: float
    sub_cluster: int
    metadata: dict[str, object]


class SubClusterInfo(BaseModel):
    index: int
    count: int
    color: str
    name: str | None = None


class SubClusterResponse(BaseModel):
    parent_cluster_index: int
    points: list[SubClusterPoint]
    sub_clusters: list[SubClusterInfo]
    total_points: int


class SuggestKRequest(BaseModel):
    point_ids: list[str] | None = None
    cluster_index: int | None = None
    max_k: int = 10

    @field_validator("max_k")
    @classmethod
    def validate_max_k(cls, v: int) -> int:
        if v < 3:
            msg = "max_k must be at least 3"
            raise ValueError(msg)
        return v


class SuggestKScoreEntry(BaseModel):
    k: int
    score: float


class SuggestKResponse(BaseModel):
    suggested_k: int
    scores: list[SuggestKScoreEntry]


class AnnotationUpdate(BaseModel):
    name: str | None = None
    notes: str | None = None
    tags: list[str] | None = None


class ClusterAnnotation(BaseModel):
    name: str | None = None
    notes: str | None = None
    tags: list[str] | None = None
    updated_at: str | None = None


class AnnotationsResponse(BaseModel):
    job_id: str
    clusters: dict[str, ClusterAnnotation]


class AiNamingRequest(BaseModel):
    job_id: str
    cluster_indices: list[int]
    api_key: str
    model: str
    base_url: str | None = None
    temperature: float = 0.5


class AiNamingResponse(BaseModel):
    names: dict[str, str]


class AiSubClusterNamingRequest(BaseModel):
    job_id: str
    point_ids: list[str]
    sub_cluster_labels: list[int]
    api_key: str
    model: str
    base_url: str | None = None
    temperature: float = 0.5
    parent_cluster_name: str | None = None


class AiTestConnectionRequest(BaseModel):
    api_key: str
    model: str
    base_url: str | None = None


class AiTestConnectionResponse(BaseModel):
    success: bool
    error: str | None = None


class OllamaModelsRequest(BaseModel):
    base_url: str = "http://localhost:11434"


class OllamaModel(BaseModel):
    name: str
    size: int | None = None
    parameter_size: str | None = None
    family: str | None = None


class OllamaModelsResponse(BaseModel):
    models: list[OllamaModel]
