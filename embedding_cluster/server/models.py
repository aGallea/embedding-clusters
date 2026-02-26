from __future__ import annotations

from pydantic import BaseModel


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
    gpt_generate_cluster_name: bool = False
    gpt_default_model: str = "gpt-3.5-turbo"
    gpt_default_temperature: float = 0.51


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
