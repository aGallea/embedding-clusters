from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CollectionInfo(BaseModel):
    name: str
    count: int


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


class PlotPoint(BaseModel):
    x: float
    y: float
    z: float
    cluster: int
    metadata: dict[str, Any]
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
