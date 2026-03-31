from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    running_mode: str = Field(default="PLOT", description="PLOT/INDEX")
    process_unit_device: str = Field(default="cpu", description="cpu/mps/cuda")
    local_csv_filename: str = Field(
        default="csv/marvel_heroes.csv", description="CSV file path"
    )
    number_of_async_tasks: int = Field(
        default=1, description="Number of async tasks - for parallalism"
    )
    index_bulk_size: int = Field(
        default=100, description="Bulk size when indexing embeddings"
    )
    index_start_line: int | None = Field(
        default=None, description="First line number to index"
    )
    index_end_line: int | None = Field(
        default=None, description="Last line number to index"
    )
    chromadb_collection_prefix: str = Field(
        default="", description="Prefix for the created chromadb collection name"
    )

    image_model_name: str = Field(
        default="openai/clip-vit-base-patch32",
        description="Image model to use for embedding images",
    )
    image_embedding_fields: list[str] | None = Field(
        default=None, description="Names of the image fields to embed"
    )
    text_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Text model to use for embedding text fields",
    )
    text_embedding_fields: list[str] | None = Field(
        default=None, description="Names of the text fields to embed"
    )
    embedding_fields_prefix: str = Field(
        default="embedding_",
        description="Prefix for the new created embedding fields",
    )
    id_field: str | None = Field(
        default=None,
        description="field name for the doc id, random id if not provided",
    )

    num_clusters: int = Field(default=10, description="Number of plot clusters")
    chromadb_collection_name: str = Field(
        default="", description="chromadb collection name to use for data source"
    )
    text_display_fields: list[str] | None = Field(
        default=None, description="field names for the name to present on plot"
    )
    image_field: str | None = Field(
        default=None, description="field name for the image to present on plot"
    )

    reduction_algorithm: str = Field(
        default="tsne",
        description="Dimensionality reduction algorithm: tsne, umap, or pca",
    )
    tsne_perplexity: float = Field(default=30.0, description="t-SNE perplexity parameter")
    tsne_learning_rate: str = Field(default="auto", description="t-SNE learning rate")
    umap_n_neighbors: int = Field(default=15, description="UMAP number of neighbors")
    umap_min_dist: float = Field(default=0.1, description="UMAP minimum distance")
    umap_metric: str = Field(default="cosine", description="UMAP distance metric")
