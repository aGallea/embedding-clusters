from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
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
    chromadb_collection_prefix: str = Field(
        default="", description="Prefix for the created chromadb collection name"
    )

    image_model_name: str = Field(
        default="openai/clip-vit-base-patch32",
        description="Image model to use for embedding images",
    )
    image_embedding_fields: Optional[list[str]] = Field(
        default=None, description="Names of the image fields to embed"
    )
    text_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Text model to use for embedding text fields",
    )
    text_embedding_fields: Optional[list[str]] = Field(
        default=None, description="Names of the text fields to embed"
    )
    embedding_fields_prefix: str = Field(
        default="embedding_", description="Prefix for the new created embedding fields"
    )
    id_field: Optional[str] = Field(
        default=None,
        description="field name for the doc id, random id will be generated if not provided",
    )

    num_clusters: int = Field(default=10, description="Number of plot clusters")
    chromadb_collection_name: str = Field(
        default="", description="chromadb collection name to use for data source"
    )
    text_display_fields: Optional[list[str]] = Field(
        default=None, description="field names for the name to present on plot"
    )
    image_field: Optional[str] = Field(
        default=None, description="field name for the image to present on plot"
    )

    gpt_generate_cluster_name: bool = Field(
        default=False, description="Generate cluster names using GPT"
    )
    # gpt-4-1106-preview / gpt-3.5-turbo
    gpt_default_model: str = Field(default="gpt-3.5-turbo", description="GPT model name")
    gpt_default_temperature: float = Field(
        default=0.51, description="GPT model temperature"
    )
