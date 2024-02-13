import asyncio
import csv
import logging
import time
from typing import Any, Dict, List, Optional

import torch
from chromadb.api import ClientAPI, Collection
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

import chromadb
from embedding_cluster.settings import Settings
from embedding_cluster.utils import (
    ChromaDocsCollection,
    ImageDownloader,
    get_or_create_chromadb_collections,
    id_generator,
    init_chroma_docs_collection,
)

logger = logging.getLogger(__name__)


async def main_indexer():
    settings = Settings()
    chromadb_client: ClientAPI = chromadb.PersistentClient(path="./chromadb")
    chromadb_docs_collections: Dict[
        str, ChromaDocsCollection
    ] = init_chroma_docs_collection(settings)
    chromadb_collections: Dict[str, Collection] = get_or_create_chromadb_collections(
        settings, chromadb_client
    )
    sem: asyncio.Semaphore = asyncio.Semaphore(settings.number_of_async_tasks)
    csv_file = open(settings.local_csv_filename)
    csv_iter = csv.DictReader(csv_file)
    rows_read = 0
    curr_rows = []
    image_model: CLIPModel = CLIPModel.from_pretrained(settings.image_model_name).to(
        settings.process_unit_device
    )
    image_model_processor: CLIPProcessor = CLIPProcessor.from_pretrained(
        settings.image_model_name
    )
    text_model_transformer: SentenceTransformer = SentenceTransformer(
        settings.text_model_name
    ).to(settings.process_unit_device)

    skipped_rows = 0
    if settings.index_start_line is not None:
        skipped_rows = 1
        for row in csv_iter:
            skipped_rows += 1
            if settings.index_start_line == skipped_rows:
                break
            pass

    for row in csv_iter:
        rows_read += 1
        curr_rows.append(row)
        if (
            settings.index_end_line is not None
            and settings.index_end_line == rows_read + skipped_rows
        ):
            break
        if len(curr_rows) == settings.index_bulk_size:
            await handle(
                settings=settings,
                rows=curr_rows,
                sem=sem,
                image_model=image_model,
                image_model_processor=image_model_processor,
                text_model_transformer=text_model_transformer,
                chromadb_docs_collections=chromadb_docs_collections,
                chromadb_collections=chromadb_collections,
            )
            curr_rows = []
            chromadb_docs_collections = init_chroma_docs_collection(settings)
            logger.info(f"Indexed {rows_read} rows. [{skipped_rows + rows_read}]")
    if len(curr_rows) > 0:
        await handle(
            settings=settings,
            rows=curr_rows,
            sem=sem,
            image_model=image_model,
            image_model_processor=image_model_processor,
            text_model_transformer=text_model_transformer,
            chromadb_docs_collections=chromadb_docs_collections,
            chromadb_collections=chromadb_collections,
        )


async def handle(
    settings: Settings,
    rows: List[Any],
    sem: asyncio.Semaphore,
    image_model: CLIPModel,
    image_model_processor: CLIPProcessor,
    text_model_transformer: SentenceTransformer,
    chromadb_docs_collections: Dict[str, ChromaDocsCollection],
    chromadb_collections: Dict[str, Collection],
):
    tasks = [
        asyncio.ensure_future(
            async_wrapper_build_and_encode(
                image_model=image_model,
                image_model_processor=image_model_processor,
                image_embedding_fields=settings.image_embedding_fields,
                text_model_transformer=text_model_transformer,
                text_embedding_fields=settings.text_embedding_fields,
                embedding_fields_prefix=settings.embedding_fields_prefix,
                source=curr_row,
                device=settings.process_unit_device,
                sem=sem,
                id_field=settings.id_field,
            )
        )
        for curr_row in rows
    ]
    docs = await asyncio.gather(*tasks)

    for doc in docs:
        embeddings, meta, ids = doc
        if (
            settings.image_embedding_fields is not None
            and len(settings.image_embedding_fields) > 0
        ):
            model_type = "image"
            for image_embedding_field in settings.image_embedding_fields:
                embedding_field_name = generate_embedding_field_name(
                    settings.embedding_fields_prefix, model_type, image_embedding_field
                )
                curr_embedding = (
                    embeddings.get(embedding_field_name).tolist()
                    if embeddings.get(embedding_field_name) is not None
                    else []
                )
                chromadb_docs_collections[image_embedding_field].embeddings.append(
                    curr_embedding
                )
                chromadb_docs_collections[image_embedding_field].metadatas.append(meta)
                chromadb_docs_collections[image_embedding_field].ids.append(ids)
        if (
            settings.text_embedding_fields is not None
            and len(settings.text_embedding_fields) > 0
        ):
            model_type = "text"
            for text_embedding_field in settings.text_embedding_fields:
                embedding_field_name = generate_embedding_field_name(
                    settings.embedding_fields_prefix, model_type, text_embedding_field
                )
                curr_embedding = embeddings.get(embedding_field_name).tolist()
                chromadb_docs_collections[text_embedding_field].embeddings.append(
                    curr_embedding
                )
                chromadb_docs_collections[text_embedding_field].metadatas.append(meta)
                chromadb_docs_collections[text_embedding_field].ids.append(ids)

    if (
        settings.image_embedding_fields is not None
        and len(settings.image_embedding_fields) > 0
    ):
        model_type = "image"
        for image_embedding_field in settings.image_embedding_fields:
            chromadb_collections[
                f"{settings.chromadb_collection_prefix}{image_embedding_field}"
            ].add(
                embeddings=chromadb_docs_collections[image_embedding_field].embeddings,
                metadatas=chromadb_docs_collections[image_embedding_field].metadatas,
                ids=chromadb_docs_collections[image_embedding_field].ids,
            )
    if (
        settings.text_embedding_fields is not None
        and len(settings.text_embedding_fields) > 0
    ):
        model_type = "text"
        for text_embedding_field in settings.text_embedding_fields:
            chromadb_collections[
                f"{settings.chromadb_collection_prefix}{text_embedding_field}"
            ].add(
                embeddings=chromadb_docs_collections[text_embedding_field].embeddings,
                metadatas=chromadb_docs_collections[text_embedding_field].metadatas,
                ids=chromadb_docs_collections[text_embedding_field].ids,
            )


async def async_wrapper_build_and_encode(
    image_model: CLIPModel,
    image_model_processor: CLIPProcessor,
    image_embedding_fields: list[str],
    text_model_transformer: SentenceTransformer,
    text_embedding_fields: list[str],
    embedding_fields_prefix: str,
    source: Dict[str, Any],
    device: str,
    sem: asyncio.Semaphore,
    id_field: Optional[str],
):
    try:
        async with sem:
            return await build_and_encode(
                image_model,
                image_model_processor,
                image_embedding_fields,
                text_model_transformer,
                text_embedding_fields,
                embedding_fields_prefix,
                source,
                device,
                id_field,
            )
    except Exception:
        logger.error("failed to build and encode doc")


async def build_and_encode(
    image_model: CLIPModel,
    image_model_processor: CLIPProcessor,
    image_embedding_fields: Optional[list[str]],
    text_model_transformer: SentenceTransformer,
    text_embedding_fields: Optional[list[str]],
    embedding_fields_prefix: str,
    source: Dict[str, Any],
    device: str,
    id_field: Optional[str],
):
    if id_field is None:
        _id = id_generator()
    else:
        _id = source.get(id_field)
    embedding: Dict[str, Any] = {}
    if image_embedding_fields is not None and len(image_embedding_fields) > 0:
        model_type = "image"
        for image_embedding_field in image_embedding_fields:
            image_url = source.get(image_embedding_field)
            if image_url is None or image_url == "":
                logger.warn(
                    f"skipping image embedding. image field: {image_embedding_field}, image_url is missing"
                )
                embedding[
                    generate_embedding_field_name(
                        embedding_fields_prefix, model_type, image_embedding_field
                    )
                ] = None
            else:
                image = await ImageDownloader().download_image_exp_backoff(image_url)
                if image is None:
                    logger.warn(
                        f"skipping image embedding. image field: {image_embedding_field}, image url: {image_url}"
                    )
                    embedding[
                        generate_embedding_field_name(
                            embedding_fields_prefix, model_type, image_embedding_field
                        )
                    ] = None
                else:
                    curr_embed_start_time = time.perf_counter()
                    embedding_image = encode_image(
                        image_model=image_model,
                        processor=image_model_processor,
                        image=image,
                        device=device,
                    )
                    took = "{:.3f}".format(time.perf_counter() - curr_embed_start_time)
                    logger.debug(
                        f"image embedding took:{took}s, image field: {image_embedding_field}, image_url:{image_url}"
                    )
                    embedding[
                        generate_embedding_field_name(
                            embedding_fields_prefix, model_type, image_embedding_field
                        )
                    ] = embedding_image
    if text_embedding_fields is not None and len(text_embedding_fields) > 0:
        model_type = "text"
        for text_embedding_field in text_embedding_fields:
            text = source.get(text_embedding_field)
            curr_embed_start_time = time.perf_counter()
            embedding_text = encode_text(
                text_model_transformer=text_model_transformer, text=text
            )
            took = "{:.3f}".format(time.perf_counter() - curr_embed_start_time)
            logger.debug(
                f"text embedding took:{took}s, text field: {text_embedding_field}"
            )
            embedding[
                generate_embedding_field_name(
                    embedding_fields_prefix, model_type, text_embedding_field
                )
            ] = embedding_text
    return embedding, source, _id


def generate_embedding_field_name(
    embedding_fields_prefix: str,
    model_type: str,
    field_name: str,
):
    return f"{embedding_fields_prefix}{model_type}_{field_name}"


def encode_image(
    image_model,
    processor: CLIPProcessor,
    image: Optional[Any] = None,
    device: str = "cpu",
):
    embedding_image = None
    if processor is not None:
        try:
            inputs = processor(text=None, images=image, return_tensors="pt", padding=True)
            if hasattr(image_model, "get_image_features"):
                img_emb = image_model.get_image_features(
                    inputs["pixel_values"].to(device)
                )
                embedding_image = img_emb.squeeze(0).cpu().detach().numpy()
            else:
                with torch.no_grad():
                    output = image_model(
                        inputs["pixel_values"].to(device), output_hidden_states=True
                    )
                embedding_image = (
                    output.hidden_states[11].squeeze()[0,].cpu().detach().numpy()
                )
        except Exception as e:
            logger.exception(f"failed to encode image")
            logger.error(e)
    else:
        embedding_image = image_model.encode(image, show_progress_bar=False)
    return embedding_image


def encode_text(
    text_model_transformer: SentenceTransformer,
    text: str,
):
    try:
        return text_model_transformer.encode(text, show_progress_bar=False)
    except Exception as e:
        logger.exception(f"failed to encode text")
        logger.error(e)
