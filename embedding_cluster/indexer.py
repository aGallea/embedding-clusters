from __future__ import annotations

import asyncio
import csv
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection

    from embedding_cluster.settings import Settings

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from embedding_cluster.utils import (
    ChromaDocsCollection,
    ImageDownloader,
    get_or_create_chromadb_collections,
    id_generator,
    init_chroma_docs_collection,
)

logger = logging.getLogger(__name__)

PROGRESS_UPDATE_INTERVAL = 10


async def main_indexer(
    settings: Settings,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    on_log: Callable[[str, str, str], Awaitable[None]] | None = None,
    cancel_event: asyncio.Event | None = None,
) -> None:
    async def _emit_log(
        message: str,
        level: str = "info",
        verbosity: str = "low",
    ) -> None:
        if on_log is not None:
            await on_log(message, level, verbosity)

    chromadb_client: ClientAPI = chromadb.PersistentClient(path="./chromadb")
    chromadb_docs_collections: dict[str, ChromaDocsCollection] = (
        init_chroma_docs_collection(settings)
    )
    chromadb_collections: dict[str, Collection] = get_or_create_chromadb_collections(
        settings, chromadb_client
    )
    sem: asyncio.Semaphore = asyncio.Semaphore(settings.number_of_async_tasks)

    image_model: CLIPModel | None = None
    image_model_processor: CLIPProcessor | None = None
    text_model_transformer: SentenceTransformer | None = None

    if (
        settings.image_embedding_fields is not None
        and len(settings.image_embedding_fields) > 0
    ):
        logger.info("Loading image model: %s", settings.image_model_name)
        await _emit_log(f"Loading image model: {settings.image_model_name}...")
        try:
            image_model = await asyncio.to_thread(
                lambda: CLIPModel.from_pretrained(settings.image_model_name).to(
                    settings.process_unit_device
                )
            )
            image_model_processor = await asyncio.to_thread(
                CLIPProcessor.from_pretrained, settings.image_model_name
            )
            await _emit_log("Image model loaded successfully")
        except Exception as exc:
            await _emit_log(
                f"Failed to load image model: {exc}",
                level="error",
            )
            raise

    if (
        settings.text_embedding_fields is not None
        and len(settings.text_embedding_fields) > 0
    ):
        logger.info("Loading text model: %s", settings.text_model_name)
        await _emit_log(f"Loading text model: {settings.text_model_name}...")
        try:
            text_model_transformer = await asyncio.to_thread(
                lambda: SentenceTransformer(settings.text_model_name).to(
                    settings.process_unit_device
                )
            )
            await _emit_log("Text model loaded successfully")
        except Exception as exc:
            await _emit_log(
                f"Failed to load text model: {exc}",
                level="error",
            )
            raise

    start_time = time.perf_counter()

    await _emit_log("Loading CSV file...")
    with open(settings.local_csv_filename) as csv_file:
        csv_iter = csv.DictReader(csv_file)
        await _emit_log("CSV file opened, reading rows...")
        rows_read = 0
        curr_rows: list[dict[str, Any]] = []
        batch_num = 0
        skipped_rows = 0
        if settings.index_start_line is not None:
            skipped_rows = 1
            for _row in csv_iter:
                skipped_rows += 1
                if settings.index_start_line == skipped_rows:
                    break

        for row in csv_iter:
            if cancel_event is not None and cancel_event.is_set():
                logger.info(
                    "Indexing cancelled at row %d",
                    rows_read + skipped_rows,
                )
                await _emit_log(
                    f"Indexing cancelled at row {rows_read + skipped_rows}",
                    level="warning",
                )
                break
            rows_read += 1
            curr_rows.append(row)
            if on_progress is not None and rows_read % PROGRESS_UPDATE_INTERVAL == 0:
                on_progress(
                    {
                        "rows_indexed": rows_read,
                        "total_rows": None,
                        "errors": 0,
                        "elapsed_seconds": (time.perf_counter() - start_time),
                    }
                )
                await _emit_log(
                    f"Processing row {rows_read}...",
                    verbosity="high",
                )
            if (
                settings.index_end_line is not None
                and settings.index_end_line == rows_read + skipped_rows
            ):
                break
            if len(curr_rows) == settings.index_bulk_size:
                batch_num += 1
                batch_start = rows_read - len(curr_rows) + 1
                await _emit_log(
                    f"Processing batch {batch_num} ({batch_start}-{rows_read})...",
                    verbosity="medium",
                )
                await _handle_batch(
                    settings=settings,
                    rows=curr_rows,
                    sem=sem,
                    image_model=image_model,
                    image_model_processor=image_model_processor,
                    text_model_transformer=text_model_transformer,
                    chromadb_docs_collections=chromadb_docs_collections,
                    chromadb_collections=chromadb_collections,
                )
                await _emit_log(
                    f"Batch {batch_num} complete, writing to ChromaDB...",
                    verbosity="medium",
                )
                curr_rows = []
                chromadb_docs_collections = init_chroma_docs_collection(settings)
                if on_progress is not None:
                    on_progress(
                        {
                            "rows_indexed": rows_read,
                            "total_rows": None,
                            "errors": 0,
                            "elapsed_seconds": (time.perf_counter() - start_time),
                        }
                    )
                await _emit_log(
                    f"Indexed {rows_read} rows so far",
                    verbosity="medium",
                )
                logger.info(
                    "Indexed %d rows. [%d]",
                    rows_read,
                    skipped_rows + rows_read,
                )
        if len(curr_rows) > 0:
            batch_num += 1
            batch_start = rows_read - len(curr_rows) + 1
            await _emit_log(
                f"Processing batch {batch_num} ({batch_start}-{rows_read})...",
                verbosity="medium",
            )
            await _handle_batch(
                settings=settings,
                rows=curr_rows,
                sem=sem,
                image_model=image_model,
                image_model_processor=image_model_processor,
                text_model_transformer=text_model_transformer,
                chromadb_docs_collections=chromadb_docs_collections,
                chromadb_collections=chromadb_collections,
            )
            await _emit_log(
                f"Batch {batch_num} complete, writing to ChromaDB...",
                verbosity="medium",
            )
            if on_progress is not None:
                on_progress(
                    {
                        "rows_indexed": rows_read,
                        "total_rows": None,
                        "errors": 0,
                        "elapsed_seconds": (time.perf_counter() - start_time),
                    }
                )

        elapsed = time.perf_counter() - start_time
        await _emit_log(
            f"Indexing complete: {rows_read} rows in {elapsed:.1f}s",
            level="success",
        )


async def _handle_batch(
    settings: Settings,
    rows: list[Any],
    sem: asyncio.Semaphore,
    image_model: CLIPModel | None,
    image_model_processor: CLIPProcessor | None,
    text_model_transformer: SentenceTransformer | None,
    chromadb_docs_collections: dict[str, ChromaDocsCollection],
    chromadb_collections: dict[str, Collection],
) -> None:
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

    # Filter out None results from failed build_and_encode calls
    valid_docs = [doc for doc in docs if doc is not None]

    for doc in valid_docs:
        embeddings, meta, ids = doc
        if (
            settings.image_embedding_fields is not None
            and len(settings.image_embedding_fields) > 0
        ):
            model_type = "image"
            for image_embedding_field in settings.image_embedding_fields:
                embedding_field_name = generate_embedding_field_name(
                    settings.embedding_fields_prefix,
                    model_type,
                    image_embedding_field,
                )
                curr_embedding_val = embeddings.get(embedding_field_name)
                if curr_embedding_val is None:
                    logger.warning(
                        "Skipping doc %s: missing %s embedding",
                        ids,
                        embedding_field_name,
                    )
                    continue
                chromadb_docs_collections[image_embedding_field].embeddings.append(
                    curr_embedding_val.tolist()
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
                    settings.embedding_fields_prefix,
                    model_type,
                    text_embedding_field,
                )
                curr_embedding_val = embeddings.get(embedding_field_name)
                if curr_embedding_val is None:
                    logger.warning(
                        "Skipping doc %s: missing %s embedding",
                        ids,
                        embedding_field_name,
                    )
                    continue
                chromadb_docs_collections[text_embedding_field].embeddings.append(
                    curr_embedding_val.tolist()
                )
                chromadb_docs_collections[text_embedding_field].metadatas.append(meta)
                chromadb_docs_collections[text_embedding_field].ids.append(ids)

    if (
        settings.image_embedding_fields is not None
        and len(settings.image_embedding_fields) > 0
    ):
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
        for text_embedding_field in settings.text_embedding_fields:
            chromadb_collections[
                f"{settings.chromadb_collection_prefix}{text_embedding_field}"
            ].add(
                embeddings=chromadb_docs_collections[text_embedding_field].embeddings,
                metadatas=chromadb_docs_collections[text_embedding_field].metadatas,
                ids=chromadb_docs_collections[text_embedding_field].ids,
            )


async def async_wrapper_build_and_encode(
    image_model: CLIPModel | None,
    image_model_processor: CLIPProcessor | None,
    image_embedding_fields: list[str] | None,
    text_model_transformer: SentenceTransformer | None,
    text_embedding_fields: list[str] | None,
    embedding_fields_prefix: str,
    source: dict[str, Any],
    device: str,
    sem: asyncio.Semaphore,
    id_field: str | None,
) -> tuple[dict[str, Any], dict[str, Any], str] | None:
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
        return None


async def build_and_encode(
    image_model: CLIPModel | None,
    image_model_processor: CLIPProcessor | None,
    image_embedding_fields: list[str] | None,
    text_model_transformer: SentenceTransformer | None,
    text_embedding_fields: list[str] | None,
    embedding_fields_prefix: str,
    source: dict[str, Any],
    device: str,
    id_field: str | None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    _id = id_generator() if id_field is None else source.get(id_field, id_generator())
    embedding: dict[str, Any] = {}
    if image_embedding_fields is not None and len(image_embedding_fields) > 0:
        if image_model is None or image_model_processor is None:
            msg = "Image model not loaded but image_embedding_fields specified"
            raise RuntimeError(msg)
        model_type = "image"
        for image_embedding_field in image_embedding_fields:
            image_url = source.get(image_embedding_field)
            if image_url is None or image_url == "":
                logger.warning(
                    "skipping image embedding. image field: %s, image_url is missing",
                    image_embedding_field,
                )
                embedding[
                    generate_embedding_field_name(
                        embedding_fields_prefix,
                        model_type,
                        image_embedding_field,
                    )
                ] = None
            else:
                image = await ImageDownloader().download_image_exp_backoff(image_url)
                if image is None:
                    logger.warning(
                        "skipping image embedding. image field: %s, image url: %s",
                        image_embedding_field,
                        image_url,
                    )
                    embedding[
                        generate_embedding_field_name(
                            embedding_fields_prefix,
                            model_type,
                            image_embedding_field,
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
                    took = f"{time.perf_counter() - curr_embed_start_time:.3f}"
                    logger.debug(
                        "image embedding took:%ss, image field: %s, image_url:%s",
                        took,
                        image_embedding_field,
                        image_url,
                    )
                    embedding[
                        generate_embedding_field_name(
                            embedding_fields_prefix,
                            model_type,
                            image_embedding_field,
                        )
                    ] = embedding_image
    if text_embedding_fields is not None and len(text_embedding_fields) > 0:
        if text_model_transformer is None:
            msg = "Text model not loaded but text_embedding_fields specified"
            raise RuntimeError(msg)
        model_type = "text"
        for text_embedding_field in text_embedding_fields:
            text = source.get(text_embedding_field)
            if text is None or text == "":
                logger.warning(
                    "skipping text embedding. text field: %s, text value is missing",
                    text_embedding_field,
                )
                continue
            curr_embed_start_time = time.perf_counter()
            embedding_text = encode_text(
                text_model_transformer=text_model_transformer, text=text
            )
            took = f"{time.perf_counter() - curr_embed_start_time:.3f}"
            logger.debug(
                "text embedding took:%ss, text field: %s",
                took,
                text_embedding_field,
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
) -> str:
    return f"{embedding_fields_prefix}{model_type}_{field_name}"


def encode_image(
    image_model: Any,
    processor: CLIPProcessor,
    image: Any = None,
    device: str = "cpu",
) -> Any:
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
                        inputs["pixel_values"].to(device),
                        output_hidden_states=True,
                    )
                embedding_image = (
                    output.hidden_states[11].squeeze()[0,].cpu().detach().numpy()
                )
        except Exception:
            logger.exception("failed to encode image")
    else:
        embedding_image = image_model.encode(image, show_progress_bar=False)
    return embedding_image


def encode_text(
    text_model_transformer: SentenceTransformer,
    text: str,
) -> Any:
    try:
        return text_model_transformer.encode(text, show_progress_bar=False)
    except Exception:
        logger.exception("failed to encode text")
        return None
