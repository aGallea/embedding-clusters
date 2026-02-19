from __future__ import annotations

import asyncio
import io
import logging
import random
import string
import sys
import time
from collections.abc import Mapping, Sequence  # noqa: TC003
from typing import TYPE_CHECKING, Any, ClassVar

import aiohttp
from PIL import Image
from pydantic import BaseModel

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection

    from embedding_cluster.settings import Settings

logger = logging.getLogger(__name__)


def init_logger() -> None:
    log_handler = logging.StreamHandler(stream=sys.stdout)
    log_handler.setFormatter(
        Formatter(
            fmt="%(asctime)-15s %(levelname)-18.18s %(message)s [%(filename)s:%(lineno)d]"
        )
    )
    logging.root.addHandler(log_handler)
    logging.root.setLevel(logging.INFO)


class Formatter(logging.Formatter):
    @classmethod
    def _get_level_color(cls, levelno: int) -> str:
        default = "\033[0m"
        return {
            logging.DEBUG: "\033[0;96m",
            logging.INFO: "\033[0;92m",
            logging.WARNING: "\033[0;33m",
            logging.ERROR: "\033[0;31m",
        }.get(levelno, default)

    def format(self, record: logging.LogRecord) -> str:
        record.levelname = (
            f"{self._get_level_color(record.levelno)}{record.levelname}\033[0m"
        )
        return super().format(record)


class ChromaDocsCollection(BaseModel):
    ids: list[str]
    embeddings: list[Sequence[float] | Sequence[int]]
    metadatas: list[Mapping[str, str | int | float | bool]]


class Singleton(type):
    """Metaclass that ensures only one instance per class (thread-unsafe)."""

    _instances: ClassVar[dict[Any, Any]] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class ImageDownloader(metaclass=Singleton):
    def __init__(self) -> None:
        # Lazy session — created on first download request
        self.session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> None:
        """Create session lazily to avoid creating it at import time."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=100))

    async def close_session(self) -> None:
        if self.session:
            await self.session.close()
            self.session = None

    async def recreate_session(self) -> None:
        logger.info("recreating session")
        await self.close_session()
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=100))

    async def download_image_exp_backoff(
        self, image_url: str, retries: int = 6
    ) -> Image.Image | None:
        start_time = time.perf_counter()
        delay = 1
        await self._ensure_session()
        assert self.session is not None  # ensured by _ensure_session
        while retries > 0:
            try:
                if image_url is None:
                    retries = 0
                    raise Exception("image_url is None")
                if not self.session or self.session.closed:
                    await self.recreate_session()
                async with self.session.get(image_url) as resp:
                    if resp.status >= 400:
                        raise ValueError(resp.status, resp.reason)
                    image_raw = await resp.read()
                    if retries < 6:
                        logger.info(
                            "image download success after %d retries, image_url:%s",
                            6 - retries,
                            image_url,
                        )
                    image = Image.open(io.BytesIO(image_raw))
                    took = f"{time.perf_counter() - start_time:.3f}"
                    logger.debug("image get took:%ss, image_url:%s", took, image_url)
                    return image
            except (TimeoutError, aiohttp.ClientResponseError, ValueError) as e:
                # Initialize defaults before the if-chain so they're always set
                status = 500
                reason = "Unknown error"
                if isinstance(e, ValueError):
                    status = 400
                    reason = "Error"
                if isinstance(e, asyncio.TimeoutError):
                    status = 408
                    reason = "Timeout"
                if isinstance(e, aiohttp.ClientResponseError):
                    status = e.status
                    reason = str(e)

                log = f"[{retries}] failed to download image: {image_url} Error: {reason}"
                if status in (429, 403, 408):
                    pass
                elif 400 <= status < 600:
                    retries = 0

                delay *= 2
                retries -= 1
                if retries > 0:
                    log += f", Retrying in {delay} seconds..."
                    await asyncio.sleep(delay)
                logger.warning(log)
            except Exception as e:
                logger.warning(
                    "Failed to download image: %s error: %s",
                    image_url,
                    str(e),
                )
                retries = 0

        took = f"{time.perf_counter() - start_time:.3f}"
        logger.error("Failed to download image: %s took:%ss", image_url, took)
        return None


def get_or_create_chromadb_collections(
    settings: Settings, chromadb_client: ClientAPI
) -> dict[str, Collection]:
    chromadb_collections: dict[str, Collection] = {}
    if settings.image_embedding_fields is not None:
        for image_embedding_field in settings.image_embedding_fields:
            collection_name = (
                f"{settings.chromadb_collection_prefix}{image_embedding_field}"
            )
            chromadb_collections[collection_name] = (
                chromadb_client.get_or_create_collection(collection_name)
            )
    if settings.text_embedding_fields is not None:
        for text_embedding_field in settings.text_embedding_fields:
            collection_name = (
                f"{settings.chromadb_collection_prefix}{text_embedding_field}"
            )
            chromadb_collections[collection_name] = (
                chromadb_client.get_or_create_collection(collection_name)
            )
    return chromadb_collections


def init_chroma_docs_collection(
    settings: Settings,
) -> dict[str, ChromaDocsCollection]:
    chroma_docs: dict[str, ChromaDocsCollection] = {}
    if settings.image_embedding_fields is not None:
        for image_embedding_field in settings.image_embedding_fields:
            chroma_docs[image_embedding_field] = ChromaDocsCollection(
                embeddings=[], metadatas=[], ids=[]
            )
    if settings.text_embedding_fields is not None:
        for text_embedding_field in settings.text_embedding_fields:
            chroma_docs[text_embedding_field] = ChromaDocsCollection(
                embeddings=[], metadatas=[], ids=[]
            )
    return chroma_docs


def id_generator(
    size: int = 6, chars: str = string.ascii_uppercase + string.digits
) -> str:
    return "".join(random.choice(chars) for _ in range(size))
