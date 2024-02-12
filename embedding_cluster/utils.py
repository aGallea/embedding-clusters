import asyncio
import io
import logging
import random
import string
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence, Union

import aiohttp
from chromadb.api import ClientAPI, Collection
from PIL import Image
from pydantic import BaseModel

from embedding_cluster.settings import Settings

logger = logging.getLogger(__name__)


def init_logger():
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
    def _get_level_color(cls, levelno):
        default = "\033[0m"
        return {
            logging.DEBUG: "\033[0;96m",
            logging.INFO: "\033[0;92m",
            logging.WARNING: "\033[0;33m",
            logging.WARN: "\033[0;33m",
            logging.ERROR: "\033[0;31m",
        }.get(levelno, default)

    def format(self, record):
        record.levelname = (
            f"{self._get_level_color(record.levelno)}{record.levelname}\033[0m"
        )
        return super().format(record)


class ChromaDocsCollection(BaseModel):
    ids: List[str]
    embeddings: List[Union[Sequence[float], Sequence[int]]]
    metadatas: List[Mapping[str, Union[str, int, float, bool]]]


class Singleton(type):
    _instances: Dict[Any, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ImageDownloader(metaclass=Singleton):
    def __init__(self) -> None:
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=100))

    async def close_session(self) -> None:
        if self.session:
            await self.session.close()
            self.session = None

    async def recreate_session(self) -> None:
        logger.info("recreating session")
        await self.close_session()
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=100))

    async def download_image_exp_backoff(self, image_url: str, retries=6):
        start_time = time.perf_counter()
        delay = 1
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
                            f"image download success after {6 - retries} retries, image_url:{image_url}"
                        )
                    image = Image.open(io.BytesIO(image_raw))
                    took = "{:.3f}".format(time.perf_counter() - start_time)
                    logger.debug(f"image get took:{took}s, image_url:{image_url}")
                    return image
            except (asyncio.TimeoutError, aiohttp.ClientResponseError, ValueError) as e:
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
                if status == 429 or status == 403 or status == 408:
                    pass
                elif 400 <= status < 600:
                    retries = 0

                delay *= 2
                retries -= 1
                if retries > 0:
                    log += f", Retrying in {delay} seconds..."
                    await asyncio.sleep(delay)
                logger.warn(log)
            except Exception as e:
                logger.warning(f"Failed to download image: {image_url} error: {str(e)}")
                retries = 0

        took = "{:.3f}".format(time.perf_counter() - start_time)
        logger.error(f"Failed to download image: {image_url} took:{took}s")


def get_or_create_chromadb_collections(
    settings: Settings, chromadb_client: ClientAPI
) -> Dict[str, Collection]:
    chromadb_collections = {}
    if settings.image_embedding_fields is not None:
        for image_embedding_field in settings.image_embedding_fields:
            collection_name = (
                f"{settings.chromadb_collection_prefix}{image_embedding_field}"
            )
            chromadb_collections[
                collection_name
            ] = chromadb_client.get_or_create_collection(collection_name)
    if settings.text_embedding_fields is not None:
        for text_embedding_field in settings.text_embedding_fields:
            collection_name = (
                f"{settings.chromadb_collection_prefix}{text_embedding_field}"
            )
            chromadb_collections[
                collection_name
            ] = chromadb_client.get_or_create_collection(collection_name)
    return chromadb_collections


def init_chroma_docs_collection(settings: Settings) -> Dict[str, ChromaDocsCollection]:
    chroma_docs: Dict[str, ChromaDocsCollection] = {}
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


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))
