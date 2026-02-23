from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    import pathlib

import numpy as np
import pytest

from embedding_cluster.settings import Settings


class TestGenerateEmbeddingFieldName:
    def test_basic(self) -> None:
        from embedding_cluster.indexer import generate_embedding_field_name

        result = generate_embedding_field_name("embedding_", "image", "url")
        assert result == "embedding_image_url"

    def test_empty_prefix(self) -> None:
        from embedding_cluster.indexer import generate_embedding_field_name

        result = generate_embedding_field_name("", "text", "desc")
        assert result == "text_desc"


class TestEncodeText:
    def test_encode_text_success(self) -> None:
        from embedding_cluster.indexer import encode_text

        mock_transformer = MagicMock()
        mock_transformer.encode.return_value = np.array([1.0, 2.0, 3.0])

        result = encode_text(text_model_transformer=mock_transformer, text="hello")

        assert result is not None
        mock_transformer.encode.assert_called_once_with("hello", show_progress_bar=False)

    def test_encode_text_exception(self) -> None:
        from embedding_cluster.indexer import encode_text

        mock_transformer = MagicMock()
        mock_transformer.encode.side_effect = RuntimeError("boom")

        result = encode_text(text_model_transformer=mock_transformer, text="hello")

        assert result is None


class TestEncodeImage:
    def test_encode_image_with_get_image_features(self) -> None:
        from embedding_cluster.indexer import encode_image

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_image = MagicMock()

        fake_tensor = MagicMock()
        fake_tensor.to.return_value = fake_tensor
        mock_processor.return_value = {"pixel_values": fake_tensor}

        mock_features = MagicMock()
        mock_features.squeeze.return_value = mock_features
        mock_features.cpu.return_value = mock_features
        mock_features.detach.return_value = mock_features
        mock_features.numpy.return_value = np.array([1.0, 2.0])
        mock_model.get_image_features.return_value = mock_features

        result = encode_image(
            image_model=mock_model,
            processor=mock_processor,
            image=mock_image,
            device="cpu",
        )

        assert result is not None

    def test_encode_image_without_get_image_features(self) -> None:
        from embedding_cluster.indexer import encode_image

        mock_model = MagicMock(spec=[])
        mock_processor = MagicMock()
        mock_image = MagicMock()

        fake_tensor = MagicMock()
        fake_tensor.to.return_value = fake_tensor
        mock_processor.return_value = {"pixel_values": fake_tensor}

        hidden = MagicMock()
        hidden.squeeze.return_value = hidden
        hidden.__getitem__ = MagicMock(return_value=hidden)
        hidden.cpu.return_value = hidden
        hidden.detach.return_value = hidden
        hidden.numpy.return_value = np.array([3.0, 4.0])

        mock_output = MagicMock()
        mock_output.hidden_states = [None] * 11 + [hidden]
        mock_model.return_value = mock_output

        result = encode_image(
            image_model=mock_model,
            processor=mock_processor,
            image=mock_image,
            device="cpu",
        )

        assert result is not None

    def test_encode_image_exception(self) -> None:
        from embedding_cluster.indexer import encode_image

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.side_effect = RuntimeError("fail")

        result = encode_image(
            image_model=mock_model,
            processor=mock_processor,
            image=MagicMock(),
            device="cpu",
        )

        assert result is None

    def test_encode_image_no_processor(self) -> None:
        from embedding_cluster.indexer import encode_image

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([5.0, 6.0])

        result = encode_image(
            image_model=mock_model,
            processor=None,  # type: ignore[arg-type]
            image=MagicMock(),
            device="cpu",
        )

        assert result is not None
        mock_model.encode.assert_called_once()


class TestBuildAndEncode:
    @pytest.mark.asyncio
    async def test_build_and_encode_with_id_field(self) -> None:
        from embedding_cluster.indexer import build_and_encode

        mock_image_model = MagicMock()
        mock_processor = MagicMock()
        mock_text_transformer = MagicMock()

        source = {"myid": "123", "name": "test"}

        result = await build_and_encode(
            image_model=mock_image_model,
            image_model_processor=mock_processor,
            image_embedding_fields=None,
            text_model_transformer=mock_text_transformer,
            text_embedding_fields=None,
            embedding_fields_prefix="emb_",
            source=source,
            device="cpu",
            id_field="myid",
        )

        embedding, meta, _id = result
        assert _id == "123"
        assert meta == source
        assert embedding == {}

    @pytest.mark.asyncio
    async def test_build_and_encode_no_id_field(self) -> None:
        from embedding_cluster.indexer import build_and_encode

        result = await build_and_encode(
            image_model=MagicMock(),
            image_model_processor=MagicMock(),
            image_embedding_fields=None,
            text_model_transformer=MagicMock(),
            text_embedding_fields=None,
            embedding_fields_prefix="emb_",
            source={"key": "val"},
            device="cpu",
            id_field=None,
        )

        _, _, _id = result
        assert isinstance(_id, str)
        assert len(_id) == 6

    @pytest.mark.asyncio
    async def test_build_and_encode_text_field(self) -> None:
        from embedding_cluster.indexer import build_and_encode

        mock_text_transformer = MagicMock()
        mock_text_transformer.encode.return_value = np.array([1.0, 2.0])

        result = await build_and_encode(
            image_model=MagicMock(),
            image_model_processor=MagicMock(),
            image_embedding_fields=None,
            text_model_transformer=mock_text_transformer,
            text_embedding_fields=["desc"],
            embedding_fields_prefix="emb_",
            source={"desc": "A product", "myid": "1"},
            device="cpu",
            id_field="myid",
        )

        embedding, _, _ = result
        assert "emb_text_desc" in embedding

    @pytest.mark.asyncio
    async def test_build_and_encode_skips_empty_text(self) -> None:
        from embedding_cluster.indexer import build_and_encode

        mock_text_transformer = MagicMock()

        result = await build_and_encode(
            image_model=MagicMock(),
            image_model_processor=MagicMock(),
            image_embedding_fields=None,
            text_model_transformer=mock_text_transformer,
            text_embedding_fields=["desc"],
            embedding_fields_prefix="emb_",
            source={"desc": "", "myid": "1"},
            device="cpu",
            id_field="myid",
        )

        embedding, _, _ = result
        assert "emb_text_desc" not in embedding
        mock_text_transformer.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_build_and_encode_image_field_missing_url(self) -> None:
        from embedding_cluster.indexer import build_and_encode

        result = await build_and_encode(
            image_model=MagicMock(),
            image_model_processor=MagicMock(),
            image_embedding_fields=["imageUrl"],
            text_model_transformer=MagicMock(),
            text_embedding_fields=None,
            embedding_fields_prefix="emb_",
            source={"myid": "1"},
            device="cpu",
            id_field="myid",
        )

        embedding, _, _ = result
        assert embedding["emb_image_imageUrl"] is None


class TestAsyncWrapperBuildAndEncode:
    @pytest.mark.asyncio
    async def test_wrapper_catches_exception(self) -> None:
        import asyncio

        from embedding_cluster.indexer import (
            async_wrapper_build_and_encode,
        )

        sem = asyncio.Semaphore(1)

        with patch(
            "embedding_cluster.indexer.build_and_encode",
            side_effect=RuntimeError("fail"),
        ):
            result = await async_wrapper_build_and_encode(
                image_model=MagicMock(),
                image_model_processor=MagicMock(),
                image_embedding_fields=None,
                text_model_transformer=MagicMock(),
                text_embedding_fields=None,
                embedding_fields_prefix="emb_",
                source={},
                device="cpu",
                sem=sem,
                id_field=None,
            )

        assert result is None


class TestMainIndexer:
    @pytest.mark.asyncio
    async def test_main_indexer_reads_csv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        from embedding_cluster.indexer import main_indexer

        csv_content = "id,name,imageUrl\n1,test,http://example.com/img.png\n"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        monkeypatch.setenv("RUNNING_MODE", "INDEX")
        monkeypatch.setenv("LOCAL_CSV_FILENAME", str(csv_file))
        monkeypatch.setenv("ID_FIELD", "id")
        monkeypatch.setenv("IMAGE_EMBEDDING_FIELDS", '["imageUrl"]')
        monkeypatch.setenv("CHROMADB_COLLECTION_PREFIX", "test_")
        settings = Settings()

        with (
            patch("embedding_cluster.indexer.chromadb") as mock_chromadb,
            patch("embedding_cluster.indexer.CLIPModel") as mock_clip,
            patch("embedding_cluster.indexer.CLIPProcessor") as mock_proc,
            patch("embedding_cluster.indexer.SentenceTransformer") as mock_st,
            patch("embedding_cluster.indexer.ImageDownloader") as mock_dl_cls,
        ):
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_clip.from_pretrained.return_value = mock_model

            mock_processor = MagicMock()
            mock_proc.from_pretrained.return_value = mock_processor

            mock_text_model = MagicMock()
            mock_text_model.to.return_value = mock_text_model
            mock_st.return_value = mock_text_model

            mock_downloader = MagicMock()
            mock_downloader.download_image_exp_backoff = AsyncMock(return_value=None)
            mock_dl_cls.return_value = mock_downloader

            progress_calls = []

            def on_progress(data: dict[str, float | int | None]) -> None:
                progress_calls.append(data)

            await main_indexer(settings, on_progress=on_progress)

            mock_collection.add.assert_called_once()
            assert progress_calls
            for call in progress_calls:
                assert "elapsed_seconds" in call
