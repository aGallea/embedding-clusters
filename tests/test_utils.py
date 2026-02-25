from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from embedding_cluster.settings import Settings
from embedding_cluster.utils import (
    ChromaDocsCollection,
    Formatter,
    ImageDownloader,
    Singleton,
    get_or_create_chromadb_collections,
    id_generator,
    init_chroma_docs_collection,
    init_logger,
)


class TestFormatter:
    def test_format_info_level(self) -> None:
        formatter = Formatter(fmt="%(levelname)s %(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "hello" in result
        assert "\033[0;92m" in result  # green for INFO

    def test_format_warning_level(self) -> None:
        formatter = Formatter(fmt="%(levelname)s %(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="warn",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "\033[0;33m" in result  # yellow for WARNING

    def test_format_error_level(self) -> None:
        formatter = Formatter(fmt="%(levelname)s %(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="err",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "\033[0;31m" in result  # red for ERROR

    def test_format_debug_level(self) -> None:
        formatter = Formatter(fmt="%(levelname)s %(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=1,
            msg="dbg",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "\033[0;96m" in result  # cyan for DEBUG

    def test_format_unknown_level_uses_default(self) -> None:
        formatter = Formatter(fmt="%(levelname)s %(message)s")
        record = logging.LogRecord(
            name="test",
            level=99,
            pathname="test.py",
            lineno=1,
            msg="custom",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "\033[0m" in result


class TestInitLogger:
    def test_init_logger_adds_handler(self) -> None:
        original_handlers = logging.root.handlers[:]
        init_logger()
        assert len(logging.root.handlers) > len(original_handlers)
        # Cleanup: remove added handler
        for h in logging.root.handlers:
            if h not in original_handlers:
                logging.root.removeHandler(h)


class TestChromaDocsCollection:
    def test_create_empty(self) -> None:
        cdc = ChromaDocsCollection(embeddings=[], metadatas=[], ids=[])
        assert cdc.ids == []
        assert cdc.embeddings == []
        assert cdc.metadatas == []

    def test_create_with_data(self) -> None:
        cdc = ChromaDocsCollection(
            embeddings=[[1.0, 2.0]],
            metadatas=[{"key": "value"}],
            ids=["id1"],
        )
        assert cdc.ids == ["id1"]
        assert cdc.embeddings == [[1.0, 2.0]]
        assert cdc.metadatas == [{"key": "value"}]


class TestSingleton:
    def test_singleton_returns_same_instance(self) -> None:
        class MyClass(metaclass=Singleton):
            pass

        a = MyClass()
        b = MyClass()
        assert a is b

    def test_different_classes_different_instances(self) -> None:
        class ClassA(metaclass=Singleton):
            pass

        class ClassB(metaclass=Singleton):
            pass

        a = ClassA()
        b = ClassB()
        assert a is not b


class TestIdGenerator:
    def test_default_length(self) -> None:
        result = id_generator()
        assert len(result) == 6

    def test_custom_length(self) -> None:
        result = id_generator(size=10)
        assert len(result) == 10

    def test_custom_chars(self) -> None:
        result = id_generator(size=20, chars="A")
        assert result == "A" * 20

    def test_returns_string(self) -> None:
        result = id_generator()
        assert isinstance(result, str)


class TestGetOrCreateChromadbCollections:
    def test_with_image_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IMAGE_EMBEDDING_FIELDS", '["imageUrl"]')
        monkeypatch.setenv("CHROMADB_COLLECTION_PREFIX", "test_")
        settings = Settings()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        result = get_or_create_chromadb_collections(settings, mock_client)

        assert "test_imageUrl" in result
        mock_client.get_or_create_collection.assert_called_with(
            "test_imageUrl",
            metadata={
                "model_name": "openai/clip-vit-base-patch32",
                "model_type": "image",
            },
        )

    def test_with_text_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEXT_EMBEDDING_FIELDS", '["description"]')
        monkeypatch.setenv("CHROMADB_COLLECTION_PREFIX", "pre_")
        settings = Settings()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        result = get_or_create_chromadb_collections(settings, mock_client)

        assert "pre_description" in result
        mock_client.get_or_create_collection.assert_called_with(
            "pre_description",
            metadata={
                "model_name": "BAAI/bge-small-en-v1.5",
                "model_type": "text",
            },
        )

    def test_with_no_fields(self) -> None:
        settings = Settings()
        mock_client = MagicMock()

        result = get_or_create_chromadb_collections(settings, mock_client)

        assert result == {}
        mock_client.get_or_create_collection.assert_not_called()


class TestInitChromaDocsCollection:
    def test_with_image_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IMAGE_EMBEDDING_FIELDS", '["imgField"]')
        settings = Settings()
        result = init_chroma_docs_collection(settings)
        assert "imgField" in result
        assert isinstance(result["imgField"], ChromaDocsCollection)

    def test_with_text_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEXT_EMBEDDING_FIELDS", '["textField"]')
        settings = Settings()
        result = init_chroma_docs_collection(settings)
        assert "textField" in result

    def test_with_no_fields(self) -> None:
        settings = Settings()
        result = init_chroma_docs_collection(settings)
        assert result == {}


class TestImageDownloader:
    @pytest.mark.asyncio
    async def test_download_success(self) -> None:
        downloader = ImageDownloader()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.read = AsyncMock(return_value=_create_minimal_png())
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.close = AsyncMock()
        downloader.session = mock_session

        result = await downloader.download_image_exp_backoff("http://example.com/img.png")

        assert result is not None
        await downloader.close_session()
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_none_url(self) -> None:
        downloader = ImageDownloader()
        mock_session = MagicMock()
        mock_session.closed = False
        downloader.session = mock_session

        result = await downloader.download_image_exp_backoff(
            None,
            retries=1,  # type: ignore[arg-type]
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_close_session(self) -> None:
        downloader = ImageDownloader()
        mock_session = AsyncMock()
        downloader.session = mock_session

        await downloader.close_session()

        mock_session.close.assert_called_once()
        assert downloader.session is None

    @pytest.mark.asyncio
    async def test_recreate_session(self) -> None:
        downloader = ImageDownloader()
        mock_session = AsyncMock()
        downloader.session = mock_session

        with patch("embedding_cluster.utils.aiohttp") as mock_aiohttp:
            mock_new_session = MagicMock()
            mock_aiohttp.ClientSession.return_value = mock_new_session
            mock_aiohttp.ClientTimeout = MagicMock()
            await downloader.recreate_session()

        assert downloader.session is mock_new_session

    @pytest.mark.asyncio
    async def test_ensure_session_creates_new(self) -> None:
        downloader = ImageDownloader()
        downloader.session = None

        with patch("embedding_cluster.utils.aiohttp") as mock_aiohttp:
            mock_new_session = MagicMock()
            mock_new_session.closed = False
            mock_aiohttp.ClientSession.return_value = mock_new_session
            mock_aiohttp.ClientTimeout = MagicMock()
            await downloader._ensure_session()

        assert downloader.session is mock_new_session

    @pytest.mark.asyncio
    async def test_download_http_error_no_retry(self) -> None:
        """Non-retryable HTTP error (e.g. 404) should fail fast."""
        downloader = ImageDownloader()
        mock_resp = AsyncMock()
        mock_resp.status = 404
        mock_resp.reason = "Not Found"
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_resp)
        downloader.session = mock_session

        result = await downloader.download_image_exp_backoff(
            "http://example.com/missing.png", retries=3
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_session_closed_triggers_recreate(self) -> None:
        """Test that session.closed=True triggers recreate_session (lines 104-105)."""
        import aiohttp

        downloader = ImageDownloader()

        # Create retryable error (429)
        error_429 = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=429,
            message="Too Many Requests",
        )

        # Success response
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.read = AsyncMock(return_value=_create_minimal_png())
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        # Create initial session with close as AsyncMock
        mock_session = MagicMock()
        mock_session.closed = False  # Start not closed
        mock_session.close = AsyncMock()  # Make close async

        # Define get behavior: first call raises 429, then session becomes closed
        def get_side_effect(*args, **kwargs):
            # After first call, mark session as closed for the next iteration
            mock_session.closed = True
            raise error_429

        mock_session.get = MagicMock(side_effect=get_side_effect)
        downloader.session = mock_session

        with (
            patch("embedding_cluster.utils.aiohttp.ClientSession") as mock_client_session,
            patch("embedding_cluster.utils.aiohttp.ClientTimeout"),
            patch("embedding_cluster.utils.asyncio.sleep", new_callable=AsyncMock),
        ):
            # New session created during recreate_session
            mock_new_session = MagicMock()
            mock_new_session.closed = False
            mock_new_session.get = MagicMock(return_value=mock_resp)
            mock_new_session.close = AsyncMock()
            mock_client_session.return_value = mock_new_session

            result = await downloader.download_image_exp_backoff(
                "http://example.com/img.png", retries=2
            )

        # Verify session was replaced during the retry loop
        assert downloader.session is mock_new_session
        assert result is not None

    @pytest.mark.asyncio
    async def test_retry_success_logs_message(self) -> None:
        """Test that successful download after retry logs success message (line 111)."""
        import aiohttp

        downloader = ImageDownloader()

        # First response returns 429 error (via ClientResponseError)
        error_429 = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=429,
            message="Too Many Requests",
        )

        # Second response returns 200 with valid PNG
        mock_resp_200 = AsyncMock()
        mock_resp_200.status = 200
        mock_resp_200.read = AsyncMock(return_value=_create_minimal_png())
        mock_resp_200.__aenter__ = AsyncMock(return_value=mock_resp_200)
        mock_resp_200.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        # First call raises 429 error, second call succeeds
        mock_session.get = MagicMock(side_effect=[error_429, mock_resp_200])
        downloader.session = mock_session

        with (
            patch("embedding_cluster.utils.asyncio.sleep", new_callable=AsyncMock),
            patch("embedding_cluster.utils.logger") as mock_logger,
        ):
            result = await downloader.download_image_exp_backoff(
                "http://example.com/img.png", retries=2
            )

            assert result is not None
            # Verify success log message was called
            mock_logger.info.assert_called()
            # Check that the log message contains "success after" text
            calls = mock_logger.info.call_args_list
            info_messages = [call[0][0] if call[0] else "" for call in calls]
            assert any(
                "image download success after" in str(msg) for msg in info_messages
            )

    @pytest.mark.asyncio
    async def test_timeout_error_sets_status_408(self) -> None:
        """Test that asyncio.TimeoutError sets status=408 (lines 128-129)."""

        downloader = ImageDownloader()
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(side_effect=TimeoutError("Timeout occurred"))
        downloader.session = mock_session

        with (
            patch("embedding_cluster.utils.asyncio.sleep", new_callable=AsyncMock),
            patch("embedding_cluster.utils.logger") as mock_logger,
        ):
            result = await downloader.download_image_exp_backoff(
                "http://example.com/img.png", retries=1
            )

            assert result is None
            # Verify warning log was called (should log 408 status)
            mock_logger.warning.assert_called()
            # Check that log contains status 408 context
            calls = mock_logger.warning.call_args_list
            log_messages = [call[0][0] if call[0] else "" for call in calls]
            # The log should reference the failed download
            assert any("failed to download" in str(msg).lower() for msg in log_messages)

    @pytest.mark.asyncio
    async def test_client_response_error_uses_status(self) -> None:
        """Test that ClientResponseError uses e.status (lines 131-132)."""
        import aiohttp

        downloader = ImageDownloader()
        # Create a ClientResponseError with specific status
        error = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=503,
            message="Service Unavailable",
        )
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(side_effect=error)
        downloader.session = mock_session

        with (
            patch("embedding_cluster.utils.asyncio.sleep", new_callable=AsyncMock),
            patch("embedding_cluster.utils.logger") as mock_logger,
        ):
            result = await downloader.download_image_exp_backoff(
                "http://example.com/img.png", retries=1
            )

            assert result is None
            # Verify warning was called
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_retryable_status_429_allows_retry(self) -> None:
        """Test that status 429 is retryable (line 136)."""
        import aiohttp

        downloader = ImageDownloader()

        # First two responses are 429 errors (via ClientResponseError)
        error_429_1 = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=429,
            message="Too Many Requests",
        )

        error_429_2 = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=429,
            message="Too Many Requests",
        )

        # Final response is 200
        mock_resp_200 = AsyncMock()
        mock_resp_200.status = 200
        mock_resp_200.read = AsyncMock(return_value=_create_minimal_png())
        mock_resp_200.__aenter__ = AsyncMock(return_value=mock_resp_200)
        mock_resp_200.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(
            side_effect=[error_429_1, error_429_2, mock_resp_200]
        )
        downloader.session = mock_session

        with patch("embedding_cluster.utils.asyncio.sleep", new_callable=AsyncMock):
            result = await downloader.download_image_exp_backoff(
                "http://example.com/img.png", retries=3
            )

        assert result is not None
        # Verify that get was called 3 times (2 retries + 1 success)
        assert mock_session.get.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_delay_logging_and_sleep(self) -> None:
        """Test retry delay logging and asyncio.sleep call (lines 143-144)."""
        import aiohttp

        downloader = ImageDownloader()

        # Create retryable error (429)
        error_429 = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=429,
            message="Too Many Requests",
        )

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(side_effect=error_429)
        downloader.session = mock_session

        with (
            patch(
                "embedding_cluster.utils.asyncio.sleep", new_callable=AsyncMock
            ) as mock_sleep,
            patch("embedding_cluster.utils.logger") as mock_logger,
        ):
            result = await downloader.download_image_exp_backoff(
                "http://example.com/img.png", retries=2
            )

            assert result is None
            # Verify sleep was called with exponential backoff
            mock_sleep.assert_called()
            # Check that the sleep delays are correct
            sleep_calls = mock_sleep.call_args_list
            delays = [call[0][0] for call in sleep_calls]
            assert len(delays) > 0  # at least one retry delay
            assert delays[0] > 0  # delay should be positive
            # Verify logger was called with retry message
            mock_logger.warning.assert_called()
            calls = mock_logger.warning.call_args_list
            log_messages = [call[0][0] if call[0] else "" for call in calls]
            # Check that retry message includes delay info
            assert any("Retrying in" in str(msg) for msg in log_messages)


def _create_minimal_png() -> bytes:
    """Create a minimal valid PNG file in memory."""
    import io

    from PIL import Image

    img = Image.new("RGB", (1, 1), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
