from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

if TYPE_CHECKING:
    import pytest

from embedding_cluster.__main__ import main


class TestMain:
    def test_index_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RUNNING_MODE", "INDEX")

        with (
            patch("embedding_cluster.__main__.init_logger") as mock_logger,
            patch(
                "embedding_cluster.indexer.main_indexer",
                new_callable=AsyncMock,
            ) as mock_indexer,
        ):
            main()

            mock_logger.assert_called_once()
            mock_indexer.assert_called_once()

    def test_plot_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RUNNING_MODE", "PLOT")

        with (
            patch("embedding_cluster.__main__.init_logger") as mock_logger,
            patch(
                "embedding_cluster.scatter_plot.main_scatter_plot",
                new_callable=AsyncMock,
            ) as mock_plot,
        ):
            main()

            mock_logger.assert_called_once()
            mock_plot.assert_called_once()

    def test_server_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RUNNING_MODE", "SERVER")

        with (
            patch("embedding_cluster.__main__.init_logger") as mock_logger,
            patch("uvicorn.run") as mock_uvicorn_run,
        ):
            main()

            mock_logger.assert_called_once()
            mock_uvicorn_run.assert_called_once()
            call_kwargs = mock_uvicorn_run.call_args
            assert call_kwargs[1]["host"] == "0.0.0.0"
            assert call_kwargs[1]["port"] == 8000

    def test_unknown_mode_does_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RUNNING_MODE", "UNKNOWN")

        with patch("embedding_cluster.__main__.init_logger") as mock_logger:
            main()
            mock_logger.assert_called_once()
