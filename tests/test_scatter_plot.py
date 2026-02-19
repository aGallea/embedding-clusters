from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np

if TYPE_CHECKING:
    import pytest

from embedding_cluster.settings import Settings


class TestGetFieldAsList:
    def test_basic(self) -> None:
        from embedding_cluster.scatter_plot import get_field_as_list

        metadata = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        result = get_field_as_list(metadata, "name")
        assert result == ["a", "b", "c"]


class TestCreateCollectionTextDisplay:
    def test_single_field(self) -> None:
        from embedding_cluster.scatter_plot import (
            create_collection_text_display,
        )

        metadata = [{"name": "Alice"}, {"name": "Bob"}]
        result = create_collection_text_display(metadata, ["name"])
        assert result == ["Alice", "Bob"]

    def test_multiple_fields(self) -> None:
        from embedding_cluster.scatter_plot import (
            create_collection_text_display,
        )

        metadata = [
            {"name": "Alice", "city": "NYC"},
            {"name": "Bob", "city": "LA"},
        ]
        result = create_collection_text_display(metadata, ["name", "city"])
        assert result == ["Alice,NYC", "Bob,LA"]

    def test_custom_separator(self) -> None:
        from embedding_cluster.scatter_plot import (
            create_collection_text_display,
        )

        metadata = [{"a": "1", "b": "2"}]
        result = create_collection_text_display(metadata, ["a", "b"], seperator=" | ")
        assert result == ["1 | 2"]


class TestGenerateClusterProps:
    def test_without_gpt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from embedding_cluster.scatter_plot import generate_cluster_props

        settings = Settings()
        pred_arr = [0, 0, 1, 1, 2]
        text_display = ["a", "b", "c", "d", "e"]

        clusters_indices, cluster_names = generate_cluster_props(
            num_clusters=3,
            pred_arr=pred_arr,
            collection_content_text_display=text_display,
            settings=settings,
        )

        assert len(clusters_indices) == 3
        assert len(cluster_names) == 3
        assert clusters_indices[0] == [0, 1]
        assert clusters_indices[1] == [2, 3]
        assert clusters_indices[2] == [4]
        assert cluster_names == ["Group 1", "Group 2", "Group 3"]

    def test_with_gpt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from embedding_cluster.scatter_plot import generate_cluster_props

        monkeypatch.setenv("GPT_GENERATE_CLUSTER_NAME", "true")
        settings = Settings()
        pred_arr = [0, 0]
        text_display = ["item1", "item2"]

        with patch(
            "embedding_cluster.scatter_plot.gpt_get_cluster_name",
            return_value="Cool Group",
        ):
            _clusters_indices, cluster_names = generate_cluster_props(
                num_clusters=1,
                pred_arr=pred_arr,
                collection_content_text_display=text_display,
                settings=settings,
            )

        assert cluster_names == ["Cool Group"]


class TestGptGetClusterName:
    def test_gpt_get_cluster_name(self) -> None:
        from embedding_cluster.scatter_plot import gpt_get_cluster_name

        settings = Settings()

        with patch("embedding_cluster.scatter_plot.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_completion = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "Fashion Items"
            mock_completion.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_completion

            result = gpt_get_cluster_name("item1\nitem2", settings)

        assert result == "Fashion Items"

    def test_gpt_truncates_long_name(self) -> None:
        from embedding_cluster.scatter_plot import gpt_get_cluster_name

        settings = Settings()

        with patch("embedding_cluster.scatter_plot.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_completion = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "A" * 50
            mock_completion.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_completion

            result = gpt_get_cluster_name("info", settings)

        assert len(result) == 32  # 30 chars + ".."

    def test_gpt_none_content(self) -> None:
        from embedding_cluster.scatter_plot import gpt_get_cluster_name

        settings = Settings()

        with patch("embedding_cluster.scatter_plot.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_completion = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = None
            mock_completion.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_completion

            result = gpt_get_cluster_name("info", settings)

        assert result == ""


class TestLoadChromadbCollection:
    def test_load(self) -> None:
        from embedding_cluster.scatter_plot import (
            load_chromadb_collection,
        )

        settings = Settings()

        with patch("embedding_cluster.scatter_plot.chromadb") as mock_chromadb:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.get.return_value = {
                "ids": ["1"],
                "embeddings": [[1.0]],
                "metadatas": [{"k": "v"}],
            }
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            result = load_chromadb_collection(settings)

        assert result["ids"] == ["1"]


class TestDisplayHover:
    def test_display_hover_none(self) -> None:
        from embedding_cluster.scatter_plot import display_hover

        show, _bbox, _children = display_hover(None)
        assert show is False

    def test_display_hover_with_data(self) -> None:
        import embedding_cluster.scatter_plot as sp
        from embedding_cluster.scatter_plot import display_hover

        # Set module-level globals
        sp.cluster_images = [["http://img.png"]]
        sp.cluster_item_names = [["Item A"]]

        hover_data = {"points": [{"bbox": {"x0": 0}, "pointNumber": 0, "curveNumber": 0}]}

        show, bbox, children = display_hover(hover_data)

        assert show is True
        assert bbox == {"x0": 0}
        assert len(children) == 1

        # Cleanup
        sp.cluster_images = []
        sp.cluster_item_names = []


class TestPrepareData:
    def test_prepare_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import plotly.graph_objects as go

        import embedding_cluster.scatter_plot as sp
        from embedding_cluster.scatter_plot import prepare_data

        monkeypatch.setenv("NUM_CLUSTERS", "2")
        monkeypatch.setenv("TEXT_DISPLAY_FIELDS", '["name"]')
        monkeypatch.setenv("IMAGE_FIELD", "img")
        settings = Settings()

        fake_collection = {
            "ids": ["1", "2", "3", "4"],
            "embeddings": np.random.default_rng(42).random((4, 10)).tolist(),
            "metadatas": [
                {"name": f"item{i}", "img": f"http://img{i}.png"} for i in range(4)
            ],
        }

        # Clear module globals before test
        sp.cluster_images = []
        sp.cluster_item_names = []

        with (
            patch(
                "embedding_cluster.scatter_plot.load_chromadb_collection",
                return_value=fake_collection,
            ),
            patch("embedding_cluster.scatter_plot.TSNE") as mock_tsne_cls,
            patch("embedding_cluster.scatter_plot.KMeans") as mock_kmeans_cls,
            patch("embedding_cluster.scatter_plot.StandardScaler") as mock_scaler_cls,
        ):
            mock_tsne = MagicMock()
            mock_tsne.fit_transform.return_value = np.array(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
            )
            mock_tsne_cls.return_value = mock_tsne

            mock_kmeans = MagicMock()
            mock_kmeans.fit_predict.return_value = [0, 0, 1, 1]
            mock_kmeans_cls.return_value = mock_kmeans

            mock_scaler = MagicMock()
            mock_scaler.fit_transform.return_value = np.array(
                fake_collection["embeddings"]
            )
            mock_scaler_cls.return_value = mock_scaler

            fig = prepare_data(settings)

        assert isinstance(fig, go.Figure)
        assert len(sp.cluster_images) == 2
        assert len(sp.cluster_item_names) == 2

        # Cleanup
        sp.cluster_images = []
        sp.cluster_item_names = []
