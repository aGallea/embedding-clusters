from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
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
            patch(
                "embedding_cluster.scatter_plot.reduce_dimensions",
                return_value=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            ),
            patch("embedding_cluster.scatter_plot.KMeans") as mock_kmeans_cls,
            patch("embedding_cluster.scatter_plot.StandardScaler") as mock_scaler_cls,
        ):
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


class TestLoadChromadbEmbeddings:
    def test_load_success(self) -> None:
        from embedding_cluster.scatter_plot import load_chromadb_embeddings

        with patch("embedding_cluster.scatter_plot.chromadb") as mock_chromadb:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.get.return_value = {
                "embeddings": [[1.0, 2.0], [3.0, 4.0]],
            }
            mock_client.get_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            result = load_chromadb_embeddings("test_collection")

        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_collection_not_found(self) -> None:
        from embedding_cluster.scatter_plot import load_chromadb_embeddings

        with patch("embedding_cluster.scatter_plot.chromadb") as mock_chromadb:
            mock_client = MagicMock()
            mock_client.get_collection.side_effect = Exception("not found")
            mock_chromadb.PersistentClient.return_value = mock_client

            with pytest.raises(ValueError, match="not found"):
                load_chromadb_embeddings("nonexistent")

    def test_no_embeddings(self) -> None:
        from embedding_cluster.scatter_plot import load_chromadb_embeddings

        with patch("embedding_cluster.scatter_plot.chromadb") as mock_chromadb:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.get.return_value = {"embeddings": None}
            mock_client.get_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            with pytest.raises(ValueError, match="No embeddings found"):
                load_chromadb_embeddings("empty_collection")


class TestSuggestOptimalClusters:
    def test_suggest_optimal_clusters_progress_callback(self) -> None:
        from embedding_cluster.scatter_plot import suggest_optimal_clusters

        rng = np.random.default_rng(42)
        cluster1 = rng.normal(loc=0.0, scale=0.1, size=(20, 10))
        cluster2 = rng.normal(loc=5.0, scale=0.1, size=(20, 10))
        cluster3 = rng.normal(loc=10.0, scale=0.1, size=(20, 10))
        embeddings = np.vstack([cluster1, cluster2, cluster3])

        progress_updates: list[dict[str, object]] = []

        def on_progress(info: dict[str, object]) -> None:
            progress_updates.append(info)

        result = suggest_optimal_clusters(
            embeddings, k_range=range(2, 6), on_progress=on_progress
        )

        assert result["suggested_k"] in list(range(2, 6))
        assert len(progress_updates) == 4
        assert progress_updates[0]["phase"] == "analyzing"
        assert progress_updates[0]["current_k"] == 2
        assert progress_updates[0]["total_k"] == 4
        assert progress_updates[-1]["current_k"] == 5

    def test_returns_correct_structure(self) -> None:
        from embedding_cluster.scatter_plot import suggest_optimal_clusters

        rng = np.random.default_rng(42)
        cluster1 = rng.normal(loc=0.0, scale=0.1, size=(20, 10))
        cluster2 = rng.normal(loc=5.0, scale=0.1, size=(20, 10))
        cluster3 = rng.normal(loc=10.0, scale=0.1, size=(20, 10))
        embeddings = np.vstack([cluster1, cluster2, cluster3])

        result = suggest_optimal_clusters(embeddings, k_range=range(2, 11))

        assert "k_values" in result
        assert "inertias" in result
        assert "silhouette_scores" in result
        assert "suggested_k" in result
        assert result["k_values"] == list(range(2, 11))
        assert len(result["inertias"]) == 9
        assert len(result["silhouette_scores"]) == 9

    def test_silhouette_scores_in_range(self) -> None:
        from embedding_cluster.scatter_plot import suggest_optimal_clusters

        rng = np.random.default_rng(42)
        embeddings = rng.random((60, 10))

        result = suggest_optimal_clusters(embeddings, k_range=range(2, 11))

        for score in result["silhouette_scores"]:
            assert -1.0 <= score <= 1.0

    def test_suggested_k_within_range(self) -> None:
        from embedding_cluster.scatter_plot import suggest_optimal_clusters

        rng = np.random.default_rng(42)
        embeddings = rng.random((60, 10))

        result = suggest_optimal_clusters(embeddings, k_range=range(2, 11))

        assert 2 <= result["suggested_k"] <= 10

    def test_inertia_monotonically_decreasing(self) -> None:
        from embedding_cluster.scatter_plot import suggest_optimal_clusters

        rng = np.random.default_rng(42)
        embeddings = rng.random((60, 10))

        result = suggest_optimal_clusters(embeddings, k_range=range(2, 11))

        inertias = result["inertias"]
        for i in range(1, len(inertias)):
            assert inertias[i] <= inertias[i - 1]

    def test_single_k_value(self) -> None:
        from embedding_cluster.scatter_plot import suggest_optimal_clusters

        rng = np.random.default_rng(42)
        embeddings = rng.random((20, 10))

        result = suggest_optimal_clusters(embeddings, k_range=range(5, 6))

        assert result["k_values"] == [5]
        assert len(result["inertias"]) == 1
        assert len(result["silhouette_scores"]) == 1
        assert result["suggested_k"] == 5

    def test_well_separated_clusters_suggest_correct_k(self) -> None:
        from embedding_cluster.scatter_plot import suggest_optimal_clusters

        rng = np.random.default_rng(42)
        cluster1 = rng.normal(loc=0.0, scale=0.1, size=(30, 10))
        cluster2 = rng.normal(loc=10.0, scale=0.1, size=(30, 10))
        cluster3 = rng.normal(loc=20.0, scale=0.1, size=(30, 10))
        embeddings = np.vstack([cluster1, cluster2, cluster3])

        result = suggest_optimal_clusters(embeddings, k_range=range(2, 11))

        assert result["suggested_k"] == 3

    def test_large_dataset_uses_sample(self) -> None:
        from embedding_cluster.scatter_plot import suggest_optimal_clusters

        rng = np.random.default_rng(42)
        embeddings = rng.random((10000, 10))

        result = suggest_optimal_clusters(
            embeddings, k_range=range(2, 6), max_samples=500
        )

        assert result["k_values"] == list(range(2, 6))
        assert 2 <= result["suggested_k"] <= 5


class TestReduceDimensions:
    def test_tsne_output_shape(self) -> None:
        from embedding_cluster.scatter_plot import reduce_dimensions

        rng = np.random.default_rng(42)
        embeddings = rng.random((50, 10))

        result = reduce_dimensions(embeddings, algorithm="tsne", n_components=3)

        assert result.shape == (50, 3)

    def test_pca_output_shape(self) -> None:
        from embedding_cluster.scatter_plot import reduce_dimensions

        rng = np.random.default_rng(42)
        embeddings = rng.random((30, 10))

        result = reduce_dimensions(embeddings, algorithm="pca", n_components=3)

        assert result.shape == (30, 3)

    def test_pca_deterministic(self) -> None:
        from embedding_cluster.scatter_plot import reduce_dimensions

        rng = np.random.default_rng(42)
        embeddings = rng.random((30, 10))

        result1 = reduce_dimensions(embeddings, algorithm="pca", n_components=3)
        result2 = reduce_dimensions(embeddings, algorithm="pca", n_components=3)

        np.testing.assert_array_equal(result1, result2)

    def test_tsne_custom_perplexity(self) -> None:
        from embedding_cluster.scatter_plot import reduce_dimensions

        rng = np.random.default_rng(42)
        embeddings = rng.random((30, 10))

        result = reduce_dimensions(
            embeddings,
            algorithm="tsne",
            n_components=3,
            perplexity=10.0,
        )

        assert result.shape == (30, 3)

    def test_umap_output_shape(self) -> None:
        from embedding_cluster.scatter_plot import reduce_dimensions

        rng = np.random.default_rng(42)
        embeddings = rng.random((30, 10))

        try:
            result = reduce_dimensions(
                embeddings,
                algorithm="umap",
                n_components=3,
                n_neighbors=5,
                min_dist=0.1,
            )
            assert result.shape == (30, 3)
        except ImportError:
            pytest.skip("umap-learn not installed")

    def test_umap_import_guard(self) -> None:
        from embedding_cluster.scatter_plot import reduce_dimensions

        rng = np.random.default_rng(42)
        embeddings = rng.random((30, 10))

        with (
            patch.dict("sys.modules", {"umap": None}),
            pytest.raises(ImportError, match="umap-learn is not installed"),
        ):
            reduce_dimensions(embeddings, algorithm="umap", n_components=3)

    def test_invalid_algorithm_raises(self) -> None:
        from embedding_cluster.scatter_plot import reduce_dimensions

        rng = np.random.default_rng(42)
        embeddings = rng.random((30, 10))

        with pytest.raises(ValueError, match="Unknown reduction algorithm"):
            reduce_dimensions(embeddings, algorithm="invalid", n_components=3)

    def test_pca_two_components(self) -> None:
        from embedding_cluster.scatter_plot import reduce_dimensions

        rng = np.random.default_rng(42)
        embeddings = rng.random((30, 10))

        result = reduce_dimensions(embeddings, algorithm="pca", n_components=2)

        assert result.shape == (30, 2)


class TestComputePlotData:
    def test_filters_metadata_to_display_fields(self) -> None:
        from embedding_cluster.scatter_plot import compute_plot_data
        from embedding_cluster.settings import Settings

        collection_content = {
            "ids": ["1", "2"],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "metadatas": [
                {"name": "item1", "imageUrl": "url1", "category": "a"},
                {"name": "item2", "imageUrl": "url2", "category": "b"},
            ],
        }

        settings = Settings(
            running_mode="PLOT",
            chromadb_collection_name="test_collection",
            num_clusters=1,
            reduction_algorithm="pca",
            text_display_fields=["name"],
            image_field="imageUrl",
        )

        with (
            patch(
                "embedding_cluster.scatter_plot.load_chromadb_collection",
                return_value=collection_content,
            ),
            patch(
                "embedding_cluster.scatter_plot.reduce_dimensions",
                return_value=np.zeros((2, 3)),
            ),
            patch(
                "embedding_cluster.scatter_plot.KMeans.fit_predict",
                return_value=np.array([0, 0]),
            ),
        ):
            result = compute_plot_data(settings)

        points = result["points"]
        assert len(points) == 2
        assert points[0]["metadata"] == {"name": "item1"}
        assert points[1]["metadata"] == {"name": "item2"}

    def test_defaults_to_all_metadata_when_no_fields_selected(self) -> None:
        from embedding_cluster.scatter_plot import compute_plot_data
        from embedding_cluster.settings import Settings

        collection_content = {
            "ids": ["1"],
            "embeddings": [[0.1, 0.2]],
            "metadatas": [
                {"name": "item1", "imageUrl": "url1", "category": "a"},
            ],
        }

        settings = Settings(
            running_mode="PLOT",
            chromadb_collection_name="test_collection",
            num_clusters=1,
            reduction_algorithm="pca",
            text_display_fields=[],
            image_field="imageUrl",
        )

        with (
            patch(
                "embedding_cluster.scatter_plot.load_chromadb_collection",
                return_value=collection_content,
            ),
            patch(
                "embedding_cluster.scatter_plot.reduce_dimensions",
                return_value=np.zeros((1, 3)),
            ),
            patch(
                "embedding_cluster.scatter_plot.KMeans.fit_predict",
                return_value=np.array([0]),
            ),
        ):
            result = compute_plot_data(settings)

        points = result["points"]
        assert len(points) == 1
        assert points[0]["metadata"] == {
            "name": "item1",
            "imageUrl": "url1",
            "category": "a",
        }
