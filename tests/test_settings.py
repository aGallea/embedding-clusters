from __future__ import annotations

import pytest

from embedding_cluster.settings import Settings


class TestSettingsDefaults:
    def test_default_running_mode(self) -> None:
        s = Settings()
        assert s.running_mode == "PLOT"

    def test_default_process_unit_device(self) -> None:
        s = Settings()
        assert s.process_unit_device == "cpu"

    def test_default_num_clusters(self) -> None:
        s = Settings()
        assert s.num_clusters == 10

    def test_default_index_bulk_size(self) -> None:
        s = Settings()
        assert s.index_bulk_size == 100

    def test_default_number_of_async_tasks(self) -> None:
        s = Settings()
        assert s.number_of_async_tasks == 1

    def test_default_none_fields(self) -> None:
        s = Settings()
        assert s.index_start_line is None
        assert s.index_end_line is None
        assert s.image_embedding_fields is None
        assert s.text_embedding_fields is None
        assert s.text_display_fields is None
        assert s.image_field is None
        assert s.id_field is None

    def test_default_model_names(self) -> None:
        s = Settings()
        assert s.image_model_name == "openai/clip-vit-base-patch32"
        assert s.text_model_name == "BAAI/bge-small-en-v1.5"

    def test_default_reduction_algorithm(self) -> None:
        s = Settings()
        assert s.reduction_algorithm == "tsne"

    def test_default_tsne_params(self) -> None:
        s = Settings()
        assert s.tsne_perplexity == pytest.approx(30.0)
        assert s.tsne_learning_rate == "auto"

    def test_default_umap_params(self) -> None:
        s = Settings()
        assert s.umap_n_neighbors == 15
        assert s.umap_min_dist == pytest.approx(0.1)
        assert s.umap_metric == "cosine"


class TestSettingsEnvVars:
    def test_running_mode_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RUNNING_MODE", "INDEX")
        s = Settings()
        assert s.running_mode == "INDEX"

    def test_num_clusters_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NUM_CLUSTERS", "20")
        s = Settings()
        assert s.num_clusters == 20

    def test_json_list_fields_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IMAGE_EMBEDDING_FIELDS", '["imageUrl", "thumbnailUrl"]')
        monkeypatch.setenv("TEXT_EMBEDDING_FIELDS", '["description"]')
        s = Settings()
        assert s.image_embedding_fields == ["imageUrl", "thumbnailUrl"]
        assert s.text_embedding_fields == ["description"]

    def test_text_display_fields_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEXT_DISPLAY_FIELDS", '["productDisplayName"]')
        s = Settings()
        assert s.text_display_fields == ["productDisplayName"]

    def test_start_end_lines_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INDEX_START_LINE", "5")
        monkeypatch.setenv("INDEX_END_LINE", "100")
        s = Settings()
        assert s.index_start_line == 5
        assert s.index_end_line == 100

    def test_local_csv_filename_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCAL_CSV_FILENAME", "/tmp/data.csv")
        s = Settings()
        assert s.local_csv_filename == "/tmp/data.csv"

    def test_reduction_algorithm_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REDUCTION_ALGORITHM", "pca")
        s = Settings()
        assert s.reduction_algorithm == "pca"

    def test_tsne_perplexity_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TSNE_PERPLEXITY", "50.0")
        s = Settings()
        assert s.tsne_perplexity == pytest.approx(50.0)

    def test_umap_params_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("UMAP_N_NEIGHBORS", "30")
        monkeypatch.setenv("UMAP_MIN_DIST", "0.5")
        monkeypatch.setenv("UMAP_METRIC", "euclidean")
        s = Settings()
        assert s.umap_n_neighbors == 30
        assert s.umap_min_dist == pytest.approx(0.5)
        assert s.umap_metric == "euclidean"
