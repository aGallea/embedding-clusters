from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from embedding_cluster.annotations import AnnotationManager

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def annotations_dir(tmp_path: Path) -> Path:
    return tmp_path / "annotations"


@pytest.fixture
def manager(annotations_dir: Path) -> AnnotationManager:
    return AnnotationManager(base_dir=annotations_dir)


class TestAnnotationManager:
    def test_get_empty_annotations(self, manager: AnnotationManager) -> None:
        result = manager.get_annotations("job1")
        assert result == {"job_id": "job1", "clusters": {}}

    def test_update_cluster_annotation(self, manager: AnnotationManager) -> None:
        manager.update_annotation("job1", 0, name="Shoes", notes="Athletic shoes")
        result = manager.get_annotations("job1")
        assert "0" in result["clusters"]
        assert result["clusters"]["0"]["name"] == "Shoes"
        assert result["clusters"]["0"]["notes"] == "Athletic shoes"
        assert result["clusters"]["0"]["updated_at"] is not None

    def test_update_partial_fields(self, manager: AnnotationManager) -> None:
        manager.update_annotation("job1", 0, name="Shoes")
        manager.update_annotation("job1", 0, notes="Running shoes")
        result = manager.get_annotations("job1")
        assert result["clusters"]["0"]["name"] == "Shoes"
        assert result["clusters"]["0"]["notes"] == "Running shoes"

    def test_update_tags(self, manager: AnnotationManager) -> None:
        manager.update_annotation("job1", 0, tags=["footwear", "sport"])
        result = manager.get_annotations("job1")
        assert result["clusters"]["0"]["tags"] == [
            "footwear",
            "sport",
        ]

    def test_multiple_clusters(self, manager: AnnotationManager) -> None:
        manager.update_annotation("job1", 0, name="Shoes")
        manager.update_annotation("job1", 1, name="Hats")
        result = manager.get_annotations("job1")
        assert len(result["clusters"]) == 2
        assert result["clusters"]["0"]["name"] == "Shoes"
        assert result["clusters"]["1"]["name"] == "Hats"

    def test_persistence(self, annotations_dir: Path) -> None:
        manager1 = AnnotationManager(base_dir=annotations_dir)
        manager1.update_annotation("job1", 0, name="Shoes")
        manager2 = AnnotationManager(base_dir=annotations_dir)
        result = manager2.get_annotations("job1")
        assert result["clusters"]["0"]["name"] == "Shoes"

    def test_delete_annotations(self, manager: AnnotationManager) -> None:
        manager.update_annotation("job1", 0, name="Shoes")
        manager.delete_annotations("job1")
        result = manager.get_annotations("job1")
        assert result == {"job_id": "job1", "clusters": {}}

    def test_file_created_in_correct_dir(
        self,
        manager: AnnotationManager,
        annotations_dir: Path,
    ) -> None:
        manager.update_annotation("job1", 0, name="Shoes")
        file_path = annotations_dir / "job1.json"
        assert file_path.exists()
        data = json.loads(file_path.read_text())
        assert data["job_id"] == "job1"

    def test_multiple_jobs(self, manager: AnnotationManager) -> None:
        manager.update_annotation("job1", 0, name="Shoes")
        manager.update_annotation("job2", 0, name="Hats")
        r1 = manager.get_annotations("job1")
        r2 = manager.get_annotations("job2")
        assert r1["clusters"]["0"]["name"] == "Shoes"
        assert r2["clusters"]["0"]["name"] == "Hats"
