from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_BASE_DIR = Path("./annotations")


class AnnotationManager:
    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or _DEFAULT_BASE_DIR
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, job_id: str) -> Path:
        return self._base_dir / f"{job_id}.json"

    def _read(self, job_id: str) -> dict[str, Any]:
        path = self._file_path(job_id)
        if not path.exists():
            return {"job_id": job_id, "clusters": {}}
        data: dict[str, Any] = json.loads(path.read_text())
        return data

    def _write(self, job_id: str, data: dict[str, Any]) -> None:
        path = self._file_path(job_id)
        path.write_text(json.dumps(data, indent=2))

    def get_annotations(self, job_id: str) -> dict[str, Any]:
        return self._read(job_id)

    def update_annotation(
        self,
        job_id: str,
        cluster_index: int,
        name: str | None = None,
        notes: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        data = self._read(job_id)
        key = str(cluster_index)
        if key not in data["clusters"]:
            data["clusters"][key] = {
                "name": None,
                "notes": None,
                "tags": None,
                "updated_at": None,
            }
        cluster = data["clusters"][key]
        if name is not None:
            cluster["name"] = name
        if notes is not None:
            cluster["notes"] = notes
        if tags is not None:
            cluster["tags"] = tags
        cluster["updated_at"] = datetime.now(
            tz=timezone.utc  # noqa: UP017
        ).isoformat()
        self._write(job_id, data)
        return data

    def delete_annotations(self, job_id: str) -> None:
        path = self._file_path(job_id)
        if path.exists():
            path.unlink()
