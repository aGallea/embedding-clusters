from __future__ import annotations

import csv
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from embedding_cluster.server.models import CsvPreviewResponse, CsvUploadResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/csv", tags=["csv"])

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class PreviewRequest(BaseModel):
    filename: str
    limit: int = 10


@router.post("/upload", response_model=CsvUploadResponse)
async def upload_csv(file: UploadFile) -> CsvUploadResponse:
    """Upload a CSV file and return basic metadata."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        content = await file.read()

        with open(file_path, "wb") as f:
            f.write(content)

        # Read CSV to get metadata
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            row_count = sum(1 for _ in reader)

        logger.info(f"Uploaded CSV: {file.filename}, rows: {row_count}")

        return CsvUploadResponse(
            filename=file.filename, rows=row_count, columns=list(columns)
        )

    except csv.Error as e:
        logger.error(f"Invalid CSV format in {file.filename}: {e}")
        raise HTTPException(status_code=400, detail="Invalid CSV format") from e
    except Exception as e:
        logger.error(f"Error processing upload {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Error processing file") from e


@router.post("/preview", response_model=CsvPreviewResponse)
async def preview_csv(request: PreviewRequest) -> CsvPreviewResponse:
    """Preview uploaded CSV file with first N rows."""
    file_path = UPLOAD_DIR / request.filename

    if not file_path.exists():
        logger.error(f"File not found: {request.filename}")
        raise HTTPException(status_code=404, detail="File not found")

    try:
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []

            # Read all rows to get total count, but only keep first N
            all_rows = list(reader)
            total_rows = len(all_rows)
            preview_rows = all_rows[: request.limit]

        logger.info(
            f"Preview CSV: {request.filename}, total: {total_rows}, "
            f"showing: {len(preview_rows)}"
        )

        return CsvPreviewResponse(
            columns=list(columns), rows=preview_rows, total_rows=total_rows
        )

    except csv.Error as e:
        logger.error(f"Invalid CSV format in {request.filename}: {e}")
        raise HTTPException(status_code=400, detail="Invalid CSV format") from e
    except Exception as e:
        logger.error(f"Error reading {request.filename}: {e}")
        raise HTTPException(status_code=500, detail="Error reading file") from e
