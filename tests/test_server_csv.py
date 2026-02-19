from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from embedding_cluster.server.app import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_csv_content():
    """Create sample CSV content for testing."""
    return b"""id,name,category,price
1,Product A,Electronics,99.99
2,Product B,Clothing,29.99
3,Product C,Electronics,149.99
4,Product D,Home,19.99
5,Product E,Clothing,39.99
"""


@pytest.fixture
def mock_upload_dir(tmp_path, monkeypatch):
    """Mock the upload directory to use a temporary path."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    # Patch the UPLOAD_DIR in the csv module
    with patch("embedding_cluster.server.routes.csv.UPLOAD_DIR", upload_dir):
        yield upload_dir


async def test_upload_csv_success(client, sample_csv_content, mock_upload_dir):
    """Test uploading a valid CSV file."""
    files = {"file": ("test.csv", sample_csv_content, "text/csv")}

    response = await client.post("/api/csv/upload", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.csv"
    assert data["rows"] == 5
    assert data["columns"] == ["id", "name", "category", "price"]

    # Verify file was saved
    saved_file = mock_upload_dir / "test.csv"
    assert saved_file.exists()


async def test_upload_csv_invalid_format(client, mock_upload_dir):
    """Test uploading invalid CSV content."""
    # Invalid CSV with unclosed quotes
    invalid_csv = b'id,name\n1,"unclosed quote\n2,valid'
    files = {"file": ("invalid.csv", invalid_csv, "text/csv")}

    response = await client.post("/api/csv/upload", files=files)

    # Should handle gracefully - depending on csv module behavior
    # Most likely will parse it, but let's verify no server crash
    assert response.status_code in [200, 400]


async def test_preview_csv_success(client, sample_csv_content, mock_upload_dir):
    """Test previewing an uploaded CSV file."""
    # First upload the file
    files = {"file": ("preview_test.csv", sample_csv_content, "text/csv")}
    upload_response = await client.post("/api/csv/upload", files=files)
    assert upload_response.status_code == 200

    # Now preview it
    preview_request = {"filename": "preview_test.csv", "limit": 3}
    response = await client.post("/api/csv/preview", json=preview_request)

    assert response.status_code == 200
    data = response.json()
    assert data["columns"] == ["id", "name", "category", "price"]
    assert len(data["rows"]) == 3
    assert data["total_rows"] == 5
    assert data["rows"][0] == {
        "id": "1",
        "name": "Product A",
        "category": "Electronics",
        "price": "99.99",
    }


async def test_preview_csv_custom_limit(client, sample_csv_content, mock_upload_dir):
    """Test previewing CSV with custom row limit."""
    # Upload file
    files = {"file": ("limit_test.csv", sample_csv_content, "text/csv")}
    await client.post("/api/csv/upload", files=files)

    # Preview with limit of 2
    preview_request = {"filename": "limit_test.csv", "limit": 2}
    response = await client.post("/api/csv/preview", json=preview_request)

    assert response.status_code == 200
    data = response.json()
    assert len(data["rows"]) == 2
    assert data["total_rows"] == 5


async def test_preview_csv_default_limit(client, sample_csv_content, mock_upload_dir):
    """Test previewing CSV with default limit of 10."""
    # Upload file
    files = {"file": ("default_limit.csv", sample_csv_content, "text/csv")}
    await client.post("/api/csv/upload", files=files)

    # Preview without specifying limit (should default to 10)
    preview_request = {"filename": "default_limit.csv"}
    response = await client.post("/api/csv/preview", json=preview_request)

    assert response.status_code == 200
    data = response.json()
    # Sample has only 5 rows, so we get all 5
    assert len(data["rows"]) == 5
    assert data["total_rows"] == 5


async def test_preview_csv_not_found(client, mock_upload_dir):
    """Test previewing a non-existent CSV file."""
    preview_request = {"filename": "nonexistent.csv", "limit": 10}
    response = await client.post("/api/csv/preview", json=preview_request)

    assert response.status_code == 404
    data = response.json()
    assert "File not found" in data["detail"]


async def test_preview_csv_limit_exceeds_rows(
    client, sample_csv_content, mock_upload_dir
):
    """Test preview when limit exceeds total rows."""
    # Upload file with 5 rows
    files = {"file": ("exceed_test.csv", sample_csv_content, "text/csv")}
    await client.post("/api/csv/upload", files=files)

    # Request 100 rows
    preview_request = {"filename": "exceed_test.csv", "limit": 100}
    response = await client.post("/api/csv/preview", json=preview_request)

    assert response.status_code == 200
    data = response.json()
    # Should return all 5 rows, not 100
    assert len(data["rows"]) == 5
    assert data["total_rows"] == 5


async def test_upload_empty_filename(client, mock_upload_dir):
    """Test uploading file with no filename."""
    files = {"file": ("", b"id,name\n1,test", "text/csv")}

    response = await client.post("/api/csv/upload", files=files)

    # Should fail with 400
    assert response.status_code in [400, 422]
