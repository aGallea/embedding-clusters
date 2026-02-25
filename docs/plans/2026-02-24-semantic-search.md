# Semantic Search Within Clusters Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a search bar to the Plot page that accepts text or image URL input, generates an embedding, queries ChromaDB for nearest neighbors, and highlights matching points in the 3D scatter plot.

**Architecture:** New `POST /api/search` endpoint generates query embeddings using the same model that indexed the collection (CLIP for images, SentenceTransformer for text), then calls ChromaDB's `.query()`. Frontend adds a search bar + results panel to PlotPage, with highlighted points in all 3D render modes. Model type is passed explicitly in the search request since collections don't store model metadata.

**Tech Stack:** Python 3.13 / FastAPI / ChromaDB / CLIP / SentenceTransformer / React 19 / TypeScript / Zustand / Tailwind CSS / Three.js

---

## Task 1: Backend - Pydantic models for search request/response

**Files:**
- Modify: `embedding_cluster/server/models.py`

**Step 1: Add SearchRequest and SearchResponse models**

Add to the bottom of `embedding_cluster/server/models.py`:

```python
class SearchResult(BaseModel):
    id: str
    distance: float
    metadata: dict[str, Any]


class SearchRequest(BaseModel):
    collection_name: str
    query_text: str | None = None
    query_image_url: str | None = None
    n_results: int = 10
    model_type: str = "text"
    image_model_name: str = "openai/clip-vit-base-patch32"
    text_model_name: str = "BAAI/bge-small-en-v1.5"


class SearchResponse(BaseModel):
    results: list[SearchResult]
```

**Step 2: Run type check**

Run: `uv run mypy embedding_cluster/server/models.py`
Expected: PASS

---

## Task 2: Backend - Search route with embedding generation

**Files:**
- Create: `embedding_cluster/server/routes/search.py`
- Modify: `embedding_cluster/server/app.py` (register router)

**Step 1: Create the search route**

Create `embedding_cluster/server/routes/search.py`:

```python
from __future__ import annotations

import logging
from typing import Any

import chromadb
import torch
from fastapi import APIRouter, HTTPException
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from embedding_cluster.server.models import SearchRequest, SearchResponse, SearchResult
from embedding_cluster.utils import ImageDownloader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])

# Lazy-loaded model cache
_model_cache: dict[str, Any] = {}


def _get_chromadb_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path="./chromadb")


def _get_text_model(model_name: str) -> SentenceTransformer:
    cache_key = f"text:{model_name}"
    if cache_key not in _model_cache:
        logger.info("Loading text model: %s", model_name)
        _model_cache[cache_key] = SentenceTransformer(model_name)
    return _model_cache[cache_key]


def _get_image_model(
    model_name: str,
) -> tuple[CLIPModel, CLIPProcessor]:
    cache_key = f"image:{model_name}"
    if cache_key not in _model_cache:
        logger.info("Loading image model: %s", model_name)
        _model_cache[cache_key] = (
            CLIPModel.from_pretrained(model_name),
            CLIPProcessor.from_pretrained(model_name),
        )
    return _model_cache[cache_key]


async def _generate_text_embedding(
    query_text: str, model_name: str
) -> list[float]:
    model = _get_text_model(model_name)
    embedding = model.encode(query_text, show_progress_bar=False)
    return embedding.tolist()


async def _generate_image_embedding(
    image_url: str, model_name: str
) -> list[float]:
    image = await ImageDownloader().download_image_exp_backoff(image_url)
    if image is None:
        msg = f"Failed to download image from {image_url}"
        raise ValueError(msg)

    model, processor = _get_image_model(model_name)
    inputs = processor(
        text=None, images=image, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        img_features = model.get_image_features(inputs["pixel_values"])
    return img_features.squeeze(0).cpu().numpy().tolist()


@router.post("", response_model=SearchResponse)
async def search_collection(request: SearchRequest) -> SearchResponse:
    if not request.query_text and not request.query_image_url:
        raise HTTPException(
            status_code=400,
            detail="Either query_text or query_image_url is required",
        )

    client = _get_chromadb_client()
    try:
        collection = client.get_collection(request.collection_name)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection not found: {request.collection_name}",
        ) from e

    if collection.count() == 0:
        return SearchResponse(results=[])

    # Generate embedding
    if request.query_text:
        embedding = await _generate_text_embedding(
            request.query_text, request.text_model_name
        )
    else:
        assert request.query_image_url is not None
        embedding = await _generate_image_embedding(
            request.query_image_url, request.image_model_name
        )

    # Query ChromaDB
    query_result = collection.query(
        query_embeddings=[embedding],
        n_results=min(request.n_results, collection.count()),
    )

    results: list[SearchResult] = []
    if query_result["ids"] and query_result["distances"]:
        ids = query_result["ids"][0]
        distances = query_result["distances"][0]
        metadatas = (
            query_result["metadatas"][0]
            if query_result["metadatas"]
            else [{}] * len(ids)
        )
        for i, doc_id in enumerate(ids):
            results.append(
                SearchResult(
                    id=doc_id,
                    distance=distances[i],
                    metadata=metadatas[i] if metadatas[i] else {},
                )
            )

    return SearchResponse(results=results)
```

**Step 2: Register the search router in app.py**

Add import and `include_router` in `embedding_cluster/server/app.py`:

```python
from embedding_cluster.server.routes.search import router as search_router
# ...
app.include_router(search_router)
```

**Step 3: Run type check and lint**

Run: `uv run mypy embedding_cluster/server/routes/search.py`
Run: `uv run ruff check embedding_cluster/server/routes/search.py`
Expected: PASS (may need minor fixes)

---

## Task 3: Backend tests for search endpoint

**Files:**
- Create: `tests/test_server_search.py`

**Step 1: Write comprehensive tests**

Create `tests/test_server_search.py` following the pattern from `test_server_collections.py` and `test_server_plot.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
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
def mock_chromadb_client():
    mock_client = MagicMock()
    return mock_client


@pytest.fixture
def mock_collection(mock_chromadb_client):
    mock_coll = MagicMock()
    mock_coll.count.return_value = 100
    mock_coll.query.return_value = {
        "ids": [["id1", "id2", "id3"]],
        "distances": [[0.1, 0.3, 0.5]],
        "metadatas": [
            [
                {"name": "item1", "imageUrl": "http://example.com/1.jpg"},
                {"name": "item2", "imageUrl": "http://example.com/2.jpg"},
                {"name": "item3", "imageUrl": "http://example.com/3.jpg"},
            ]
        ],
    }
    mock_chromadb_client.get_collection.return_value = mock_coll
    return mock_coll


@pytest.fixture
def mock_text_model():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros(384)
    return mock_model


@pytest.fixture
def mock_image_model():
    mock_clip = MagicMock()
    mock_processor = MagicMock()

    import torch
    mock_features = torch.zeros(1, 512)
    mock_clip.get_image_features.return_value = mock_features

    return mock_clip, mock_processor


async def test_search_text_query(
    client, mock_chromadb_client, mock_collection, mock_text_model
):
    with (
        patch(
            "embedding_cluster.server.routes.search._get_chromadb_client",
            return_value=mock_chromadb_client,
        ),
        patch(
            "embedding_cluster.server.routes.search._get_text_model",
            return_value=mock_text_model,
        ),
    ):
        response = await client.post(
            "/api/search",
            json={
                "collection_name": "test_collection",
                "query_text": "red shoes",
                "n_results": 3,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 3
    assert data["results"][0]["id"] == "id1"
    assert data["results"][0]["distance"] == 0.1
    assert "metadata" in data["results"][0]


async def test_search_image_query(
    client, mock_chromadb_client, mock_collection, mock_image_model
):
    mock_clip, mock_processor = mock_image_model
    mock_image = MagicMock()

    with (
        patch(
            "embedding_cluster.server.routes.search._get_chromadb_client",
            return_value=mock_chromadb_client,
        ),
        patch(
            "embedding_cluster.server.routes.search._get_image_model",
            return_value=(mock_clip, mock_processor),
        ),
        patch(
            "embedding_cluster.server.routes.search.ImageDownloader"
        ) as mock_downloader_cls,
    ):
        mock_instance = MagicMock()
        mock_instance.download_image_exp_backoff = AsyncMock(
            return_value=mock_image
        )
        mock_downloader_cls.return_value = mock_instance

        response = await client.post(
            "/api/search",
            json={
                "collection_name": "test_collection",
                "query_image_url": "http://example.com/query.jpg",
                "model_type": "image",
                "n_results": 3,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 3


async def test_search_collection_not_found(client):
    mock_client = MagicMock()
    mock_client.get_collection.side_effect = Exception("Not found")

    with patch(
        "embedding_cluster.server.routes.search._get_chromadb_client",
        return_value=mock_client,
    ):
        response = await client.post(
            "/api/search",
            json={
                "collection_name": "nonexistent",
                "query_text": "test",
            },
        )

    assert response.status_code == 404
    assert "Collection not found" in response.json()["detail"]


async def test_search_empty_query(client):
    response = await client.post(
        "/api/search",
        json={"collection_name": "test_collection"},
    )
    assert response.status_code == 400


async def test_search_custom_n_results(
    client, mock_chromadb_client, mock_collection, mock_text_model
):
    with (
        patch(
            "embedding_cluster.server.routes.search._get_chromadb_client",
            return_value=mock_chromadb_client,
        ),
        patch(
            "embedding_cluster.server.routes.search._get_text_model",
            return_value=mock_text_model,
        ),
    ):
        response = await client.post(
            "/api/search",
            json={
                "collection_name": "test_collection",
                "query_text": "blue jacket",
                "n_results": 5,
            },
        )

    assert response.status_code == 200
    mock_collection.query.assert_called_once()
    call_kwargs = mock_collection.query.call_args
    assert call_kwargs.kwargs.get("n_results") == 5 or call_kwargs[1].get("n_results") == 5


async def test_search_response_structure(
    client, mock_chromadb_client, mock_collection, mock_text_model
):
    with (
        patch(
            "embedding_cluster.server.routes.search._get_chromadb_client",
            return_value=mock_chromadb_client,
        ),
        patch(
            "embedding_cluster.server.routes.search._get_text_model",
            return_value=mock_text_model,
        ),
    ):
        response = await client.post(
            "/api/search",
            json={
                "collection_name": "test_collection",
                "query_text": "test",
            },
        )

    data = response.json()
    assert "results" in data
    for result in data["results"]:
        assert "id" in result
        assert "distance" in result
        assert "metadata" in result
        assert isinstance(result["distance"], float)


async def test_search_empty_collection(client, mock_chromadb_client):
    mock_coll = MagicMock()
    mock_coll.count.return_value = 0
    mock_chromadb_client.get_collection.return_value = mock_coll

    with patch(
        "embedding_cluster.server.routes.search._get_chromadb_client",
        return_value=mock_chromadb_client,
    ):
        response = await client.post(
            "/api/search",
            json={
                "collection_name": "empty_collection",
                "query_text": "test",
            },
        )

    assert response.status_code == 200
    assert response.json()["results"] == []


async def test_search_image_download_failure(
    client, mock_chromadb_client, mock_collection
):
    with (
        patch(
            "embedding_cluster.server.routes.search._get_chromadb_client",
            return_value=mock_chromadb_client,
        ),
        patch(
            "embedding_cluster.server.routes.search.ImageDownloader"
        ) as mock_downloader_cls,
    ):
        mock_instance = MagicMock()
        mock_instance.download_image_exp_backoff = AsyncMock(
            return_value=None
        )
        mock_downloader_cls.return_value = mock_instance

        response = await client.post(
            "/api/search",
            json={
                "collection_name": "test_collection",
                "query_image_url": "http://example.com/bad.jpg",
                "model_type": "image",
            },
        )

    assert response.status_code == 500
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_server_search.py -v`
Expected: All PASS

---

## Task 4: Frontend - Types and API client for search

**Files:**
- Modify: `frontend/src/types/index.ts`
- Modify: `frontend/src/api/plot.ts`

**Step 1: Add search types**

Add to end of `frontend/src/types/index.ts`:

```typescript
// Search
export interface SearchResult {
  id: string;
  distance: number;
  metadata: Record<string, unknown>;
}

export interface SearchRequest {
  collection_name: string;
  query_text?: string;
  query_image_url?: string;
  n_results?: number;
  model_type?: string;
  image_model_name?: string;
  text_model_name?: string;
}

export interface SearchResponse {
  results: SearchResult[];
}
```

**Step 2: Add search API function**

Add to `frontend/src/api/plot.ts`:

```typescript
import type { SearchRequest, SearchResponse } from "../types";

export async function searchCollection(
  request: SearchRequest,
): Promise<SearchResponse> {
  return apiPost<SearchResponse>("/search", request);
}
```

---

## Task 5: Frontend - Search state in plotStore

**Files:**
- Modify: `frontend/src/stores/plotStore.ts`

**Step 1: Add search state and actions**

Add to the `PlotState` interface and store:

```typescript
// Add to interface:
searchResults: SearchResult[] | null
highlightedIds: Set<string>
isSearching: boolean

// Add actions:
setSearchResults: (results: SearchResult[] | null) => void
setHighlightedIds: (ids: Set<string>) => void
setIsSearching: (searching: boolean) => void
clearSearch: () => void

// Add to store implementation:
searchResults: null,
highlightedIds: new Set(),
isSearching: false,

setSearchResults: (results) => set({
  searchResults: results,
  highlightedIds: new Set(results?.map(r => r.id) ?? []),
}),
setHighlightedIds: (ids) => set({ highlightedIds: ids }),
setIsSearching: (searching) => set({ isSearching: searching }),
clearSearch: () => set({
  searchResults: null,
  highlightedIds: new Set(),
  isSearching: false,
}),
```

---

## Task 6: Frontend - SearchBar component

**Files:**
- Create: `frontend/src/components/plot/SearchBar.tsx`

**Step 1: Create SearchBar component**

```tsx
import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { searchCollection } from '../../api/plot'
import { usePlotStore } from '../../stores/plotStore'
import type { SearchRequest } from '../../types'

interface SearchBarProps {
  collectionName: string
}

export default function SearchBar({ collectionName }: SearchBarProps) {
  const [queryType, setQueryType] = useState<'text' | 'image'>('text')
  const [queryValue, setQueryValue] = useState('')
  const [nResults, setNResults] = useState(10)
  const { setSearchResults, setIsSearching, clearSearch } = usePlotStore()

  const mutation = useMutation({
    mutationFn: (request: SearchRequest) => searchCollection(request),
    onMutate: () => setIsSearching(true),
    onSuccess: (data) => {
      setSearchResults(data.results)
      setIsSearching(false)
    },
    onError: () => setIsSearching(false),
  })

  const handleSearch = () => {
    if (!queryValue.trim()) return
    const request: SearchRequest = {
      collection_name: collectionName,
      n_results: nResults,
      ...(queryType === 'text'
        ? { query_text: queryValue }
        : { query_image_url: queryValue, model_type: 'image' }),
    }
    mutation.mutate(request)
  }

  const handleClear = () => {
    setQueryValue('')
    clearSearch()
  }

  return (
    <div className="space-y-3 p-4 border-t border-gray-200">
      <h3 className="text-sm font-semibold text-gray-700">Semantic Search</h3>

      <div className="flex space-x-2">
        {(['text', 'image'] as const).map((type) => (
          <label key={type} className="flex items-center text-xs cursor-pointer">
            <input
              type="radio"
              name="queryType"
              value={type}
              checked={queryType === type}
              onChange={() => setQueryType(type)}
              className="mr-1"
            />
            {type === 'text' ? 'Text' : 'Image URL'}
          </label>
        ))}
      </div>

      <input
        type="text"
        value={queryValue}
        onChange={(e) => setQueryValue(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
        placeholder={queryType === 'text' ? 'Search by text...' : 'Paste image URL...'}
        className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm"
      />

      <div className="space-y-1">
        <label className="block text-xs text-gray-500">
          Results: {nResults}
        </label>
        <input
          type="range"
          min="5"
          max="50"
          value={nResults}
          onChange={(e) => setNResults(Number(e.target.value))}
          className="w-full"
        />
      </div>

      <div className="flex space-x-2">
        <button
          onClick={handleSearch}
          disabled={mutation.isPending || !queryValue.trim()}
          className={`flex-1 py-1.5 px-3 rounded text-white text-sm font-medium ${
            mutation.isPending || !queryValue.trim()
              ? 'bg-green-300 cursor-not-allowed'
              : 'bg-green-600 hover:bg-green-700'
          }`}
        >
          {mutation.isPending ? 'Searching...' : 'Search'}
        </button>
        <button
          onClick={handleClear}
          className="py-1.5 px-3 rounded border border-gray-300 text-sm text-gray-600 hover:bg-gray-50"
        >
          Clear
        </button>
      </div>

      {mutation.isError && (
        <p className="text-xs text-red-500">
          Search failed: {(mutation.error as Error).message}
        </p>
      )}
    </div>
  )
}
```

---

## Task 7: Frontend - SearchResults component

**Files:**
- Create: `frontend/src/components/plot/SearchResults.tsx`

**Step 1: Create SearchResults panel**

```tsx
import { usePlotStore } from '../../stores/plotStore'

export default function SearchResults() {
  const searchResults = usePlotStore((state) => state.searchResults)
  const highlightedIds = usePlotStore((state) => state.highlightedIds)
  const setHighlightedIds = usePlotStore((state) => state.setHighlightedIds)

  if (!searchResults || searchResults.length === 0) return null

  const handleClickResult = (id: string) => {
    setHighlightedIds(new Set([id]))
  }

  const handleShowAll = () => {
    setHighlightedIds(new Set(searchResults.map((r) => r.id)))
  }

  const getImageUrl = (metadata: Record<string, unknown>): string | null => {
    for (const value of Object.values(metadata)) {
      if (typeof value === 'string' && (value.startsWith('http') || value.startsWith('/'))) {
        return value
      }
    }
    return null
  }

  return (
    <div className="border-t border-gray-200 bg-white overflow-y-auto max-h-64">
      <div className="flex items-center justify-between p-3 border-b border-gray-100">
        <h3 className="text-sm font-semibold text-gray-700">
          Results ({searchResults.length})
        </h3>
        <button
          onClick={handleShowAll}
          className="text-xs text-blue-600 hover:text-blue-700"
        >
          Highlight All
        </button>
      </div>
      <div className="divide-y divide-gray-100">
        {searchResults.map((result) => {
          const imageUrl = getImageUrl(result.metadata)
          const isActive = highlightedIds.has(result.id)
          return (
            <button
              key={result.id}
              onClick={() => handleClickResult(result.id)}
              className={`w-full text-left p-3 hover:bg-blue-50 transition-colors flex items-center gap-3 ${
                isActive ? 'bg-blue-50 border-l-2 border-blue-500' : ''
              }`}
            >
              {imageUrl && (
                <img
                  src={imageUrl}
                  alt=""
                  className="w-10 h-10 object-cover rounded flex-shrink-0"
                  onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
                />
              )}
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-gray-900 truncate">
                  {result.id}
                </p>
                <div className="text-xs text-gray-500 truncate">
                  {Object.entries(result.metadata)
                    .filter(([, v]) => typeof v === 'string' && !String(v).startsWith('http'))
                    .slice(0, 2)
                    .map(([k, v]) => `${k}: ${v}`)
                    .join(' | ')}
                </div>
              </div>
              <span className="text-xs text-gray-400 flex-shrink-0">
                {result.distance.toFixed(3)}
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
```

---

## Task 8: Frontend - Integrate SearchBar and SearchResults into PlotPage

**Files:**
- Modify: `frontend/src/pages/PlotPage.tsx`

**Step 1: Add search components to the left sidebar**

Import and render `SearchBar` below `PlotControls` (inside the sidebar div), and `SearchResults` below it. Pass the selected collection name to SearchBar.

The PlotPage needs access to the selected collection name. We can get it from the plotStore or from the URL search params. Since `PlotControls` already uses `searchParams`, extract collection from there.

Add the SearchBar after `PlotControls` and SearchResults after that, inside the sidebar div. SearchBar should only show when plotData is available (meaning a collection has been computed).

---

## Task 9: Frontend - Highlight matching points in 3D renderers

**Files:**
- Modify: `frontend/src/components/plot/ParticleCloud.tsx`
- Modify: `frontend/src/components/plot/InstancedSpheres.tsx`
- Modify: `frontend/src/components/plot/ImageSpriteCloud.tsx`

**Step 1: ParticleCloud - highlight logic**

Read `highlightedIds` from the store. When `highlightedIds.size > 0`:
- Matched points: full opacity, normal size
- Non-matched points: reduced opacity (0.15), normal size

Use a custom ShaderMaterial or set alpha in the color buffer. The simplest approach: add an `alpha` buffer attribute and use `transparent={true}` with a custom opacity per point. Since `pointsMaterial` doesn't support per-point opacity natively, use the color channel trick: dim non-highlighted colors by multiplying RGB by 0.15.

```typescript
// In the useMemo, after setting colors:
const hasHighlights = highlightedIds.size > 0

// Modify color loop:
const dimFactor = hasHighlights && !highlightedIds.has(p.id) ? 0.15 : 1.0
cols[i * 3] = color.r * dimFactor
cols[i * 3 + 1] = color.g * dimFactor
cols[i * 3 + 2] = color.b * dimFactor
```

Add `highlightedIds` to the useMemo dependency array.

**Step 2: InstancedSpheres - highlight logic**

Same approach: dim non-highlighted instance colors.

```typescript
// In useEffect, modify the color setting:
const dimFactor = highlightedIds.size > 0 && !highlightedIds.has(p.id) ? 0.15 : 1.0
const c = colorObjects[p.cluster % colorObjects.length].clone().multiplyScalar(dimFactor)
mesh.setColorAt(i, c)
```

Add `highlightedIds` store subscription.

**Step 3: ImageSpriteCloud - highlight logic**

Add opacity prop based on highlight state:

```typescript
// In spritesToRender mapping:
const isHighlighted = highlightedIds.size === 0 || highlightedIds.has(point.id)
const opacity = isHighlighted ? 1.0 : 0.15
// Pass opacity to PointSprite and apply to spriteMaterial
```

---

## Task 10: Backend - Run full test suite and lint

**Step 1: Run all backend checks**

Run: `uv run ruff check embedding_cluster/ tests/`
Run: `uv run ruff format --check embedding_cluster/ tests/`
Run: `uv run mypy embedding_cluster/`
Run: `uv run pytest --cov=embedding_cluster --cov-report=term-missing --cov-fail-under=70`

Fix any failures.

---

## Task 11: Frontend - Build check

**Step 1: Run frontend build**

Run: `cd frontend && npm run build`

Fix any TypeScript or build errors.

---

## Task 12: Update README.md

**Files:**
- Modify: `README.md`

**Step 1: Add search feature to Features list**

Add bullet: `* Semantic search within clusters — find similar items by text query or image URL, with results highlighted in the 3D view.`

**Step 2: Update Web UI section**

Add to the Web UI bullet list:
`* **Search** -- Type a text query or paste an image URL to find the most similar items in a collection. Results are highlighted in the 3D scatter plot with distance scores.`
