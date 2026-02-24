# FastAPI Web UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a FastAPI + React SPA that wraps the existing embedding indexing and 3D scatter plot visualization, with Three.js rendering and real-time WebSocket progress.

**Architecture:** FastAPI backend wraps existing `indexer.py` and `scatter_plot.py` as REST/WS endpoints. React frontend with @react-three/fiber handles 3D visualization. The existing core modules are reused with minimal refactoring.

**Tech Stack:** Python 3.13 / FastAPI / uvicorn / React 19 / Vite / TypeScript / @react-three/fiber / @react-three/drei / Three.js / Zustand / TanStack Query / Tailwind CSS / shadcn/ui

**Design doc:** `docs/plans/2026-02-19-fastapi-web-ui-design.md`

---

## Task 1: Backend scaffolding - FastAPI app with health check

**Files:**
- Create: `embedding_cluster/server/__init__.py`
- Create: `embedding_cluster/server/app.py`
- Create: `embedding_cluster/server/models.py`
- Modify: `pyproject.toml` (add fastapi, uvicorn, python-multipart, websockets deps)
- Test: `tests/test_server_app.py`

**Step 1: Add backend dependencies to pyproject.toml**

Add to `[project.dependencies]`:
```
"fastapi>=0.115,<1",
"uvicorn>=0.34,<1",
"python-multipart>=0.0.20,<1",
"websockets>=14,<15",
```

**Step 2: Run `uv sync --all-extras` to install deps**

Run: `uv sync --all-extras`
Expected: Success, new packages installed.

**Step 3: Create server package with FastAPI app factory**

Create `embedding_cluster/server/__init__.py` (empty).

Create `embedding_cluster/server/app.py`:
```python
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    app = FastAPI(
        title="Embedding Clusters",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app
```

Create `embedding_cluster/server/models.py` (empty Pydantic models placeholder):
```python
from __future__ import annotations
```

**Step 4: Write test for health endpoint**

Create `tests/test_server_app.py`:
```python
from __future__ import annotations

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


async def test_health_check(client):
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

**Step 5: Run test**

Run: `uv run pytest tests/test_server_app.py -v`
Expected: PASS

**Step 6: Add httpx to dev deps (needed for async test client)**

Add `"httpx>=0.28,<1"` to `[project.optional-dependencies] dev`.
Run: `uv sync --all-extras`

**Step 7: Run linting and type check**

Run: `uv run ruff check embedding_cluster/server/ tests/test_server_app.py`
Run: `uv run mypy embedding_cluster/server/`

**Step 8: Commit**

```bash
git add embedding_cluster/server/ tests/test_server_app.py pyproject.toml
git commit -m "feat(server): scaffold FastAPI app with health endpoint"
```

---

## Task 2: Collections API - list, info, delete

**Files:**
- Create: `embedding_cluster/server/routes/__init__.py`
- Create: `embedding_cluster/server/routes/collections.py`
- Modify: `embedding_cluster/server/app.py` (register router)
- Modify: `embedding_cluster/server/models.py` (add response models)
- Test: `tests/test_server_collections.py`

**Step 1: Define response models in models.py**

```python
from __future__ import annotations

from pydantic import BaseModel


class CollectionInfo(BaseModel):
    name: str
    count: int


class CollectionDetail(BaseModel):
    name: str
    count: int
    metadata_fields: list[str]


class MessageResponse(BaseModel):
    message: str
```

**Step 2: Create collections router**

Create `embedding_cluster/server/routes/__init__.py` (empty).

Create `embedding_cluster/server/routes/collections.py`:
```python
from __future__ import annotations

import chromadb
from fastapi import APIRouter, HTTPException

from embedding_cluster.server.models import (
    CollectionDetail,
    CollectionInfo,
    MessageResponse,
)

router = APIRouter(prefix="/api/collections", tags=["collections"])


def _get_chromadb_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path="./chromadb")


@router.get("", response_model=list[CollectionInfo])
async def list_collections() -> list[CollectionInfo]:
    client = _get_chromadb_client()
    collections = client.list_collections()
    return [
        CollectionInfo(name=c, count=client.get_collection(c).count())
        for c in collections
    ]


@router.get("/{name}", response_model=CollectionDetail)
async def get_collection(name: str) -> CollectionDetail:
    client = _get_chromadb_client()
    try:
        collection = client.get_collection(name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {name}") from e
    count = collection.count()
    metadata_fields: list[str] = []
    if count > 0:
        sample = collection.peek(limit=1)
        if sample["metadatas"] and len(sample["metadatas"]) > 0:
            metadata_fields = sorted(sample["metadatas"][0].keys())
    return CollectionDetail(
        name=name, count=count, metadata_fields=metadata_fields
    )


@router.delete("/{name}", response_model=MessageResponse)
async def delete_collection(name: str) -> MessageResponse:
    client = _get_chromadb_client()
    try:
        client.delete_collection(name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {name}") from e
    return MessageResponse(message=f"Deleted collection: {name}")
```

**Step 3: Register router in app.py**

Add to `create_app()`:
```python
from embedding_cluster.server.routes.collections import router as collections_router
app.include_router(collections_router)
```

**Step 4: Write tests**

Create `tests/test_server_collections.py` with tests using mocked ChromaDB client.

**Step 5: Run tests, lint, typecheck**

Run: `uv run pytest tests/test_server_collections.py -v`
Run: `uv run ruff check embedding_cluster/server/`
Run: `uv run mypy embedding_cluster/server/`

**Step 6: Commit**

```bash
git commit -m "feat(server): add collections API - list, detail, delete"
```

---

## Task 3: CSV upload and preview API

**Files:**
- Create: `embedding_cluster/server/routes/csv.py`
- Modify: `embedding_cluster/server/app.py` (register router)
- Modify: `embedding_cluster/server/models.py` (add CSV models)
- Test: `tests/test_server_csv.py`

**Step 1: Add CSV models**

```python
class CsvUploadResponse(BaseModel):
    filename: str
    rows: int
    columns: list[str]


class CsvPreviewResponse(BaseModel):
    columns: list[str]
    rows: list[dict[str, str]]
    total_rows: int
```

**Step 2: Create CSV router**

Handles file upload (saves to `./uploads/` dir), preview (reads first N rows and returns columns + data).

Key behaviors:
- `POST /api/csv/upload` accepts multipart file, saves it, returns column names
- `POST /api/csv/preview` accepts `{"filename": "...", "limit": 10}`, returns first N rows

**Step 3: Write tests with sample CSV fixture**

**Step 4: Run tests, lint, typecheck, commit**

```bash
git commit -m "feat(server): add CSV upload and preview API"
```

---

## Task 4: Background task registry and WebSocket manager

**Files:**
- Create: `embedding_cluster/server/tasks.py`
- Create: `embedding_cluster/server/ws.py`
- Test: `tests/test_server_tasks.py`

**Step 1: Create task registry**

In-memory dict mapping `job_id -> TaskState`. TaskState holds status (pending/running/completed/failed/cancelled), progress data, and an `asyncio.Event` for cancellation.

```python
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskState:
    job_id: str
    status: TaskStatus = TaskStatus.PENDING
    progress: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskState] = {}

    def create(self) -> TaskState:
        job_id = str(uuid.uuid4())
        task = TaskState(job_id=job_id)
        self._tasks[job_id] = task
        return task

    def get(self, job_id: str) -> TaskState | None:
        return self._tasks.get(job_id)

    def cancel(self, job_id: str) -> bool:
        task = self._tasks.get(job_id)
        if task and task.status == TaskStatus.RUNNING:
            task.cancel_event.set()
            task.status = TaskStatus.CANCELLED
            return True
        return False


task_registry = TaskRegistry()
```

**Step 2: Create WebSocket manager**

Manages per-job WebSocket connections. When the indexer reports progress, the manager broadcasts to all connected clients for that job.

```python
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from fastapi import WebSocket


class WebSocketManager:
    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = defaultdict(list)

    async def connect(self, job_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[job_id].append(websocket)

    async def disconnect(self, job_id: str, websocket: WebSocket) -> None:
        self._connections[job_id].remove(websocket)

    async def broadcast(self, job_id: str, data: dict[str, Any]) -> None:
        for ws in self._connections.get(job_id, []):
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                pass


ws_manager = WebSocketManager()
```

**Step 3: Write tests**

**Step 4: Commit**

```bash
git commit -m "feat(server): add background task registry and WebSocket manager"
```

---

## Task 5: Indexing API - start, status, WebSocket, cancel

**Files:**
- Create: `embedding_cluster/server/routes/index.py`
- Modify: `embedding_cluster/server/app.py` (register router)
- Modify: `embedding_cluster/server/models.py` (add indexing models)
- Modify: `embedding_cluster/indexer.py` (refactor to accept progress callback)
- Test: `tests/test_server_index.py`

**Step 1: Add indexing request/response models**

```python
class IndexRequest(BaseModel):
    csv_filename: str
    id_field: str | None = None
    image_embedding_fields: list[str] | None = None
    text_embedding_fields: list[str] | None = None
    image_model_name: str = "openai/clip-vit-base-patch32"
    text_model_name: str = "BAAI/bge-small-en-v1.5"
    chromadb_collection_prefix: str = ""
    number_of_async_tasks: int = 1
    index_bulk_size: int = 100
    index_start_line: int | None = None
    index_end_line: int | None = None
    process_unit_device: str = "cpu"
    embedding_fields_prefix: str = "embedding_"


class IndexStartResponse(BaseModel):
    job_id: str
    status: str


class IndexStatusResponse(BaseModel):
    job_id: str
    status: str
    rows_indexed: int
    total_rows: int | None
    errors: int
```

**Step 2: Refactor `indexer.py` to accept a progress callback**

The current `main_indexer` prints progress via `logger.info`. Refactor to accept an optional `on_progress` callback:

```python
async def main_indexer(
    settings: Settings,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    cancel_event: asyncio.Event | None = None,
) -> None:
```

Inside the batch loop, call `on_progress({"rows_indexed": rows_read, ...})` after each batch. Check `cancel_event.is_set()` to support cancellation.

This is a non-breaking change - the CLI path still calls `main_indexer(settings)` with no callback.

**Step 3: Create index router**

- `POST /api/index/start` - creates task, spawns `asyncio.create_task` running the indexer with progress callback that broadcasts via WebSocket manager
- `GET /api/index/status/{job_id}` - returns current task state
- `WS /api/index/ws/{job_id}` - WebSocket connection for real-time progress
- `POST /api/index/cancel/{job_id}` - sets cancel event

**Step 4: Write tests (mocked indexer, test WS messages)**

**Step 5: Run all tests, lint, typecheck**

**Step 6: Commit**

```bash
git commit -m "feat(server): add indexing API with WebSocket progress"
```

---

## Task 6: Plot compute API

**Files:**
- Create: `embedding_cluster/server/routes/plot.py`
- Modify: `embedding_cluster/server/app.py` (register router)
- Modify: `embedding_cluster/server/models.py` (add plot models)
- Refactor: `embedding_cluster/scatter_plot.py` (extract data computation from Dash rendering)
- Test: `tests/test_server_plot.py`

**Step 1: Refactor scatter_plot.py**

Extract a pure-data function from `prepare_data()` that returns the computed plot data (positions, clusters, metadata) without creating a Dash app or Plotly figure. The existing `prepare_data` can then call this new function.

New function:
```python
def compute_plot_data(settings: Settings) -> dict[str, Any]:
    """Compute t-SNE + k-means and return raw data (no Plotly/Dash)."""
    # ... existing logic from prepare_data() ...
    # Returns: {"points": [...], "clusters": [...], "total_points": N}
```

**Step 2: Add plot models**

```python
class PlotRequest(BaseModel):
    chromadb_collection_name: str
    num_clusters: int = 10
    text_display_fields: list[str] | None = None
    image_field: str | None = None
    gpt_generate_cluster_name: bool = False
    gpt_default_model: str = "gpt-3.5-turbo"
    gpt_default_temperature: float = 0.51


class PlotPoint(BaseModel):
    x: float
    y: float
    z: float
    cluster: int
    metadata: dict[str, Any]
    id: str


class PlotCluster(BaseModel):
    index: int
    name: str
    color: str
    count: int


class PlotResponse(BaseModel):
    points: list[PlotPoint]
    clusters: list[PlotCluster]
    total_points: int
```

**Step 3: Create plot router**

- `POST /api/plot/compute` - runs compute in background task, returns job_id
- `GET /api/plot/data/{job_id}` - returns computed data when ready

**Step 4: Write tests, lint, typecheck, commit**

```bash
git commit -m "feat(server): add plot compute API with t-SNE/k-means"
```

---

## Task 7: Add `__main__.py` server mode

**Files:**
- Modify: `embedding_cluster/__main__.py`

**Step 1: Add SERVER mode dispatch**

```python
elif settings.running_mode == "SERVER":
    import uvicorn
    from embedding_cluster.server.app import create_app
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Step 2: Test manually**

Run: `RUNNING_MODE=SERVER uv run python -m embedding_cluster`
Expected: Server starts on port 8000, health endpoint responds.

**Step 3: Commit**

```bash
git commit -m "feat: add SERVER running mode to __main__.py"
```

---

## Task 8: Frontend scaffolding - Vite + React + TypeScript

**Files:**
- Create: `frontend/` directory with full Vite scaffold
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts` (with API proxy to FastAPI)
- Create: `frontend/tsconfig.json`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/types/index.ts`

**Step 1: Scaffold Vite React-TS project**

Run: `npm create vite@latest frontend -- --template react-ts`

**Step 2: Install dependencies**

```bash
cd frontend
npm install @react-three/fiber @react-three/drei three @tanstack/react-query zustand react-router-dom
npm install -D @types/three tailwindcss @tailwindcss/vite
```

**Step 3: Configure Vite proxy**

In `vite.config.ts`, proxy `/api` to `http://localhost:8000`.

**Step 4: Set up Tailwind CSS**

Add Tailwind via `@tailwindcss/vite` plugin and `@import "tailwindcss"` in CSS.

**Step 5: Create App shell with routing**

Three routes: `/` (Index), `/plot` (Plot), `/collections` (Collections).
Navigation bar with links.

**Step 6: Create shared TypeScript types**

`frontend/src/types/index.ts` matching backend Pydantic models.

**Step 7: Verify it runs**

Run: `npm run dev` (from frontend/)
Expected: Vite dev server at localhost:5173, shows navigation shell.

**Step 8: Commit**

```bash
git commit -m "feat(frontend): scaffold React + Vite + Tailwind app shell"
```

---

## Task 9: API client layer

**Files:**
- Create: `frontend/src/api/client.ts` (base fetch wrapper)
- Create: `frontend/src/api/collections.ts`
- Create: `frontend/src/api/csv.ts`
- Create: `frontend/src/api/index.ts`
- Create: `frontend/src/api/plot.ts`

**Step 1: Create typed API client functions**

Each file exports functions wrapping fetch calls with proper TypeScript types. TanStack Query hooks will use these.

**Step 2: Commit**

```bash
git commit -m "feat(frontend): add typed API client layer"
```

---

## Task 10: Collections page (frontend)

**Files:**
- Create: `frontend/src/pages/CollectionsPage.tsx`
- Create: `frontend/src/components/collections/CollectionList.tsx`

**Step 1: Build CollectionList component**

Table showing collection name, item count. Delete button with confirmation. "Visualize" button linking to `/plot?collection=<name>`.

Uses `useQuery` from TanStack Query to fetch collections.

**Step 2: Verify end-to-end**

Start backend (`RUNNING_MODE=SERVER`), start frontend (`npm run dev`).
Navigate to `/collections`. Should show any existing ChromaDB collections.

**Step 3: Commit**

```bash
git commit -m "feat(frontend): add Collections page with list/delete"
```

---

## Task 11: CSV upload and Index page (frontend)

**Files:**
- Create: `frontend/src/pages/IndexPage.tsx`
- Create: `frontend/src/components/csv/CsvUpload.tsx`
- Create: `frontend/src/components/csv/CsvPreview.tsx`
- Create: `frontend/src/components/index/IndexForm.tsx`
- Create: `frontend/src/components/index/IndexProgress.tsx`
- Create: `frontend/src/hooks/useIndexWebSocket.ts`

**Step 1: Build CsvUpload component**

Drag-and-drop zone. Calls `/api/csv/upload`. On success, shows CsvPreview.

**Step 2: Build CsvPreview component**

Renders first 10 rows as a table. Column headers used to populate form dropdowns.

**Step 3: Build IndexForm component**

Form with all indexing parameters. Dropdowns populated from CSV columns.
Fields: id_field, image_embedding_fields (multi-select), text_embedding_fields (multi-select), image_model_name, text_model_name, collection_prefix, async_tasks, bulk_size, start_line, end_line, device.

**Step 4: Build useIndexWebSocket hook**

Connects to `ws://localhost:8000/api/index/ws/{job_id}`. Parses messages. Returns `{progress, logs, status, isConnected}`.

**Step 5: Build IndexProgress component**

Shows: progress bar (rows_indexed / total_rows), elapsed time, error count, scrollable log panel. Cancel button.

**Step 6: Wire up IndexPage**

Upload -> Preview -> Form -> Start -> Progress. State machine flow.

**Step 7: End-to-end test with sample CSV**

**Step 8: Commit**

```bash
git commit -m "feat(frontend): add Index page with upload, form, and WebSocket progress"
```

---

## Task 12: Plot page - basic scatter with colored particles

**Files:**
- Create: `frontend/src/pages/PlotPage.tsx`
- Create: `frontend/src/components/plot/ScatterPlot.tsx`
- Create: `frontend/src/components/plot/ParticleCloud.tsx`
- Create: `frontend/src/components/plot/TooltipCard.tsx`
- Create: `frontend/src/components/plot/ClusterLegend.tsx`
- Create: `frontend/src/components/plot/PlotControls.tsx`
- Create: `frontend/src/stores/plotStore.ts`
- Create: `frontend/src/hooks/usePlotData.ts`

**Step 1: Create Zustand plot store**

Stores: plotData, visibleClusters, hoveredPoint, renderMode.

**Step 2: Build PlotControls**

Collection selector dropdown. Cluster count slider. Display fields multi-select. Image field dropdown. "Compute" button.

Fetches collection list and metadata fields from API.

**Step 3: Build ParticleCloud component**

R3F component using `<points>` with `<bufferGeometry>`. Position, color, and size attributes from plotData. Uses `Float32BufferAttribute`.

Cluster colors: generate distinct palette (e.g., D3 categorical colors).

Hover: use `onPointerMove` with raycaster, find nearest point within threshold.

**Step 4: Build TooltipCard**

drei `<Html>` overlay. Shows image (if available) and metadata key-value pairs. Styled card with shadow.

**Step 5: Build ClusterLegend**

List of clusters outside canvas. Color swatch + name + count. Click to toggle visibility.

**Step 6: Build ScatterPlot**

Wraps `<Canvas>` with `<OrbitControls>`, ambient + directional light, and the active point cloud component. Renders TooltipCard when point is hovered.

**Step 7: Build PlotPage**

Left sidebar: PlotControls. Main area: ScatterPlot. Bottom: ClusterLegend.

**Step 8: End-to-end test**

Index sample CSV, then visualize. Verify 3D scatter renders, hover works, clusters toggle.

**Step 9: Commit**

```bash
git commit -m "feat(frontend): add Plot page with 3D colored particle scatter"
```

---

## Task 13: Image sprite rendering mode

**Files:**
- Create: `frontend/src/components/plot/ImageSpriteCloud.tsx`

**Step 1: Build ImageSpriteCloud**

Load images as textures. Create texture atlas or individual sprite materials. Each point renders its image thumbnail. Uses `<sprite>` or custom `ShaderMaterial` on `<points>`.

Handle loading states (show placeholder until texture loads). Limit concurrent texture loads.

**Step 2: Add to render mode toggle in PlotControls**

**Step 3: Commit**

```bash
git commit -m "feat(frontend): add image sprite rendering mode"
```

---

## Task 14: Instanced spheres rendering mode

**Files:**
- Create: `frontend/src/components/plot/InstancedSpheres.tsx`

**Step 1: Build InstancedSpheres**

R3F `<instancedMesh>` with `<sphereGeometry>`. Set per-instance position, color, and scale via `instanceMatrix` and `instanceColor`.

Add directional light for depth perception.

Hover: use R3F raycasting on instanced mesh (native support).

**Step 2: Add to render mode toggle**

**Step 3: Commit**

```bash
git commit -m "feat(frontend): add instanced spheres rendering mode"
```

---

## Task 15: Polish and integration testing

**Files:**
- Various touch-ups across frontend and backend

**Step 1: Add loading states and error handling**

Skeleton loaders for data fetching. Error boundaries for 3D canvas. Toast notifications for API errors.

**Step 2: Add fullscreen toggle and camera reset**

**Step 3: Run full test suite**

Run: `uv run pytest --cov=embedding_cluster --cov-report=term-missing --cov-fail-under=70`
Run: `cd frontend && npm run build` (verify production build)

**Step 4: Run linting**

Run: `uv run ruff check embedding_cluster/ tests/`
Run: `uv run mypy embedding_cluster/`
Run: `cd frontend && npx tsc --noEmit`

**Step 5: Update README.md with new SERVER mode instructions**

**Step 6: Final commit**

```bash
git commit -m "feat: polish UI, error handling, and update docs"
```

---

## Task Order and Dependencies

```
Task 1 (FastAPI scaffold)
  └─> Task 2 (Collections API)
  └─> Task 3 (CSV API)
  └─> Task 4 (Task registry + WS manager)
        └─> Task 5 (Indexing API)
        └─> Task 6 (Plot API)
  └─> Task 7 (SERVER mode)

Task 8 (Frontend scaffold)
  └─> Task 9 (API client)
        └─> Task 10 (Collections page)
        └─> Task 11 (Index page)
        └─> Task 12 (Plot page - particles)
              └─> Task 13 (Image sprites)
              └─> Task 14 (Instanced spheres)

Task 15 (Polish) - depends on all above
```

Tasks 1-7 (backend) and Task 8-9 (frontend scaffold) can be parallelized.
Tasks 2, 3, 4 can be parallelized after Task 1.
Tasks 13, 14 can be parallelized after Task 12.
