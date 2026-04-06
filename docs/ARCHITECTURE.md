# Architecture

This document describes the system design, component responsibilities, and
data flow of **embedding-clusters**.

## Overview

The application converts CSV data into interactive 3D embedding
visualizations. It has three running modes, all dispatched from a single
entry point (`python -m embedding_cluster`):

| Mode | Entry | Purpose |
|------|-------|---------|
| `SERVER` | `server/app.py` | FastAPI backend + React SPA |
| `INDEX` | `indexer.py` | CLI embedding pipeline |
| `PLOT` | `scatter_plot.py` | CLI cluster visualization |

```text
                          __main__.py
                         /     |     \
                        /      |      \
                  INDEX      SERVER     PLOT
                    |          |          |
               indexer.py   FastAPI   scatter_plot.py
                    |       /  |  \       |
                    |   routes |  SPA     |
                    |          |          |
                    +--- ChromaDB --------+
```

## Backend Components

### Configuration (`settings.py`)

All configuration is driven by environment variables, parsed by
`pydantic-settings` `BaseSettings`. Each setting has a `Field()` with a
default value and description. List fields accept JSON-encoded strings
(e.g. `'["field1","field2"]'`).

### Indexing Pipeline (`indexer.py`)

Responsible for the INDEX mode and also used by the server's indexing route.

1. Read CSV rows (with optional start/stop line range)
2. Load embedding models:
   - **SentenceTransformer** for text fields
   - **CLIP** (via HuggingFace Transformers) for image URL fields
3. Generate embeddings in batches with semaphore-controlled concurrency
4. Store embeddings + metadata in ChromaDB collections
5. Report progress via callback (used by WebSocket in server mode)
6. Support cancellation via `asyncio.Event`

Images are downloaded asynchronously with exponential backoff retry
(up to 6 attempts) using a singleton `ImageDownloader` backed by
`aiohttp.ClientSession`.

### Plot Computation (`scatter_plot.py`)

Responsible for the PLOT mode and used by the server's plot route.

1. Load embeddings from a ChromaDB collection
2. Standardize with `StandardScaler`
3. Reduce dimensions using t-SNE, UMAP, or PCA
4. Cluster with KMeans
5. Compute silhouette scores, centroids, and per-point distances
6. Return structured point and cluster data

Additional capabilities:
- **Optimal cluster suggestion** — evaluates k=2..30 with inertia and
  silhouette scores
- **Sub-clustering** — re-run KMeans within a single cluster or on a
  selected subset of points

### AI Naming (`ai_naming.py`)

Uses [LiteLLM](https://github.com/BerriAI/litellm) as a universal gateway
to call any LLM provider (OpenAI, Google, Anthropic, Ollama) with a single
interface. Generates short (max 5 words) descriptive names for clusters
based on sampled items.

### Annotations (`annotations.py`)

Persists cluster metadata (name, notes, tags) as JSON sidecar files in
`./annotations/`, one file per plot job. The `AnnotationManager` handles
read/write with automatic timestamping.

### Utilities (`utils.py`)

- **Logging** — colored console formatter
- **ChromaDB helpers** — collection creation, batch document initialization
- **ImageDownloader** — singleton async image fetcher with retry logic
- **ID generator** — random alphanumeric IDs for jobs and documents

## Server Architecture

The `SERVER` mode runs a FastAPI application that serves both the REST API
and the built React SPA.

### App Factory (`server/app.py`)

`create_app()` assembles the FastAPI app:
- Registers all API route modules under `/api`
- Adds CORS middleware for frontend dev server (`localhost:5173`)
- Serves the React SPA from `frontend/dist` (if built), with catch-all
  fallback to `index.html` for client-side routing

### Task Management (`server/tasks.py`)

Long-running operations (indexing, plot computation) run as background
async tasks tracked by an in-memory `TaskRegistry`:

- Each job gets a unique ID and a `TaskState` with status, progress dict,
  result, error, and a cancel event
- Status lifecycle: `PENDING` → `RUNNING` → `COMPLETED` | `FAILED` | `CANCELLED`
- Clients poll status via REST or subscribe via WebSocket

### WebSocket Manager (`server/ws.py`)

Manages per-job WebSocket connections for real-time progress streaming.
Broadcasts JSON messages (progress, log, heartbeat, completed, error) to
all connected clients for a given job ID.

### API Routes (`server/routes/`)

| Route module | Prefix | Responsibility |
|-------------|--------|---------------|
| `csv.py` | `/api/csv` | Upload and preview CSV files |
| `index.py` | `/api/index` | Start/cancel indexing jobs, WebSocket progress |
| `collections.py` | `/api/collections` | List, detail, delete ChromaDB collections |
| `plot.py` | `/api/plot` | Compute plots, cluster detail, sub-clustering, suggest k |
| `search.py` | `/api/search` | Semantic search (text or image query) |
| `ai.py` | `/api/ai` | LLM cluster naming, connection testing, Ollama proxy |
| `annotations.py` | `/api/annotations` | CRUD for cluster annotations |

### Request/Response Models (`server/models.py`)

All API contracts are defined as Pydantic models. The frontend TypeScript
types in `frontend/src/types/index.ts` mirror these models.

## Frontend Architecture

The frontend is a React 19 SPA built with Vite and Tailwind CSS 4.

### Routing (`App.tsx`)

Four pages mapped via React Router:

| Path | Page | Purpose |
|------|------|---------|
| `/` | `HomePage` | Collection browser, quick actions |
| `/index` | `IndexPage` | CSV upload, embedding config, progress |
| `/plot` | `PlotPage` | 3D visualization, search, annotations |
| `/settings` | `SettingsPage` | AI provider configuration |

### State Management

- **Zustand** (`stores/plotStore.ts`) — single store for all plot-related
  state: points, clusters, visibility, search results, drill-down path,
  annotations, render mode, algorithm parameters
- **TanStack React Query** — server state (collections, plot data polling)

### 3D Visualization

Uses [React Three Fiber](https://github.com/pmndrs/react-three-fiber)
(`@react-three/fiber`) with `drei` helpers. Three render modes:

1. **Particles** — GPU-accelerated point cloud (default, best performance)
2. **Sprites** — image thumbnails at each point (when image field available)
3. **Instanced Spheres** — 3D sphere meshes with lighting

### API Client Layer (`api/`)

Typed fetch wrappers organized by domain (`client.ts`, `indexing.ts`,
`plot.ts`, `ai.ts`, `collections.ts`, `csv.ts`). All requests go through
a shared `apiFetch<T>()` utility with error handling.

### Hooks

- `useIndexWebSocket` — real-time indexing progress with stuck detection
  (warning after 15s, error after 30s of silence)
- `usePlotData` — starts plot computation, polls for results every 2s

## Data Flow

### Indexing (Web UI)

```text
Browser                          Server                    Storage
  |                                |                         |
  |-- POST /csv/upload ---------->|                         |
  |<---- filename, columns -------|                         |
  |                                |                         |
  |-- POST /index/start -------->|                         |
  |<---- job_id ------------------|                         |
  |                                |-- load models           |
  |== WS /index/ws/{job_id} ====>|                         |
  |                                |-- read CSV              |
  |<--- progress messages --------|-- embed rows ---------->|
  |<--- log messages -------------|-- store in ChromaDB --->|
  |<--- completed message --------|                         |
```

### Plot Generation

```text
Browser                          Server                    Storage
  |                                |                         |
  |-- POST /plot/compute -------->|                         |
  |<---- job_id ------------------|                         |
  |                                |-- load embeddings <----|
  |-- GET /plot/data/{id} ------->|-- reduce dimensions     |
  |<---- ready: false ------------|-- KMeans clustering     |
  |-- GET /plot/data/{id} ------->|-- compute centroids     |
  |<---- ready: true, data -------|                         |
  |                                |                         |
  |-- render 3D scene             |                         |
```

### Semantic Search

```text
Browser                          Server                    Storage
  |                                |                         |
  |-- POST /search -------------->|                         |
  |                                |-- infer model type      |
  |                                |-- embed query           |
  |                                |-- ChromaDB.query() <----|
  |<---- results + distances -----|                         |
  |                                |                         |
  |-- highlight in 3D scene       |                         |
```

## Storage

| Directory | Contents | Persistence |
|-----------|----------|-------------|
| `./chromadb/` | Vector database (embeddings + metadata) | Persistent, gitignored |
| `./uploads/` | User-uploaded CSV files | Persistent, gitignored |
| `./annotations/` | Cluster annotation JSON files | Persistent, gitignored |

## Design Decisions

### Why ChromaDB?

ChromaDB provides embedded vector storage with no external dependencies.
Collections persist to disk automatically, support metadata filtering, and
offer nearest-neighbor search out of the box — exactly what this tool needs
without requiring a separate database server.

### Why LiteLLM?

Rather than coupling to a single LLM provider, LiteLLM provides a unified
interface to OpenAI, Google, Anthropic, and Ollama. Users can switch
providers from the settings page without code changes.

### Why React Three Fiber?

The 3D visualization needs to render thousands of points interactively.
React Three Fiber provides a React-native API over Three.js, enabling
declarative scene composition while retaining GPU-level performance through
instanced rendering and point clouds.

### Job-Based Architecture

Embedding generation and plot computation can take seconds to minutes. The
task registry pattern decouples request handling from execution, allowing
the frontend to poll or subscribe via WebSocket without blocking HTTP
connections.
