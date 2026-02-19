# FastAPI Web UI Design

## Overview

Replace the CLI + Dash interface with a FastAPI backend serving a React
SPA. Users get the same indexing and visualization capabilities through
a browser UI with real-time progress and an interactive Three.js-based
3D scatter plot.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backend framework | FastAPI | Async-native, WebSocket support, Pydantic integration |
| Frontend framework | React 19 + Vite | Largest ecosystem for 3D viz libraries |
| 3D rendering | @react-three/fiber + @react-three/drei | Best performance (100k+ points), full visual control |
| Point rendering | 3 switchable modes | User toggles between colored particles, image sprites, instanced spheres |
| Real-time progress | WebSocket | Native FastAPI support, low latency |
| State management | Zustand | Lightweight, minimal boilerplate |
| UI components | Tailwind CSS + shadcn/ui | Modern, accessible, consistent |
| HTTP client | TanStack Query | Caching, retry, loading states |

## Architecture

```
React SPA (Vite)
  Index Page ──WebSocket──┐
  Plot Page ──REST────────┤
  Collections Page ──REST─┤
                          │
FastAPI Backend            │
  /api/index/*  ◄─────────┤
  /api/plot/*   ◄─────────┤
  /api/csv/*    ◄─────────┤
  /api/collections/* ◄────┘
       │
  Existing Core Modules
  (indexer.py, scatter_plot.py, utils.py)
       │
  ChromaDB (./chromadb/)
```

The existing `indexer.py`, `scatter_plot.py`, and `utils.py` are reused
with minor refactoring to decouple from Dash. FastAPI wraps them as API
endpoints. React handles the UI.

## API Endpoints

### CSV Management

| Method | Path | Purpose |
|--------|------|---------|
| POST | /api/csv/upload | Upload CSV file |
| POST | /api/csv/preview | Preview first N rows, return column names |

### Indexing

| Method | Path | Purpose |
|--------|------|---------|
| POST | /api/index/start | Start indexing job (JSON body with all settings) |
| GET | /api/index/status/{job_id} | Get indexing job status |
| WS | /api/index/ws/{job_id} | WebSocket for real-time progress |
| POST | /api/index/cancel/{job_id} | Cancel a running indexing job |

### Collections

| Method | Path | Purpose |
|--------|------|---------|
| GET | /api/collections | List all ChromaDB collections with counts |
| GET | /api/collections/{name} | Collection info (count, metadata field names) |
| DELETE | /api/collections/{name} | Delete a collection |

### Plot

| Method | Path | Purpose |
|--------|------|---------|
| POST | /api/plot/compute | Compute t-SNE + k-means, return job_id |
| GET | /api/plot/data/{job_id} | Get computed plot data |

## Indexing Request Body

Maps directly to current Settings fields:

```json
{
  "csv_filename": "uploaded/fashion.csv",
  "id_field": "id",
  "image_embedding_fields": ["imageUrl"],
  "text_embedding_fields": null,
  "image_model_name": "openai/clip-vit-base-patch32",
  "text_model_name": "BAAI/bge-small-en-v1.5",
  "chromadb_collection_prefix": "fashion_",
  "number_of_async_tasks": 10,
  "index_bulk_size": 100,
  "index_start_line": null,
  "index_end_line": null,
  "process_unit_device": "cpu",
  "embedding_fields_prefix": "embedding_"
}
```

## Plot Request Body

```json
{
  "chromadb_collection_name": "fashion_imageUrl",
  "num_clusters": 10,
  "text_display_fields": ["productDisplayName"],
  "image_field": "imageUrl",
  "gpt_generate_cluster_name": false,
  "gpt_default_model": "gpt-3.5-turbo",
  "gpt_default_temperature": 0.51
}
```

## Plot Response Data

```json
{
  "points": [
    {
      "x": 1.23, "y": -0.45, "z": 2.67,
      "cluster": 0,
      "metadata": {"productDisplayName": "Blue Shirt", "imageUrl": "https://..."},
      "id": "ABC123"
    }
  ],
  "clusters": [
    {"index": 0, "name": "Group 1", "color": "#e41a1c", "count": 42}
  ],
  "total_points": 500
}
```

## WebSocket Progress Messages

```json
{"type": "progress", "rows_indexed": 150, "total_rows": 1000, "errors": 2, "elapsed_seconds": 12.3}
{"type": "log", "level": "info", "message": "Loading image model: openai/clip-vit-base-patch32"}
{"type": "completed", "collection_names": ["fashion_imageUrl"], "total_indexed": 998}
{"type": "error", "message": "Failed to load CSV"}
```

## Frontend Pages

### Index Page

- CSV upload (drag-and-drop or file picker)
- CSV preview table (first 10 rows after upload)
- Configuration form with all indexing parameters:
  - ID field (dropdown populated from CSV columns)
  - Image embedding fields (multi-select from CSV columns)
  - Text embedding fields (multi-select from CSV columns)
  - Image model name (text input with default)
  - Text model name (text input with default)
  - Collection prefix (text input)
  - Async tasks count (number, default 1)
  - Bulk size (number, default 100)
  - Start/end line (optional numbers)
  - Processing device (dropdown: cpu/mps/cuda)
- Start button
- Real-time progress panel: rows indexed, elapsed time, error count, log stream

### Plot Page

- Collection selector (dropdown of available collections)
- Plot configuration:
  - Number of clusters (slider, default 10)
  - Display fields (multi-select from collection metadata keys)
  - Image field (dropdown from collection metadata keys)
  - GPT cluster naming toggle with model/temperature settings
- 3D scatter plot (React Three Fiber):
  - Render mode toggle: Colored Particles / Image Sprites / 3D Spheres
  - OrbitControls for rotation, zoom, pan
  - Cluster legend (click to toggle visibility)
  - Hover tooltip: floating card with image + metadata
  - Point count indicator
- Controls: reset camera, fullscreen

### Collections Page

- Table of all ChromaDB collections with item count
- Delete button per collection
- "Visualize" button to jump to Plot page with collection pre-selected

## 3D Visualization Detail

Three rendering modes, user-switchable:

### Colored Particles (default)

Uses Three.js `Points` with `BufferGeometry`. Each point is a
GPU-rendered circle. Colors assigned by cluster. Handles millions of
points.

### Image Sprites

Uses Three.js `Points` with `ShaderMaterial`. Each point renders a
small thumbnail of the actual image loaded as a texture atlas. Works
well up to ~10-20k unique textures.

### Instanced Spheres

Uses Three.js `InstancedMesh` with sphere geometry. Proper lighting
and depth. Good up to ~100k points.

### Hover Detection

R3F `onPointerOver`/`onPointerOut` events with GPU raycasting. For
`Points` geometry, use spatial indexing (octree from drei) for
efficient nearest-point lookup. On hover, show a `<Html>` overlay
from drei containing a React tooltip card with image and metadata.

### Cluster Legend

React component outside the canvas. Each cluster entry shows color
swatch, name, and point count. Click to toggle visibility
(filters points from the scene).

## Project Structure

```
embedding_cluster/
  (existing modules unchanged)
  server/
    __init__.py
    app.py              # FastAPI app factory
    routes/
      __init__.py
      index.py          # /api/index/* endpoints
      plot.py           # /api/plot/* endpoints
      csv.py            # /api/csv/* endpoints
      collections.py    # /api/collections/* endpoints
    models.py           # Pydantic request/response models
    tasks.py            # Background task registry
    ws.py               # WebSocket manager

frontend/
  package.json
  vite.config.ts
  tsconfig.json
  src/
    main.tsx
    App.tsx
    api/                # API client functions
      index.ts
      plot.ts
      collections.ts
      csv.ts
    pages/
      IndexPage.tsx
      PlotPage.tsx
      CollectionsPage.tsx
    components/
      csv/
        CsvUpload.tsx
        CsvPreview.tsx
      index/
        IndexForm.tsx
        IndexProgress.tsx
      plot/
        ScatterPlot.tsx
        ParticleCloud.tsx
        ImageSpriteCloud.tsx
        InstancedSpheres.tsx
        TooltipCard.tsx
        ClusterLegend.tsx
        PlotControls.tsx
      collections/
        CollectionList.tsx
      ui/                # shadcn/ui components
    hooks/
      useIndexWebSocket.ts
      usePlotData.ts
    stores/
      plotStore.ts       # Zustand store for plot state
    types/
      index.ts           # Shared TypeScript types
```

## Dependencies

### Backend (new)

- fastapi
- uvicorn
- python-multipart (file uploads)
- websockets

### Frontend

- react, react-dom
- @react-three/fiber, @react-three/drei, three
- @tanstack/react-query
- zustand
- tailwindcss
- shadcn/ui (radix-based components)
- vite

## Non-Goals (for initial version)

- Authentication / multi-user support
- Persistent job history (jobs live in memory, lost on restart)
- UMAP as alternative to t-SNE
- Embedding model fine-tuning from UI
- Docker deployment
