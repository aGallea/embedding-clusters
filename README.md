# embedding-clusters

![python-version][python-version]

* [Description](#description)
  * [Features](#features)
  * [Architecture](#architecture)
  * [Tech Stack](#tech-stack)
* [Installation](#installation)
* [Usage](#usage)
  * [Index (CLI)](#index-cli)
  * [Plot (CLI)](#plot-cli)
  * [Web UI (Server Mode)](#web-ui-server-mode)
* [Development](#development)
* [Contributing](#contributing)

## Description

A full-stack tool for generating, indexing, and visualizing embedding
clusters from CSV data. Feed it a CSV with image URLs or text fields,
and it generates vector embeddings using CLIP (images) and
SentenceTransformer (text) models, stores them in ChromaDB, clusters
via k-means, and renders an interactive 3D scatter plot with
selectable dimensionality reduction (t-SNE, UMAP, or PCA).

The project supports three running modes: CLI-based batch indexing,
CLI-based Dash visualization, and a full web application with a
FastAPI backend and React frontend.

### Features

* Generate embeddings from images and text using CLIP and
  SentenceTransformer models.
* Configurable model selection for both image and text embeddings.
* Index embeddings into ChromaDB for persistent vector storage.
* Perform k-means clustering on embeddings with configurable cluster
  count.
* Automatic cluster count suggestion using elbow method (inertia
  curve) and silhouette score analysis across k=2..30, with an
  interactive chart to review the trade-off before applying.
* Visualize clustering results in an interactive 3D scatter plot
  with selectable dimensionality reduction: t-SNE, UMAP, or PCA,
  each with configurable algorithm-specific parameters.
* Web UI with FastAPI backend and React frontend for browser-based
  indexing and visualization.
* Three switchable 3D render modes: colored particles, image sprites,
  and instanced spheres.
* Real-time indexing progress via WebSocket streaming.
* Hover tooltips showing item metadata and images.
* Cluster visibility toggling and fullscreen mode.
* Optional GPT-powered automatic cluster naming via OpenAI API.
* Collection management (list, inspect, delete) through UI and API.
* Async batch indexing with configurable parallelism and retry logic.
* Hardware acceleration support (CPU, MPS, CUDA).
* Semantic search within clusters -- find similar items by text
  query or image URL, with results highlighted in the 3D view.

### Architecture

```text
                    CSV Data
                       |
                  [INDEX mode]
                  /          \
          CLIP Model    SentenceTransformer
         (images)           (text)
                  \          /
                   ChromaDB
                  (vector store)
                       |
                  [PLOT mode]
                       |
            StandardScaler + KMeans
                       |
          t-SNE / UMAP / PCA (3D projection)
                       |
              3D Scatter Plot
           (Dash CLI / React Web UI)
```

### Tech Stack

**Backend:** Python 3.13, FastAPI, ChromaDB, scikit-learn,
PyTorch, Transformers, aiohttp

**Frontend:** React 19, TypeScript, Three.js (react-three-fiber),
Zustand, TanStack Query, Tailwind CSS, Vite

**Quality:** mypy (strict), ruff, pytest, GitHub Actions CI,
pre-commit hooks, commitizen

## Installation

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

2. Clone and set up the project:

    ```bash
    git clone https://github.com/aGallea/embedding-clusters.git
    cd embedding-clusters
    uv sync --all-extras
    ```

> Models are downloaded from [HuggingFace](https://huggingface.co)
> on first run. Ensure network access to `huggingface.co`.
> UMAP support is optional. To install it:
> `uv sync --extra umap` or `uv pip install umap-learn`.
> t-SNE and PCA work without extra dependencies.

## Usage

### Index (CLI)

For our primary example, we'll utilize a CSV file sourced from
[Kaggle's Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset),
containing information about various fashion products.

To initiate the indexing process, certain parameters must be
provided, which are customizable according to your needs.

| Parameter | Description | Mandatory | Default |
|-----------|-------------|-----------|---------|
| `RUNNING_MODE` | `INDEX`, `PLOT`, or `SERVER` | Yes | `PLOT` |
| `LOCAL_CSV_FILENAME` | Path to the CSV file | Yes | |
| `ID_FIELD` | CSV field name to use as ChromaDB item id | No | Random |
| `IMAGE_MODEL_NAME` | CLIP model name | No | `openai/clip-vit-base-patch32` |
| `IMAGE_EMBEDDING_FIELDS` | Image field names (JSON array) | No | None |
| `TEXT_MODEL_NAME` | Text transformer model name | No | `BAAI/bge-small-en-v1.5` |
| `TEXT_EMBEDDING_FIELDS` | Text field names (JSON array) | No | None |
| `CHROMADB_COLLECTION_PREFIX` | Prefix for ChromaDB collection | No | |
| `NUMBER_OF_ASYNC_TASKS` | Parallel task count | No | `1` |
| `INDEX_BULK_SIZE` | Batch size when indexing embeddings | No | `100` |
| `INDEX_START_LINE` | First CSV line number to index | No | None |
| `INDEX_END_LINE` | Last CSV line number to index | No | None |
| `PROCESS_UNIT_DEVICE` | Compute device (`cpu`, `mps`, `cuda`) | No | `cpu` |

Index images from the example CSV:

```bash
RUNNING_MODE=INDEX \
  LOCAL_CSV_FILENAME=./embedding_cluster/csv/fashion_small.csv \
  ID_FIELD=id \
  IMAGE_EMBEDDING_FIELDS='["imageUrl"]' \
  CHROMADB_COLLECTION_PREFIX=fashion_ \
  NUMBER_OF_ASYNC_TASKS=10 \
  uv run python -m embedding_cluster
```

Index text fields instead (or in addition to images):

```bash
RUNNING_MODE=INDEX \
  LOCAL_CSV_FILENAME=./embedding_cluster/csv/fashion_small.csv \
  ID_FIELD=id \
  TEXT_EMBEDDING_FIELDS='["productDisplayName"]' \
  CHROMADB_COLLECTION_PREFIX=fashion_ \
  uv run python -m embedding_cluster
```

> Only the models needed for your chosen fields are downloaded.
> If you only use `TEXT_EMBEDDING_FIELDS`, the CLIP image model is
> not loaded.

### Plot (CLI)

After successfully indexing data, visualize it. The ChromaDB
collection name is the prefix combined with the embedded field name.

| Parameter | Description | Mandatory | Default |
|-----------|-------------|-----------|---------|
| `RUNNING_MODE` | `PLOT` | Yes | `PLOT` |
| `CHROMADB_COLLECTION_NAME` | ChromaDB collection to visualize | Yes | |
| `TEXT_DISPLAY_FIELDS` | Text fields to show on hover (JSON array) | No | None |
| `IMAGE_FIELD` | Image field to show on hover | No | None |
| `NUM_CLUSTERS` | Number of k-means clusters | No | `10` |
| `GPT_GENERATE_CLUSTER_NAME` | Use GPT to name clusters (needs `OPENAI_API_KEY`) | No | `False` |
| `GPT_DEFAULT_MODEL` | GPT model for cluster naming | No | `gpt-3.5-turbo` |
| `GPT_DEFAULT_TEMPERATURE` | GPT temperature for cluster naming | No | `0.51` |
| `REDUCTION_ALGORITHM` | Dimensionality reduction (`tsne`, `umap`, `pca`) | No | `tsne` |
| `TSNE_PERPLEXITY` | t-SNE perplexity (5--50) | No | `30.0` |
| `TSNE_LEARNING_RATE` | t-SNE learning rate (`auto` or numeric) | No | `auto` |
| `UMAP_N_NEIGHBORS` | UMAP neighbor count (2--100) | No | `15` |
| `UMAP_MIN_DIST` | UMAP minimum distance (0--1) | No | `0.1` |
| `UMAP_METRIC` | UMAP distance metric | No | `cosine` |

```bash
RUNNING_MODE=PLOT \
  CHROMADB_COLLECTION_NAME=fashion_imageUrl \
  TEXT_DISPLAY_FIELDS='["productDisplayName"]' \
  IMAGE_FIELD=imageUrl \
  uv run python -m embedding_cluster
```

### Web UI (Server Mode)

Instead of using environment variables and the CLI, you can run a
web server with a React-based UI that provides the same indexing and
visualization features in the browser.

1. Install frontend dependencies and build:

    ```bash
    cd frontend
    npm install
    npm run build
    cd ..
    ```

2. Start the server:

    ```bash
    RUNNING_MODE=SERVER uv run python -m embedding_cluster
    ```

3. Open <http://localhost:8000> in your browser.

The web UI provides:

* **Index page** -- Upload a CSV, preview it, configure all indexing
  parameters (model names, embedding fields, collection prefix, etc.),
  and monitor progress in real time via WebSocket.
* **Plot page** -- Select a ChromaDB collection, configure clusters
  and display fields, then visualize in an interactive 3D scatter plot
  with hover tooltips showing metadata and images. Switch between
  colored particles, image sprites, and instanced spheres render modes.
  Click "Suggest" next to the cluster count slider to automatically
  analyze optimal cluster count -- the tool runs k-means for k=2..30,
  computes inertia (elbow method) and silhouette scores, and displays
  the results in a chart with the recommended k highlighted. Accept
  the suggestion to update the slider, or keep your manual value.
  Use semantic search to find similar items by text or image URL,
  with matching points highlighted in the 3D view.
* **Collections page** -- List, inspect, and delete ChromaDB
  collections.

## Development

### Prerequisites

* Python 3.13+
* [uv](https://docs.astral.sh/uv/getting-started/installation/)
* Node.js 18+ (for frontend development)

### Setup

```bash
uv sync --all-extras
uv run pre-commit install --install-hooks -t pre-commit -t commit-msg
```

### Commands

```bash
# Lint
uv run ruff check embedding_cluster/ tests/

# Format
uv run ruff format embedding_cluster/ tests/

# Type check
uv run mypy embedding_cluster/

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=embedding_cluster --cov-report=term-missing --cov-fail-under=70

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

### E2E Testing

End-to-end tests use [Playwright](https://playwright.dev/) and run
against the full stack (FastAPI backend + React frontend).

#### First-Time Setup

1. Install Playwright browsers:

    ```bash
    cd frontend
    npm install
    npx playwright install chromium
    ```

2. Index sample data for tests (one-time, from project root):

    ```bash
    RUNNING_MODE=INDEX \
      LOCAL_CSV_FILENAME=./embedding_cluster/csv/fashion_small.csv \
      ID_FIELD=id \
      TEXT_EMBEDDING_FIELDS='["productDisplayName"]' \
      CHROMADB_COLLECTION_PREFIX=fashion_ \
      uv run python -m embedding_cluster
    ```

3. Build the frontend:

    ```bash
    cd frontend
    npm run build
    ```

#### Running E2E Tests

```bash
cd frontend

# Run all E2E tests (headless, auto-starts backend)
npm run test:e2e

# Run with interactive UI for debugging
npm run test:e2e:ui

# Run a specific test file
npx playwright test e2e/search.spec.ts

# Show HTML report after a run
npx playwright show-report
```

The Playwright config auto-starts the FastAPI server. If you
already have the server running (`RUNNING_MODE=SERVER`), it reuses
the existing server instead.

### Project Structure

```text
embedding_cluster/
  __main__.py          # Entry point, mode dispatch (INDEX/PLOT/SERVER)
  settings.py          # Pydantic Settings (env var config)
  utils.py             # Shared utilities (logging, ChromaDB, image downloader)
  indexer.py           # INDEX mode: CSV parsing, embedding generation, ChromaDB storage
  scatter_plot.py      # PLOT mode: clustering, t-SNE, Dash visualization
  server/
    app.py             # FastAPI app factory with SPA serving
    models.py          # Pydantic request/response models
    tasks.py           # Background task registry
    ws.py              # WebSocket connection manager
    routes/
      collections.py   # Collection CRUD API
      csv.py           # CSV upload and preview API
      index.py         # Indexing API with WebSocket progress
      plot.py          # Plot computation API
frontend/
  src/
    pages/             # IndexPage, PlotPage, CollectionsPage
    components/        # UI components (plot renderers, forms, etc.)
    stores/            # Zustand state (plotStore)
    hooks/             # Custom hooks (usePlotData, useIndexWebSocket)
    api/               # API client layer
    types/             # TypeScript interfaces
tests/
  test_settings.py     # Settings env var parsing tests
  test_utils.py        # Utilities and ImageDownloader tests
  test_indexer.py      # Indexer pipeline tests (mocked ML models)
  test_scatter_plot.py # Scatter plot tests (mocked data)
  test_main.py         # Entry point dispatch tests
```

## Contributing

Pull requests are welcome. For major changes, please open an issue
first to discuss what you would like to change.

On your first contribution, install the pre-commit hooks:

```bash
uv run pre-commit install --install-hooks -t pre-commit -t commit-msg
```

Commits follow
[Conventional Commits](https://www.conventionalcommits.org/)
format, enforced by commitizen via pre-commit hook.

<!-- MARKDOWN LINKS & IMAGES -->
[python-version]: https://img.shields.io/badge/python-3.13-blue.svg
