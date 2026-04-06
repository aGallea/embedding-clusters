# Usage

This guide covers all three running modes and their configuration.

## Web UI (SERVER mode)

The recommended way to use embedding-clusters:

```bash
RUNNING_MODE=SERVER uv run python -m embedding_cluster
```

Open <http://localhost:8000>. The web UI provides:

1. **Home** — browse existing collections, see item counts and model info
2. **Index** — upload a CSV, select fields to embed, configure models,
   and watch real-time progress via WebSocket
3. **Plot** — pick a collection, set clustering parameters, and interact
   with a 3D scatter plot
4. **Settings** — configure AI provider for cluster naming

### Workflow

1. Navigate to the **Index** page
2. Upload a CSV file (drag-and-drop or file picker)
3. Select which columns to embed:
   - **Text fields** use a SentenceTransformer model
   - **Image URL fields** use a CLIP model
4. Click **Start** and watch the progress bar
5. Navigate to the **Plot** page
6. Select the new collection and configure:
   - Number of clusters (or click **Suggest** for auto-detection)
   - Reduction algorithm: t-SNE, UMAP, or PCA
   - Algorithm-specific parameters (perplexity, learning rate, etc.)
7. Click **Compute** to generate the 3D visualization
8. Explore:
   - **Hover** points to see metadata
   - **Search** by text or image URL to highlight similar items
   - **Click** a cluster in the legend to drill down
   - **Sub-cluster** within a cluster for hierarchical exploration
   - **Annotate** clusters with names, tags, and notes
   - **AI Name** clusters using your configured LLM provider

### Render Modes

The 3D plot supports three render modes, switchable from the plot controls:

- **Particles** — GPU-accelerated point cloud (default, best for large
  datasets)
- **Sprites** — image thumbnails at each point (requires an image field)
- **Instanced Spheres** — 3D sphere meshes with lighting effects

## CLI: INDEX mode

Embed CSV data into ChromaDB collections from the command line:

```bash
RUNNING_MODE=INDEX \
  LOCAL_CSV_FILENAME=./embedding_cluster/csv/fashion_small.csv \
  ID_FIELD=id \
  IMAGE_EMBEDDING_FIELDS='["imageUrl"]' \
  CHROMADB_COLLECTION_PREFIX=fashion_ \
  NUMBER_OF_ASYNC_TASKS=10 \
  uv run python -m embedding_cluster
```

You can also embed text fields:

```bash
RUNNING_MODE=INDEX \
  LOCAL_CSV_FILENAME=./data/products.csv \
  ID_FIELD=product_id \
  TEXT_EMBEDDING_FIELDS='["name", "description"]' \
  CHROMADB_COLLECTION_PREFIX=products_ \
  uv run python -m embedding_cluster
```

Or both text and image fields in the same run:

```bash
RUNNING_MODE=INDEX \
  LOCAL_CSV_FILENAME=./data/catalog.csv \
  ID_FIELD=id \
  TEXT_EMBEDDING_FIELDS='["title"]' \
  IMAGE_EMBEDDING_FIELDS='["thumbnail_url"]' \
  CHROMADB_COLLECTION_PREFIX=catalog_ \
  uv run python -m embedding_cluster
```

## CLI: PLOT mode

Generate a cluster visualization from an existing collection:

```bash
RUNNING_MODE=PLOT \
  CHROMADB_COLLECTION_NAME=fashion_imageUrl \
  TEXT_DISPLAY_FIELDS='["productDisplayName"]' \
  IMAGE_FIELD=imageUrl \
  NUM_CLUSTERS=8 \
  REDUCTION_ALGORITHM=umap \
  uv run python -m embedding_cluster
```

## Environment Variables

All configuration is via environment variables, parsed by
[pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).

### General

| Variable | Default | Description |
|----------|---------|-------------|
| `RUNNING_MODE` | — | `INDEX`, `PLOT`, or `SERVER` |
| `DEVICE` | `cpu` | PyTorch device (`cpu`, `mps`, `cuda`) |

### Indexing

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_CSV_FILENAME` | — | Path to CSV file |
| `ID_FIELD` | — | Column name for unique row IDs |
| `TEXT_EMBEDDING_FIELDS` | `[]` | JSON array of text columns to embed |
| `IMAGE_EMBEDDING_FIELDS` | `[]` | JSON array of image URL columns to embed |
| `TEXT_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | SentenceTransformer model |
| `IMAGE_MODEL_NAME` | `openai/clip-vit-base-patch32` | CLIP model |
| `CHROMADB_COLLECTION_PREFIX` | — | Prefix for collection names |
| `NUMBER_OF_ASYNC_TASKS` | `5` | Concurrency limit for async operations |
| `BULK_SIZE` | `100` | Batch size for ChromaDB upserts |
| `START_LINE` | — | First CSV line to process (optional) |
| `STOP_LINE` | — | Last CSV line to process (optional) |

### Plotting

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMADB_COLLECTION_NAME` | — | Collection to visualize |
| `TEXT_DISPLAY_FIELDS` | `[]` | JSON array of metadata fields to show on hover |
| `IMAGE_FIELD` | — | Metadata field containing image URLs |
| `NUM_CLUSTERS` | `10` | Number of k-means clusters |
| `REDUCTION_ALGORITHM` | `tsne` | `tsne`, `umap`, or `pca` |
| `TSNE_PERPLEXITY` | `30` | t-SNE perplexity parameter |
| `TSNE_LEARNING_RATE` | `200` | t-SNE learning rate |
| `UMAP_N_NEIGHBORS` | `15` | UMAP neighbors parameter |
| `UMAP_MIN_DIST` | `0.1` | UMAP minimum distance |
| `UMAP_METRIC` | `cosine` | UMAP distance metric |

## API Endpoints

When running in SERVER mode, the following REST endpoints are available
under `/api`:

### CSV

- `POST /api/csv/upload` — upload a CSV file
- `POST /api/csv/preview` — preview columns and sample rows

### Indexing

- `POST /api/index/start` — start an indexing job
- `GET /api/index/status/{job_id}` — poll job progress
- `POST /api/index/cancel/{job_id}` — cancel a running job
- `WS /api/index/ws/{job_id}` — real-time progress via WebSocket

### Collections

- `GET /api/collections` — list all collections
- `GET /api/collections/{name}` — collection detail with metadata fields
- `DELETE /api/collections/{name}` — delete a collection

### Plot

- `POST /api/plot/compute` — start plot computation
- `GET /api/plot/data/{job_id}` — fetch computed plot data
- `GET /api/plot/{job_id}/cluster/{index}` — paginated cluster items
- `POST /api/plot/{job_id}/cluster/{index}/sub-cluster` — sub-cluster
- `POST /api/plot/{job_id}/sub-cluster` — sub-cluster by point IDs
- `POST /api/plot/suggest-clusters` — auto-suggest optimal cluster count
- `POST /api/plot/{job_id}/suggest-k` — suggest k for sub-clustering

### Search

- `POST /api/search` — semantic search by text or image URL

### AI Naming

- `POST /api/ai/name-clusters` — generate cluster names via LLM
- `POST /api/ai/name-sub-clusters` — generate sub-cluster names
- `POST /api/ai/test-connection` — validate LLM credentials
- `POST /api/ai/ollama/models` — list available Ollama models

### Annotations

- `GET /api/annotations/{job_id}` — fetch annotations for a job
- `PUT /api/annotations/{job_id}/cluster/{index}` — update annotation
- `DELETE /api/annotations/{job_id}` — delete all annotations for a job

### Health

- `GET /api/health` — returns `{"status": "ok"}`
