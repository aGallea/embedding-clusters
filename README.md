# embedding-clusters

![python-version][python-version]

Turn raw CSV data into beautiful, interactive embedding clusters with fast
semantic search and a web UI.

![3D cluster plot](docs/screenshots/3d-cluster-plot.png)

## Quick Start (Web UI)

```bash
git clone https://github.com/aGallea/embedding-clusters.git
cd embedding-clusters
uv sync --all-extras
RUNNING_MODE=SERVER uv run python -m embedding_cluster
```

Open <http://localhost:8000>.

## Features

- **Embeddings & Storage**: CLIP (images) + SentenceTransformer (text) with
  ChromaDB persistence.
- **Clustering & Plot**: k-means clusters with 3D t-SNE, UMAP, or PCA.
- **Search & Collections**: semantic search by text or image URL, collection
  browsing and deletion.
- **Web UI**: CSV upload, live progress, plot controls, and multiple render
  modes.

## Visual Highlights

![Semantic search demo](docs/gifs/semantic-search-mini.gif)

![Home dashboard](docs/screenshots/home-page.png)
![Index page config](docs/screenshots/index-page-config.png)
![Index page progress](docs/screenshots/index-page-progress.png)
![Semantic search results](docs/screenshots/semantic-search.png)

## How Things Work

The tool turns a CSV file into an interactive 3D cluster
visualization in a few steps:

1. **Upload CSV** -- Provide a CSV file containing your data.
   The web UI lets you drag-and-drop; the CLI accepts a file path.
2. **Select fields** -- Choose which columns to embed. Text fields
   (e.g. product names) use a SentenceTransformer model; image URL
   fields use a CLIP model. You can embed both in the same dataset.
3. **Model download** -- The selected model is pulled from
   [HuggingFace](https://huggingface.co) on first use and cached
   locally for subsequent runs.
4. **Embedding & storage** -- Each row is converted into a vector
   embedding by the chosen model. Embeddings are stored in a
   [ChromaDB](https://www.trychroma.com/) collection for
   persistent, queryable vector storage.
5. **Plot configuration** -- Pick a collection, set the number of
   k-means clusters (or let the tool suggest one), and choose a
   dimensionality reduction algorithm (t-SNE, UMAP, or PCA).
6. **3D visualization** -- The reduced vectors are rendered as an
   interactive 3D scatter plot. Hover for metadata, toggle cluster
   visibility, switch render modes, or go fullscreen.
7. **Semantic search** -- Enter a text query or paste an image URL
   to find the most similar items. Matching points are highlighted
   directly in the 3D view.
8. **Cluster groupings** -- Toggle individual clusters on/off to
   focus on specific groups. Use the optional GPT-powered naming
   to label each cluster automatically.

## Cluster Drill-Down and Annotation

After generating a plot you can inspect, subdivide, and annotate
individual clusters directly from the web UI.

### Cluster Detail Panel

Click a cluster name in the legend to open a side panel listing every
item in that cluster. Items are sorted by distance to the centroid so
the most representative points appear first. The panel supports
pagination, displays item metadata, and shows image thumbnails when
an image field is available.

### Sub-Clustering

Inside the detail panel, toggle **Sub-cluster** to re-run k-means
within a single cluster. The result is rendered as a mini 3D scatter
plot (PCA-reduced) so you can explore hierarchical structure without
leaving the page.

### Annotations

Each cluster can be renamed, tagged, and annotated with free-form
notes. Changes are saved automatically (debounced) and persisted as
JSON sidecar files in the `annotations/` directory. Annotations
survive page reloads and are scoped per plot job.

### API Endpoints

The feature exposes the following REST endpoints under `/api`:

- `GET /plot/{job_id}/cluster/{index}` -- paginated cluster detail
- `POST /plot/{job_id}/cluster/{index}/sub-cluster` -- sub-cluster
  a single cluster with configurable k
- `GET /annotations/{job_id}` -- fetch all annotations for a job
- `PUT /annotations/{job_id}` -- update annotations
- `DELETE /annotations/{job_id}` -- delete annotations

```text
CSV --> Select Fields --> Download Model --> Embed & Store
  --> Configure Plot --> 3D Visualization --> Search & Explore
```

## Using CLI

### Index (CLI)

```bash
RUNNING_MODE=INDEX \
  LOCAL_CSV_FILENAME=./embedding_cluster/csv/fashion_small.csv \
  ID_FIELD=id \
  IMAGE_EMBEDDING_FIELDS='["imageUrl"]' \
  CHROMADB_COLLECTION_PREFIX=fashion_ \
  NUMBER_OF_ASYNC_TASKS=10 \
  uv run python -m embedding_cluster
```

### Plot (CLI)

```bash
RUNNING_MODE=PLOT \
  CHROMADB_COLLECTION_NAME=fashion_imageUrl \
  TEXT_DISPLAY_FIELDS='["productDisplayName"]' \
  IMAGE_FIELD=imageUrl \
  uv run python -m embedding_cluster
```

Key environment variables:

- `RUNNING_MODE`: `INDEX`, `PLOT`, or `SERVER`
- `TEXT_MODEL_NAME`: SentenceTransformer model name
- `IMAGE_MODEL_NAME`: CLIP model name
- `NUM_CLUSTERS`: k-means cluster count
- `REDUCTION_ALGORITHM`: `tsne`, `umap`, or `pca`

## Development

```bash
uv sync --all-extras
uv run ruff check embedding_cluster/ tests/
uv run ruff format embedding_cluster/ tests/
uv run mypy embedding_cluster/
uv run pytest
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

<!-- MARKDOWN LINKS & IMAGES -->
[python-version]: https://img.shields.io/badge/python-3.13-blue.svg
