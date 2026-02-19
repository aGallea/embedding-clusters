# embedding-clusters

![python-version][python-version]

* [Description](#description)
  * [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
  * [Index](#index)
  * [Plot](#plot)
* [Contributing](#contributing)

## Description

This repository contains two Python programs aimed at analyzing and
visualizing collections of embeddings derived from images and/or text
using CLIP and transformer models. The first program focuses on
generating embeddings from input data, while the second program
processes these embeddings to perform clustering and visualization
tasks. It indexes these embeddings into ChromaDB and applies k-means
clustering to group them into a specified number of clusters. The
resulting clusters are then visualized in a 3D scatter plot using
t-SNE, enabling users to interactively explore the data, view
individual items, and obtain insights from the clustering results.

### Features

* Generate embeddings from images and text using CLIP and
  transformer models.
* Index embeddings into ChromaDB for efficient retrieval.
* Perform k-means clustering on embeddings to group them into
  clusters.
* Visualize clustering results in a 3D scatter plot using t-SNE.
* Interactive visualization allows users to hover over items and view
  associated images and names.

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

## Usage

### Index

For our primary example, we'll utilize a CSV file sourced from
[Kaggle's Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset),
containing information about various fashion products.

To initiate the indexing process, certain parameters must be
provided, which are customizable according to your needs.

| Parameter | Description | Mandatory | Default |
|-----------|-------------|-----------|---------|
| `RUNNING_MODE` | `INDEX` or `PLOT` | Yes | `PLOT` |
| `LOCAL_CSV_FILENAME` | Path to the CSV file | Yes | |
| `ID_FIELD` | CSV field name to use as ChromaDB item id | No | Random |
| `IMAGE_MODEL_NAME` | CLIP model name | No | `openai/clip-vit-base-patch32` |
| `IMAGE_EMBEDDING_FIELDS` | Image field names (JSON array) | No | None |
| `TEXT_MODEL_NAME` | Text transformer model name | No | `BAAI/bge-small-en-v1.5` |
| `TEXT_EMBEDDING_FIELDS` | Text field names (JSON array) | No | None |
| `CHROMADB_COLLECTION_PREFIX` | Prefix for ChromaDB collection | No | |
| `NUMBER_OF_ASYNC_TASKS` | Parallel task count | No | `1` |

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

### Plot

After successfully indexing data, visualize it. The ChromaDB
collection name is the prefix combined with the embedded field name.

| Parameter | Description | Mandatory | Default |
|-----------|-------------|-----------|---------|
| `RUNNING_MODE` | `INDEX` or `PLOT` | Yes | `PLOT` |
| `CHROMADB_COLLECTION_NAME` | ChromaDB collection to visualize | Yes | |
| `TEXT_DISPLAY_FIELDS` | Text fields to show on hover (JSON array) | No | None |
| `IMAGE_FIELD` | Image field to show on hover | No | None |
| `NUM_CLUSTERS` | Number of k-means clusters | No | `10` |
| `GPT_GENERATE_CLUSTER_NAME` | Use GPT to name clusters (needs `OPENAI_API_KEY`) | No | `False` |

```bash
RUNNING_MODE=PLOT \
  CHROMADB_COLLECTION_NAME=fashion_imageUrl \
  TEXT_DISPLAY_FIELDS='["productDisplayName"]' \
  IMAGE_FIELD=imageUrl \
  uv run python -m embedding_cluster
```

## Contributing

Pull requests are welcome. For major changes, please open an issue
first to discuss what you would like to change.

On your first contribution, install the pre-commit hooks:

```bash
uv run pre-commit install --install-hooks -t pre-commit -t commit-msg
```

<!-- MARKDOWN LINKS & IMAGES -->
[python-version]: https://img.shields.io/badge/python-3.13-blue.svg
