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

This repository contains two Python programs aimed at analyzing and visualizing collections of embeddings derived from images and/or text using
CLIP and transformer models. The first program focuses on generating embeddings from input data, while the second program processes these
embeddings to perform clustering and visualization tasks. It indexes these embeddings into ChromaDB and applies k-means clustering to group them
into a specified number of clusters. The resulting clusters are then visualized in a 3D scatter plot using t-SNE, enabling users to
interactively explore the data, view individual items, and obtain insights from the clustering results.

### Features

* Generate embeddings from images and text using CLIP and transformer models.
* Index embeddings into ChromaDB for efficient retrieval.
* Perform k-means clustering on embeddings to group them into clusters.
* Visualize clustering results in a 3D scatter plot using t-SNE.
* Interactive visualization allows users to hover over items and view associated images and names.

## Installation

1. Clone the repository

    ```bash
    git clone https://github.com/aGallea/embedding-clusters.git
    ```

2. Setup Python Environment

    * Enter project directory: `cd embedding-clusters`
    * Create virtual env named venv: `python3 -m venv venv`
    * Activate virtual env: `. venv/bin/activate`
    * Install dependencies: `pip install -r requirements.txt`

## Usage

### Index

For our primary example, we'll utilize a CSV file sourced from
[Kaggle's Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset),
containing information about various fashion products.
To initiate the indexing process, certain parameters must be provided, which are customizable according to your needs.
| Parameter  | Description | Is Mandatory | Default |
|------------|-------------|--------------|---------|
| `RUNNING_MODE` | `INDEX/PLOT` | TRUE | PLOT |
| `LOCAL_CSV_FILENAME` | Path to the CSV file | True | |
| `ID_FIELD`   | Name of the id field to use (from the CSV) for the ChromaDB item id | False | Random |
| `IMAGE_MODEL_NAME`  | Name of the CLIP model | False | openai/clip-vit-base-patch32 |
| `IMAGE_EMBEDDING_FIELDS` | Name of the image fields to use (from the CSV) as stringify array | False | None |
| `TEXT_MODEL_NAME` | Name of the Text Transformer model | False | BAAI/bge-small-en-v1.5 |
| `TEXT_EMBEDDING_FIELDS` | Name of the text fields to use (from the CSV) as stringify array | False | None |
| `CHROMADB_COLLECTION_PREFIX` | Prefix for the ChromaDB collection that will be used/created | False | |
| `NUMBER_OF_ASYNC_TASKS` | Boost your indexing | False | 1 |
Use the next command to index our example:

```bash
RUNNING_MODE=INDEX LOCAL_CSV_FILENAME=./embedding_cluster/csv/fashion_small.csv ID_FIELD=id IMAGE_EMBEDDING_FIELDS=[\"imageUrl\"]
CHROMADB_COLLECTION_PREFIX=fashion_ NUMBER_OF_ASYNC_TASKS=10 python -m embedding_cluster
```

### Plot

After successfully indexing our data, we can proceed to visualize it. It's important to note that the ChromaDB collection name is a
combination of the prefix and the field that was embedded.
To execute the plotting process, various parameters need to be specified, allowing for customization and flexibility.
| Parameter  | Description | Is Mandatory | Default |
|------------|-------------|--------------|---------|
| `RUNNING_MODE` | `INDEX/PLOT` | TRUE | PLOT |
| `CHROMADB_COLLECTION_NAME` | The name the ChromaDB collection + field that will be used | True | |
| `TEXT_DISPLAY_FIELDS` | The name of the text property to display when hovering on items (as same as in the CSV) | False | None |
| `IMAGE_FIELD` | The name of the image property to display when hovering on items (as same as in the CSV) | False | None |
| `NUM_CLUSTERS` | Number of clusters to create | True | 10 |
| `GPT_GENERATE_CLUSTER_NAME` | Request GPT to generate cool names to our new created clusters (api_key is required) | FALSE | False |
Use the next command to plot our example:

```bash
RUNNING_MODE=PLOT CHROMADB_COLLECTION_NAME=fashion_imageUrl TEXT_DISPLAY_FIELDS=[\"productDisplayName\"] IMAGE_FIELD=imageUrl
python -m embedding_cluster
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

On your first contribution, please install `pre-commit`:

```bash
pre-commit install --install-hooks -t pre-commit -t commit-msg
```

<!-- MARKDOWN LINKS & IMAGES -->
[python-version]: https://img.shields.io/badge/python-3.11.5-blue.svg
