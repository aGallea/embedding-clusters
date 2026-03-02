from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

import chromadb
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html, no_update
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from collections.abc import Callable

    from chromadb.api import ClientAPI

    from embedding_cluster.settings import Settings

logger = logging.getLogger(__name__)


def reduce_dimensions(
    embeddings: np.ndarray,
    algorithm: str = "tsne",
    n_components: int = 3,
    random_state: int = 171,
    **kwargs: Any,
) -> np.ndarray:
    """Reduce embedding dimensions using the specified algorithm."""
    if algorithm == "tsne":
        perplexity = kwargs.get("perplexity", 30.0)
        learning_rate = kwargs.get("learning_rate", "auto")
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=1,
            max_iter=1000,
        )
    elif algorithm == "umap":
        try:
            import umap
        except ImportError as exc:
            msg = (
                "umap-learn is not installed. Install it with: uv pip install umap-learn"
            )
            raise ImportError(msg) from exc
        n_neighbors = kwargs.get("n_neighbors", 15)
        min_dist = kwargs.get("min_dist", 0.1)
        metric = kwargs.get("metric", "cosine")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
    elif algorithm == "pca":
        reducer = PCA(n_components=n_components)
    else:
        msg = f"Unknown reduction algorithm: '{algorithm}'. Supported: tsne, umap, pca"
        raise ValueError(msg)
    result: np.ndarray = reducer.fit_transform(embeddings)
    return result


def gpt_get_cluster_name(info: str, settings: Settings) -> str:
    openai_client = OpenAI()
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Your role is to find a very short (max 5 words), concise "
                "name for a group of items, one name to rule them all. "
                "the user will provide a list of item names. do your best"
            ),
        },
        {
            "role": "user",
            "content": info,
        },
    ]
    completion = openai_client.chat.completions.create(
        model=settings.gpt_default_model,
        temperature=settings.gpt_default_temperature,
        messages=messages,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    )
    content = completion.choices[0].message.content or ""
    return (content[:30] + "..") if len(content) > 30 else content


def load_chromadb_collection(settings: Settings) -> Any:
    chromadb_client: ClientAPI = chromadb.PersistentClient(path="./chromadb")
    collection = chromadb_client.get_or_create_collection(
        settings.chromadb_collection_name
    )
    return collection.get(include=["embeddings", "metadatas"])  # type: ignore[list-item]  # pyright: ignore[reportArgumentType]


def load_chromadb_embeddings(collection_name: str) -> np.ndarray:
    """Load embeddings from a ChromaDB collection by name."""
    chromadb_client: ClientAPI = chromadb.PersistentClient(path="./chromadb")
    try:
        collection = chromadb_client.get_collection(collection_name)
    except Exception as exc:
        msg = f"Collection '{collection_name}' not found"
        raise ValueError(msg) from exc
    result = collection.get(include=["embeddings"])  # type: ignore[list-item]  # pyright: ignore[reportArgumentType]
    embeddings = result.get("embeddings")
    if embeddings is None:
        msg = f"No embeddings found in collection '{collection_name}'"
        raise ValueError(msg)
    return np.array(embeddings)


def suggest_optimal_clusters(
    embeddings: np.ndarray,
    k_range: range = range(2, 31),
    max_samples: int = 5000,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run elbow method and silhouette analysis to suggest optimal k."""
    random_state = 171

    if len(embeddings) > max_samples:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(embeddings), size=max_samples, replace=False)
        sampled = embeddings[indices]
    else:
        sampled = embeddings

    scaled = StandardScaler().fit_transform(sampled)

    k_values: list[int] = []
    inertias: list[float] = []
    silhouette_scores_list: list[float] = []

    for k in k_range:
        kmeans = KMeans(
            n_clusters=k,
            n_init="auto",
            random_state=random_state,
            max_iter=1000,
        )
        labels = kmeans.fit_predict(scaled)
        k_values.append(k)
        inertia = kmeans.inertia_
        if inertia is None:
            inertia = 0.0
        inertias.append(float(inertia))
        silhouette_scores_list.append(float(silhouette_score(scaled, labels)))
        if on_progress is not None:
            on_progress(
                {
                    "phase": "analyzing",
                    "current_k": k,
                    "total_k": len(k_range),
                }
            )

    suggested_k = k_values[int(np.argmax(silhouette_scores_list))]

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores_list,
        "suggested_k": suggested_k,
    }


def get_field_as_list(metadata: list[dict[str, Any]], field_name: str) -> list[Any]:
    return [line[field_name] for line in metadata]


def create_collection_text_display(
    metadata: list[dict[str, Any]],
    text_display_fields: list[str],
    seperator: str = ",",
) -> list[str]:
    fields_content: list[str] = []
    for field in text_display_fields:
        if len(fields_content) > 0:
            new_content = [line[field] for line in metadata]
            fields_content = [
                a + seperator + b
                for a, b in zip(fields_content, new_content, strict=False)
            ]
        else:
            fields_content = [line[field] for line in metadata]
    return fields_content


def generate_cluster_props(
    num_clusters: int,
    pred_arr: Any,
    collection_content_text_display: list[str],
    settings: Settings,
    num_products_for_cluster_name: int = 10,
) -> tuple[list[list[int]], list[str]]:
    clusters_indices: list[list[int]] = []
    cluster_names: list[str] = []
    group_index = 1
    for cluster_i in range(num_clusters):
        curr_cluster_indices = [i for i, x in enumerate(pred_arr) if x == cluster_i]
        clusters_indices.append(curr_cluster_indices)
        logger.info("Generating cluster %d names ...", cluster_i)
        if settings.gpt_generate_cluster_name is True:
            random_product_indexes = random.sample(
                range(0, len(curr_cluster_indices)),
                min(
                    num_products_for_cluster_name,
                    len(curr_cluster_indices),
                ),
            )
            curr_descriptions = ""
            for product_index in random_product_indexes:
                idx = curr_cluster_indices[product_index]
                item = (
                    collection_content_text_display[idx]
                    if idx < len(collection_content_text_display)
                    else f"Item {idx}"
                )
                curr_descriptions += f"name: {item} \n"
            cluster_name = gpt_get_cluster_name(curr_descriptions, settings)
            cluster_names.append(cluster_name)
        else:
            cluster_names.append(f"Group {group_index}")
            group_index += 1
    return clusters_indices, cluster_names


# Module-level state shared between prepare_data and the Dash callback.
# These are populated during prepare_data() and read by display_hover().
cluster_images: list[list[str]] = []
cluster_item_names: list[list[str]] = []


def compute_plot_data(settings: Settings) -> dict[str, Any]:
    """Compute dimensionality reduction + k-means and return raw data."""
    random_state = 171
    n_iter = 1000
    collection_content_text_display: list[str] = []
    num_clusters = settings.num_clusters

    collection_content = load_chromadb_collection(settings)
    logger.info("Read %d items", len(collection_content["ids"]))
    if settings.text_display_fields is not None:
        collection_content_text_display = create_collection_text_display(
            collection_content["metadatas"], settings.text_display_fields
        )
    collection_content_vectors = collection_content["embeddings"]
    np_embeddings_arr = np.array(collection_content_vectors)

    algorithm = settings.reduction_algorithm
    logger.info("Calculating %s ...", algorithm.upper())
    reduction_kwargs: dict[str, Any] = {}
    if algorithm == "tsne":
        reduction_kwargs["perplexity"] = settings.tsne_perplexity
        reduction_kwargs["learning_rate"] = settings.tsne_learning_rate
    elif algorithm == "umap":
        reduction_kwargs["n_neighbors"] = settings.umap_n_neighbors
        reduction_kwargs["min_dist"] = settings.umap_min_dist
        reduction_kwargs["metric"] = settings.umap_metric
    reduced = reduce_dimensions(
        np_embeddings_arr,
        algorithm=algorithm,
        n_components=3,
        random_state=random_state,
        **reduction_kwargs,
    )

    common_params: dict[str, Any] = {
        "n_init": "auto",
        "random_state": random_state,
        "max_iter": n_iter,
    }
    embeddings_standardized = StandardScaler().fit_transform(np_embeddings_arr)
    logger.info("Calculating K-Means ...")
    pred_arr = KMeans(n_clusters=num_clusters, **common_params).fit_predict(
        embeddings_standardized
    )

    clusters_indices, cluster_names = generate_cluster_props(
        num_clusters, pred_arr, collection_content_text_display, settings
    )

    # Build structured point data
    points: list[dict[str, Any]] = []
    clusters: list[dict[str, Any]] = []

    display_fields = settings.text_display_fields or []

    for cluster_i in range(num_clusters):
        color = f"hsl({cluster_i * 360 // num_clusters}, 70%, 50%)"
        clusters.append(
            {
                "index": cluster_i,
                "name": cluster_names[cluster_i],
                "color": color,
                "count": len(clusters_indices[cluster_i]),
            }
        )

        for idx in clusters_indices[cluster_i]:
            metadata: dict[str, Any] = {}
            if idx < len(collection_content["metadatas"]):
                raw_metadata = dict(collection_content["metadatas"][idx])
                if display_fields:
                    metadata = {
                        key: value
                        for key, value in raw_metadata.items()
                        if key in display_fields
                    }
                else:
                    metadata = raw_metadata
            point_id = (
                collection_content["ids"][idx]
                if idx < len(collection_content["ids"])
                else str(idx)
            )
            points.append(
                {
                    "x": float(reduced[idx, 0]),
                    "y": float(reduced[idx, 1]),
                    "z": float(reduced[idx, 2]),
                    "cluster": cluster_i,
                    "metadata": metadata,
                    "id": point_id,
                }
            )

    return {
        "points": points,
        "clusters": clusters,
        "total_points": len(collection_content["ids"]),
    }


def prepare_data(settings: Settings) -> go.Figure:
    logger.info("Preparing data ...")
    global cluster_images
    global cluster_item_names
    cluster_images = []
    cluster_item_names = []

    plot_data = compute_plot_data(settings)

    # Rebuild cluster_images and cluster_item_names from plot_data
    num_clusters = settings.num_clusters
    # Group points by cluster
    cluster_points: dict[int, list[dict[str, Any]]] = {}
    for point in plot_data["points"]:
        cluster_i = point["cluster"]
        if cluster_i not in cluster_points:
            cluster_points[cluster_i] = []
        cluster_points[cluster_i].append(point)

    data: list[go.Scatter3d] = []
    for cluster_i in range(num_clusters):
        c_points = cluster_points.get(cluster_i, [])
        curr_images = []
        curr_names = []
        for p in c_points:
            img = (
                p["metadata"].get(
                    settings.image_field,
                    "https://upload.wikimedia.org/wikipedia/commons/5/5a/Black_question_mark.png",
                )
                if settings.image_field
                else "https://upload.wikimedia.org/wikipedia/commons/5/5a/Black_question_mark.png"
            )
            curr_images.append(img)
            if settings.text_display_fields:
                name_parts = [
                    str(p["metadata"].get(f, "")) for f in settings.text_display_fields
                ]
                curr_names.append(",".join(name_parts))
            else:
                curr_names.append(f"Item {p['id']}")

        cluster_name = next(
            (c["name"] for c in plot_data["clusters"] if c["index"] == cluster_i),
            f"Group {cluster_i}",
        )
        trace = go.Scatter3d(
            x=np.array([p["x"] for p in c_points]),
            y=np.array([p["y"] for p in c_points]),
            z=np.array([p["z"] for p in c_points]),
            mode="markers",
            name=cluster_name,
            showlegend=True,
            marker=dict(size=5, color=cluster_i),
        )
        cluster_images.append(curr_images)
        cluster_item_names.append(curr_names)
        data.append(trace)

    fig = go.Figure(data=data)
    fig.update_traces(hoverinfo="none", hovertemplate=None)
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=1000,
        paper_bgcolor="LightSteelBlue",
    )
    return fig


@callback(
    Output("scatter-graph-tooltip", "show"),
    Output("scatter-graph-tooltip", "bbox"),
    Output("scatter-graph-tooltip", "children"),
    Input("scatter-graph", "hoverData"),
)
def display_hover(hover_data_input: dict[str, Any] | None) -> tuple[bool, Any, Any]:
    if hover_data_input is None:
        return False, no_update, no_update

    hover_data = hover_data_input["points"][0]
    bbox = hover_data["bbox"]
    num = hover_data["pointNumber"]
    cluster = hover_data["curveNumber"]

    children = [
        html.Div(
            [
                html.Img(
                    src=cluster_images[cluster][num],
                    style={
                        "width": "100px",
                        "display": "block",
                        "margin": "0 auto",
                    },
                ),
                html.P(
                    str(cluster_item_names[cluster][num]),
                    style={"font-weight": "bold"},
                ),
            ]
        )
    ]
    return True, bbox, children


async def main_scatter_plot(settings: Settings) -> None:
    app = Dash(__name__)
    fig = prepare_data(settings)
    app.layout = html.Div(
        className="container",
        children=[
            dcc.Graph(id="scatter-graph", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="scatter-graph-tooltip", direction="bottom"),
        ],
    )
    app.run()
