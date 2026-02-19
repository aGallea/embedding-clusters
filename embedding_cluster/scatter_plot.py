from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html, no_update
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import chromadb

if TYPE_CHECKING:
    from chromadb.api import ClientAPI

    from embedding_cluster.settings import Settings

logger = logging.getLogger(__name__)


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
        messages=messages,  # type: ignore[arg-type]
    )
    content = completion.choices[0].message.content or ""
    return (content[:30] + "..") if len(content) > 30 else content


def load_chromadb_collection(settings: Settings) -> Any:
    chromadb_client: ClientAPI = chromadb.PersistentClient(path="./chromadb")
    collection = chromadb_client.get_or_create_collection(
        settings.chromadb_collection_name
    )
    return collection.get(include=["embeddings", "metadatas"])  # type: ignore[list-item]


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


def prepare_data(settings: Settings) -> go.Figure:
    logger.info("Preparing data ...")
    random_state = 171
    n_iter = 1000
    collection_content_images: list[str] = []
    collection_content_text_display: list[str] = []
    num_clusters = settings.num_clusters
    global cluster_images
    global cluster_item_names
    cluster_images = []
    cluster_item_names = []
    collection_content = load_chromadb_collection(settings)
    logger.info("Read %d items", len(collection_content["ids"]))
    if settings.image_field is not None:
        collection_content_images = get_field_as_list(
            collection_content["metadatas"], settings.image_field
        )
    if settings.text_display_fields is not None:
        collection_content_text_display = create_collection_text_display(
            collection_content["metadatas"], settings.text_display_fields
        )
    collection_content_vectors = collection_content["embeddings"]

    np_embeddings_arr = np.array(collection_content_vectors)
    logger.info("Calculating t-SNE ...")
    tsne = TSNE(
        verbose=1,
        learning_rate="auto",
        n_iter=n_iter,
        perplexity=30,
        n_components=3,
        random_state=random_state,
    ).fit_transform(np_embeddings_arr)

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
        num_clusters,
        pred_arr,
        collection_content_text_display,
        settings,
    )

    data: list[go.Scatter3d] = []
    for cluster_i in range(num_clusters):
        curr_images = [
            (
                "https://upload.wikimedia.org/wikipedia/commons/5/5a/Black_question_mark.png"
                if len(collection_content_images) <= x
                else collection_content_images[x]
            )
            for x in clusters_indices[cluster_i]
        ]
        curr_names = [
            (
                collection_content_text_display[x]
                if x < len(collection_content_text_display)
                else f"Item {x}"
            )
            for x in clusters_indices[cluster_i]
        ]
        trace = go.Scatter3d(
            x=np.array([tsne[x, 0] for x in clusters_indices[cluster_i]]),
            y=np.array([tsne[x, 1] for x in clusters_indices[cluster_i]]),
            z=np.array([tsne[x, 2] for x in clusters_indices[cluster_i]]),
            mode="markers",
            name=cluster_names[cluster_i],
            showlegend=True,
            marker=dict(
                size=5,
                color=cluster_i,
            ),
        )
        cluster_images.append(curr_images)
        cluster_item_names.append(curr_names)
        data.append(trace)
    fig = go.Figure(data=data)

    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    )
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
