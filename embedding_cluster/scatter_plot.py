import logging
import random
from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go
from chromadb.api import ClientAPI
from dash import Dash, Input, Output, callback, dcc, html, no_update
from openai import OpenAI

# from openai.types.chat import ChatCompletion
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import chromadb
from embedding_cluster.settings import Settings
from embedding_cluster.utils import get_or_create_chromadb_collections

# import pandas as pd
logger = logging.getLogger(__name__)


def gpt_get_cluster_name(info: str, settings: Settings):
    openai_client = OpenAI(
        # api_key="bla",
        # base_url="http://localhost:8000/v1"
    )
    messages = [
        {
            "role": "system",
            "content": "Your role is to find a very short (max 5 words), concise name for a group of items, one name to rule them all. the user will provide a list of item names. do your best",
        }
    ]
    messages.append(
        {
            "role": "user",
            "content": info,
        }
    )
    completion = openai_client.chat.completions.create(
        model=settings.gpt_default_model,
        temperature=settings.gpt_default_temperature,
        messages=messages,
    )
    content = completion.choices[0].message.content
    return (content[:30] + "..") if len(content) > 30 else content


def load_chromadb_collection(settings: Settings):
    chromadb_client: ClientAPI = chromadb.PersistentClient(path="./chromadb")
    collection = chromadb_client.get_or_create_collection(
        settings.chromadb_collection_name
    )
    return collection.get(include=["embeddings", "metadatas"])


def get_field_as_list(metadata: list[Dict[str, Any]], field_name: str) -> list[Any]:
    return [line[field_name] for line in metadata]


def create_collection_text_display(
    metadata: list[Dict[str, Any]], text_display_fields: list[str], seperator=","
) -> list[str]:
    fields_content: list[str] = []
    for field in text_display_fields:
        if len(fields_content) > 0:
            new_content = [line[field] for line in metadata]
            fields_content = [
                a + seperator + b for a, b in zip(fields_content, new_content)
            ]
        else:
            fields_content = [line[field] for line in metadata]
    return fields_content


def generate_cluster_props(
    num_clusters: int,
    pred_arr: list[int],
    collection_content_text_display: list[str],
    settings: Settings,
    num_products_for_cluster_name=10,
):
    clusters_indices = []
    cluster_names = []
    group_index = 1
    for cluster_i in range(num_clusters):
        curr_cluster_indices = [i for i, x in enumerate(pred_arr) if x == cluster_i]
        clusters_indices.append(curr_cluster_indices)
        logger.info(f"Generating cluster {cluster_i} names ...")
        if settings.gpt_generate_cluster_name is True:
            random_product_indexes = random.sample(
                range(0, len(curr_cluster_indices)),
                min(num_products_for_cluster_name, len(curr_cluster_indices)),
            )
            curr_descriptions = ""
            for product_index in random_product_indexes:
                curr_descriptions += f"name: {collection_content_text_display[curr_cluster_indices[product_index]]} \n"
            cluster_name = gpt_get_cluster_name(curr_descriptions, settings)
            cluster_names.append(cluster_name)
        else:
            cluster_names.append(f"Group {group_index}")
            group_index += 1
    return clusters_indices, cluster_names


cluster_images = []
cluster_item_names = []


def prepare_data(settings: Settings):
    logger.info("Preparing data ...")
    random_state = 171
    n_iter = 1000
    collection_content_images = []
    collection_content_text_display = []
    num_clusters = settings.num_clusters
    global cluster_images
    global cluster_item_names
    collection_content = load_chromadb_collection(settings)
    logger.info(f"Read {len(collection_content['ids'])} items")
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

    common_params = {
        "n_init": "auto",
        "random_state": random_state,
        "max_iter": n_iter,
    }
    embeddings_standardized = StandardScaler().fit_transform(np_embeddings_arr)
    # embeddings_standardized = StandardScaler().fit_transform(tsne)
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

    data = []
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
            collection_content_text_display[x] for x in clusters_indices[cluster_i]
        ]
        trace = go.Scatter3d(
            x=np.array([tsne[x, 0] for x in clusters_indices[cluster_i]]),
            y=np.array([tsne[x, 1] for x in clusters_indices[cluster_i]]),
            z=np.array([tsne[x, 2] for x in clusters_indices[cluster_i]]),
            mode="markers",
            name=cluster_names[cluster_i],  # f"group-{cluster_i}",
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
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    num = hover_data["pointNumber"]
    cluster = hover_data["curveNumber"]

    # logger.info(f"point number: {num}")
    children = [
        html.Div(
            [
                html.Img(
                    src=cluster_images[cluster][num],
                    style={"width": "100px", "display": "block", "margin": "0 auto"},
                ),
                html.P(
                    str(cluster_item_names[cluster][num]),
                    style={"font-weight": "bold"},
                ),
            ]
        )
    ]
    return True, bbox, children


async def main_scatter_plot():
    settings = Settings()
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
