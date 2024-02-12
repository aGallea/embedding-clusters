import asyncio

from embedding_cluster.indexer import main_indexer
from embedding_cluster.scatter_plot import main_scatter_plot
from embedding_cluster.settings import Settings
from embedding_cluster.utils import init_logger

if __name__ == "__main__":
    settings = Settings()
    init_logger()
    if settings.running_mode == "INDEX":
        asyncio.run(main_indexer())
    elif settings.running_mode == "PLOT":
        asyncio.run(main_scatter_plot())
