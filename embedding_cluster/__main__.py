from __future__ import annotations

import asyncio

from embedding_cluster.settings import Settings
from embedding_cluster.utils import init_logger


def main() -> None:
    settings = Settings()
    init_logger()
    if settings.running_mode == "INDEX":
        from embedding_cluster.indexer import main_indexer

        asyncio.run(main_indexer(settings))
    elif settings.running_mode == "PLOT":
        from embedding_cluster.scatter_plot import main_scatter_plot

        asyncio.run(main_scatter_plot(settings))


if __name__ == "__main__":
    main()
