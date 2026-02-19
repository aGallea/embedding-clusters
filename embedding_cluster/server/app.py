from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from embedding_cluster.server.routes.collections import (
    router as collections_router,
)
from embedding_cluster.server.routes.csv import router as csv_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Embedding Clusters",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(collections_router)
    app.include_router(csv_router)

    return app
