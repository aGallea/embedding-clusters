from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from embedding_cluster.server.routes.ai import router as ai_router
from embedding_cluster.server.routes.annotations import (
    router as annotations_router,
)
from embedding_cluster.server.routes.collections import (
    router as collections_router,
)
from embedding_cluster.server.routes.csv import router as csv_router
from embedding_cluster.server.routes.index import router as index_router
from embedding_cluster.server.routes.plot import router as plot_router
from embedding_cluster.server.routes.search import (
    router as search_router,
)

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


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

    app.include_router(ai_router)
    app.include_router(collections_router)
    app.include_router(csv_router)
    app.include_router(index_router)
    app.include_router(plot_router)
    app.include_router(annotations_router)
    app.include_router(search_router)

    if FRONTEND_DIR.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=FRONTEND_DIR / "assets"),
            name="static-assets",
        )

        @app.get("/{full_path:path}")
        async def serve_spa(request: Request, full_path: str) -> FileResponse:
            """Serve the React SPA for any non-API route."""
            file_path = (FRONTEND_DIR / full_path).resolve()
            if not str(file_path).startswith(str(FRONTEND_DIR.resolve())):
                return FileResponse(FRONTEND_DIR / "index.html")
            if file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(FRONTEND_DIR / "index.html")
    else:
        logger.warning(
            "Frontend build not found at %s. "
            "Run 'npm run build' in frontend/ to enable the web UI.",
            FRONTEND_DIR,
        )

    return app
