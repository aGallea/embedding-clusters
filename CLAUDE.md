# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python + React application for generating, indexing, and visualizing
embedding clusters from CSV data. Uses CLIP/SentenceTransformer for
embeddings, ChromaDB for vector storage, k-means for clustering, and a
React/Three.js frontend for 3D visualization.

- **Python 3.13**, managed with [uv](https://docs.astral.sh/uv/)
- **Package name**: `embedding_cluster` (underscore, not hyphen)
- **Entry point**: `python -m embedding_cluster` dispatches to INDEX, PLOT, or SERVER mode via `RUNNING_MODE` env var

## Commands

### Backend

```bash
uv sync --all-extras                                    # Install all dependencies
RUNNING_MODE=SERVER uv run python -m embedding_cluster  # Start server on :8000
uv run ruff check embedding_cluster/ tests/             # Lint
uv run ruff check --fix embedding_cluster/ tests/       # Lint with auto-fix
uv run ruff format embedding_cluster/ tests/            # Format
uv run mypy embedding_cluster/                          # Type check (strict mode)
uv run pytest                                           # Run all tests
uv run pytest tests/test_settings.py -v                 # Run single test file
uv run pytest tests/test_settings.py::test_fn -v        # Run single test function
uv run pytest --cov=embedding_cluster --cov-report=term-missing --cov-fail-under=90  # Coverage (90% CI min)
uv run pre-commit run --all-files                       # All pre-commit hooks
```

### Frontend

```bash
cd frontend && npm install                  # Install deps
cd frontend && npm run dev                  # Dev server on :5173
cd frontend && npm run build                # Production build (output: frontend/dist)
cd frontend && npm run lint                 # ESLint
cd frontend && npm run test:e2e             # Playwright E2E tests
cd frontend && npx playwright test e2e/search.spec.ts  # Single E2E test
```

E2E tests require pre-indexed ChromaDB data and a built frontend. The Playwright config auto-starts the FastAPI backend.

## Architecture

### Three Running Modes

All controlled by `RUNNING_MODE` env var, dispatched in `__main__.py`:

- **INDEX**: `indexer.py` — CSV parsing → embedding generation → ChromaDB storage
- **PLOT**: `scatter_plot.py` — ChromaDB → StandardScaler → k-means → dimensionality reduction (t-SNE/UMAP/PCA)
- **SERVER**: `server/app.py` — FastAPI backend serving REST API + built React SPA from `frontend/dist`

### Backend Structure

- `settings.py` — All config via env vars using `pydantic-settings` `BaseSettings`
- `server/app.py` — FastAPI app factory, mounts route modules and serves SPA
- `server/routes/` — API routes split by domain: `ai.py`, `annotations.py`, `collections.py`, `csv.py`, `index.py`, `plot.py`, `search.py`
- `server/tasks.py` — Background task management for long-running operations
- `server/ws.py` — WebSocket support for live progress
- `ai_naming.py` — LLM-powered cluster naming via LiteLLM (supports OpenAI, Ollama)
- `annotations.py` — Cluster annotation persistence (JSON sidecar files in `annotations/`)
- `utils.py` — ChromaDB helpers, image downloader with retry, singleton pattern

### Frontend Structure

React 19 + TypeScript + Vite + Tailwind CSS 4:

- `pages/` — `HomePage`, `IndexPage`, `PlotPage`, `SettingsPage`
- `components/` — Organized by page: `home/`, `index/`, `plot/`, `csv/`
- `stores/plotStore.ts` — Zustand store for plot state
- `api/` — API client layer
- `hooks/` — React Query hooks
- 3D visualization uses React Three Fiber (`@react-three/fiber` + `@react-three/drei`)

## Code Style

### Python

- **ruff**: line length 90, target py313
- **mypy strict mode** — all functions need type annotations
- Use `from __future__ import annotations` in every module
- Modern syntax: `str | None` (not `Optional`), `list[str]` (not `List`)
- Absolute imports only: `from embedding_cluster.settings import Settings`
- Heavy imports behind `TYPE_CHECKING` blocks where possible
- Logger per module: `logger = logging.getLogger(__name__)`

### Git Conventions

- **Conventional commits** enforced by commitizen: `type(scope): description`
- Types: `feat`, `fix`, `docs`, `test`, `refactor`
- **No direct commits to master** (enforced by pre-commit hook)
- Branch naming: `feature-name` style (e.g., `feat/ollama-provider-integration`)

### Pre-commit Hooks

Extensive setup including: ruff, commitizen, yamllint, markdownlint,
shellcheck, gitleaks, hadolint, check-jsonschema, no-commit-to-branch.
Install with:

```bash
uv run pre-commit install --install-hooks -t pre-commit -t commit-msg
```

## CI

GitHub Actions (`.github/workflows/ci.yml`): lint → typecheck → test (90% coverage minimum). All jobs use `uv sync --all-extras`.
