# AGENTS.md

## Project Overview

Python CLI tool for generating, indexing, and visualizing embedding clusters from CSV data.
Uses CLIP/transformer models for embeddings, ChromaDB for storage, k-means for clustering,
and Dash/Plotly for 3D t-SNE visualization.

- **Language**: Python 3.13
- **Package manager**: [uv](https://docs.astral.sh/uv/)
- **Package**: `embedding_cluster` (single underscore, not hyphenated)
- **Entry point**: `python -m embedding_cluster` (runs `__main__.py`)
- **Config**: Environment variables via `pydantic-settings` (`Settings` class in `settings.py`)

## Build & Run Commands

```bash
# Setup (requires uv - https://docs.astral.sh/uv/getting-started/installation/)
uv sync --all-extras

# Run indexing mode
RUNNING_MODE=INDEX LOCAL_CSV_FILENAME=./embedding_cluster/csv/fashion_small.csv \
  ID_FIELD=id IMAGE_EMBEDDING_FIELDS='["imageUrl"]' \
  CHROMADB_COLLECTION_PREFIX=fashion_ NUMBER_OF_ASYNC_TASKS=10 \
  uv run python -m embedding_cluster

# Run plot mode
RUNNING_MODE=PLOT CHROMADB_COLLECTION_NAME=fashion_imageUrl \
  TEXT_DISPLAY_FIELDS='["productDisplayName"]' IMAGE_FIELD=imageUrl \
  uv run python -m embedding_cluster
```

## Linting & Formatting

```bash
# Lint (ruff, line length 90, target Python 3.13)
uv run ruff check embedding_cluster/ tests/

# Auto-fix lint issues
uv run ruff check --fix embedding_cluster/ tests/

# Format
uv run ruff format embedding_cluster/ tests/

# Check formatting (CI mode)
uv run ruff format --check embedding_cluster/ tests/

# Type checking (mypy strict mode)
uv run mypy embedding_cluster/

# Run all pre-commit hooks
uv run pre-commit run --all-files

# Install pre-commit hooks (first-time setup)
uv run pre-commit install --install-hooks -t pre-commit -t commit-msg
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=embedding_cluster --cov-report=term-missing

# Run with coverage enforcement (90% minimum, matches CI)
uv run pytest --cov=embedding_cluster --cov-report=term-missing --cov-fail-under=90

# Run a single test file
uv run pytest tests/test_settings.py -v

# Run a single test function
uv run pytest tests/test_settings.py::test_function_name -v
```

Test configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`:

- `testpaths = ["tests"]`
- `asyncio_mode = "auto"` (pytest-asyncio auto mode)

## E2E Testing

```bash
# Install Playwright browsers (first-time)
cd frontend && npx playwright install chromium

# Index sample data for E2E tests (first-time, from project root)
RUNNING_MODE=INDEX LOCAL_CSV_FILENAME=./embedding_cluster/csv/fashion_small.csv \
  ID_FIELD=id TEXT_EMBEDDING_FIELDS='["productDisplayName"]' \
  CHROMADB_COLLECTION_PREFIX=fashion_ uv run python -m embedding_cluster

# Build frontend (required before E2E)
cd frontend && npm run build

# Run E2E tests
cd frontend && npm run test:e2e

# Run E2E tests with UI
cd frontend && npm run test:e2e:ui

# Run single test file
cd frontend && npx playwright test e2e/search.spec.ts
```

E2E tests require pre-indexed ChromaDB data. The `webServer` config in
`playwright.config.ts` auto-starts the FastAPI backend. Tests run against
`http://localhost:8000`.

## CI

GitHub Actions workflow in `.github/workflows/ci.yml` runs on push/PR:

- **lint** job: `ruff check` + `ruff format --check`
- **typecheck** job: `mypy embedding_cluster/`
- **test** job: `pytest --cov` (90% minimum enforced by coverage report)

All jobs use `uv sync --all-extras` for dependency installation.

## Code Style

### Formatting

- **ruff** with line length **90**, target `py313`
- Import sorting via ruff's isort rules (`I` select)
- `embedding_cluster` as known first-party in `[tool.ruff.lint.isort]`

### Imports

- Standard library first, then third-party, then local (ruff isort enforces this)
- Third-party imports use direct names: `import chromadb`, `from pydantic import Field`
- Local imports use full package path: `from embedding_cluster.settings import Settings`
- No relative imports (all local imports are absolute from `embedding_cluster`)
- Use `from __future__ import annotations` in every module
- Heavy imports behind `TYPE_CHECKING` blocks where possible

### Type Hints

- **mypy strict mode** configured in `pyproject.toml`
- Use modern union syntax: `str | None` (not `Optional[str]`)
- Use built-in generics: `list[str]`, `dict[str, Any]` (not `List`, `Dict`)
- `collections.abc.Sequence` / `collections.abc.Mapping` for abstract types
- Function signatures must have type annotations for parameters and return types
- `X | None` with `Field(default=None)` for nullable pydantic fields
- Third-party library stubs are set to `ignore_missing_imports` in mypy overrides

### Naming Conventions

- **snake_case** for functions, variables, modules
- **PascalCase** for classes
- **UPPER_SNAKE_CASE** for environment variable names in Settings field mappings
- Logger per module: `logger = logging.getLogger(__name__)`

### Configuration

- All config via **environment variables** parsed by `pydantic-settings` `BaseSettings`
- Each setting has a `Field()` with `default` and `description`
- List fields accept JSON-encoded strings from env vars (e.g., `'["field1","field2"]'`)

### Error Handling

- Use `logging` module (not print statements)
- Log levels: `logger.info()`, `logger.warning()`, `logger.error()`, `logger.exception()`
- **Never** use `logger.warn()` (deprecated alias for `warning()`)
- Retry logic with exponential backoff for network operations (see `ImageDownloader`)
- Catch specific exceptions, not bare `except:`

### Async Patterns

- `asyncio.run()` at entry point in `__main__.py`
- `asyncio.Semaphore` for concurrency limiting
- `asyncio.ensure_future()` + `asyncio.gather()` for parallel task execution
- `aiohttp.ClientSession` for async HTTP (lazy init via Singleton pattern)

### Project Structure

```text
embedding_cluster/
  __init__.py          # Empty
  __main__.py          # Entry point, mode dispatch via main()
  py.typed             # PEP 561 marker for typed package
  settings.py          # Pydantic Settings (env var config)
  utils.py             # Shared utilities (logging, ChromaDB helpers, image downloader)
  indexer.py           # INDEX mode: CSV parsing, embedding generation, ChromaDB storage
  scatter_plot.py      # PLOT mode: Clustering, dimensionality reduction, visualization data
  ai_naming.py         # LLM-powered cluster naming via LiteLLM
  annotations.py       # Cluster annotation persistence (JSON sidecar files)
  csv/                 # Sample data files
  server/
    app.py             # FastAPI app factory, SPA serving
    models.py          # Pydantic request/response models
    tasks.py           # Background task registry
    ws.py              # WebSocket manager for live progress
    routes/
      ai.py            # AI cluster naming endpoints
      annotations.py   # Cluster annotation CRUD
      collections.py   # ChromaDB collection management
      csv.py           # CSV upload and preview
      index.py         # Indexing jobs with WebSocket progress
      plot.py          # Plot computation, cluster detail, sub-clustering
      search.py        # Semantic search (text and image)
frontend/
  src/
    App.tsx            # Router, QueryClient, Zustand provider
    api/               # Typed API client layer
    components/        # UI components organized by page
    hooks/             # useIndexWebSocket, usePlotData
    pages/             # HomePage, IndexPage, PlotPage, SettingsPage
    stores/            # Zustand plotStore (plot state management)
    types/             # TypeScript interfaces mirroring backend models
tests/
  conftest.py          # Shared fixtures
  test_*.py            # Unit tests for each backend module and route
```

### Key Dependencies

Runtime:

- `pydantic` / `pydantic-settings` - Configuration and data models
- `chromadb` - Vector database for embedding storage
- `transformers` / `sentence-transformers` - Text and image embedding models
- `torch` - ML framework backend
- `fastapi` / `uvicorn` - Web server and REST API
- `scikit-learn` - KMeans clustering and dimensionality reduction
- `aiohttp` - Async HTTP for image downloads
- `litellm` - Multi-provider LLM integration for cluster naming
- `numpy` / `Pillow` - Numerical and image processing

Dev:

- `pytest` / `pytest-asyncio` / `pytest-cov` - Testing framework
- `mypy` - Static type checking
- `ruff` - Linting and formatting
- `pre-commit` - Git hook management
- `httpx` - Test client for FastAPI routes

## Git & Commit Conventions

- **Commitizen** enforced via pre-commit hook (commit-msg stage)
- Conventional commits format: `type(scope): description`
- Types used in history: `feat`, `fix`, `docs`
- Examples from repo:
  - `feat: initial commit`
  - `feat(logger): formatter`
  - `fix(index): set start and stop lines`
  - `docs(readme): usage`
- Direct commits to `master` are blocked by pre-commit (`no-commit-to-branch`)
- Branch naming: `feature-name` style (e.g., `readme-fix`, `logger`)

## Pre-commit Hooks

Extensive pre-commit setup. Key hooks:

- **ruff** - Python linting (with `--fix`) and formatting
- **commitizen** - Commit message linting
- **yamllint** - YAML linting (max line 300)
- **markdownlint** - Markdown linting (max line 140)
- **shellcheck** + **shfmt** - Shell script linting/formatting
- **gitleaks** - Secret detection
- **hadolint** - Dockerfile linting
- **check-jsonschema** - Validates GitHub workflows, actions, dependabot, etc.
- **no-commit-to-branch** - Prevents direct commits to master

## Data Flow

1. **INDEX mode**: CSV → parse rows → generate embeddings (CLIP for images,
   SentenceTransformer for text) → store in ChromaDB collections
2. **PLOT mode**: ChromaDB collection → StandardScaler → KMeans clustering →
   dimensionality reduction (t-SNE/UMAP/PCA) → 3D point data via REST API
3. **SERVER mode**: FastAPI serves REST API + built React SPA. Long-running
   jobs (indexing, plot computation) use a task registry with WebSocket
   progress streaming.

Persistent data:

- `./chromadb/` — Vector database (gitignored)
- `./uploads/` — Uploaded CSV files (gitignored)
- `./annotations/` — Cluster annotations as JSON sidecar files (gitignored)
