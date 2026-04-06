# Contributing

Thanks for your interest in contributing to embedding-clusters! This guide
covers everything you need to get started.

## Prerequisites

- [Python 3.13+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package
  manager
- [Node.js 18+](https://nodejs.org/) (for frontend development)

## Setup

```bash
git clone https://github.com/aGallea/embedding-clusters.git
cd embedding-clusters
uv sync --all-extras
uv run pre-commit install --install-hooks -t pre-commit -t commit-msg
```

For frontend work:

```bash
cd frontend
npm install
```

## Running Locally

Start the full application (backend + frontend):

```bash
RUNNING_MODE=SERVER uv run python -m embedding_cluster
```

For frontend development with hot reload:

```bash
# Terminal 1 — backend
RUNNING_MODE=SERVER uv run python -m embedding_cluster

# Terminal 2 — frontend dev server (proxies API to backend)
cd frontend && npm run dev
```

The Vite dev server runs on `http://localhost:5173` and proxies `/api` and
`/ws` requests to the backend on port 8000.

## Testing

### Backend (Python)

```bash
uv run pytest                                  # Run all tests
uv run pytest tests/test_settings.py -v        # Single file
uv run pytest tests/test_settings.py::test_fn  # Single test
uv run pytest --cov=embedding_cluster \
  --cov-report=term-missing --cov-fail-under=90  # With coverage
```

Tests use `pytest-asyncio` in auto mode. CI enforces a **90% minimum
coverage** threshold.

### Frontend (E2E)

```bash
cd frontend
npx playwright install chromium     # First-time setup
npm run build                       # Build required before E2E
npm run test:e2e                    # Run tests
npm run test:e2e:ui                 # Run with interactive UI
```

E2E tests require pre-indexed data in ChromaDB. See the
[AGENTS.md](AGENTS.md) E2E section for setup instructions.

## Code Style

### Python

- **ruff** for linting and formatting (line length 90, target py313)
- **mypy** in strict mode — all functions require type annotations
- `from __future__ import annotations` in every module
- Modern type syntax: `str | None`, `list[str]`, `dict[str, Any]`
- Absolute imports only: `from embedding_cluster.settings import Settings`
- Heavy imports behind `TYPE_CHECKING` blocks where possible
- Logger per module: `logger = logging.getLogger(__name__)`

```bash
uv run ruff check embedding_cluster/ tests/       # Lint
uv run ruff check --fix embedding_cluster/ tests/  # Auto-fix
uv run ruff format embedding_cluster/ tests/       # Format
uv run mypy embedding_cluster/                     # Type check
```

### Frontend (TypeScript)

- ESLint with TypeScript and React hooks plugins
- Tailwind CSS 4 for styling

```bash
cd frontend && npm run lint
```

## Pre-commit Hooks

The project uses extensive pre-commit hooks that run automatically on
commit. Key hooks include:

- **ruff** — linting (with auto-fix) and formatting
- **commitizen** — commit message validation
- **gitleaks** — secret detection
- **yamllint** / **markdownlint** — config file linting
- **no-commit-to-branch** — prevents direct commits to master

Run all hooks manually:

```bash
uv run pre-commit run --all-files
```

## Commit Messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/)
enforced by [commitizen](https://commitizen-tools.github.io/commitizen/).

Format: `type(scope): description`

| Type | Use for |
|------|---------|
| `feat` | New features |
| `fix` | Bug fixes |
| `docs` | Documentation changes |
| `test` | Adding or updating tests |
| `refactor` | Code changes that neither fix bugs nor add features |

Examples:

```text
feat(search): add image URL search support
fix(indexer): handle empty CSV rows gracefully
docs(readme): update quick start instructions
test(server): add collection deletion tests
```

## Pull Request Process

1. Create a branch from `master` (e.g. `feat/my-feature`)
2. Make your changes and ensure all checks pass:
   ```bash
   uv run ruff check embedding_cluster/ tests/
   uv run ruff format --check embedding_cluster/ tests/
   uv run mypy embedding_cluster/
   uv run pytest --cov=embedding_cluster --cov-fail-under=90
   ```
3. Push and open a pull request against `master`
4. CI will run lint, typecheck, and test jobs automatically
5. All conversations must be resolved before merging
6. At least one approving review is required

## Project Structure

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system
design and component breakdown.

## Good First Issues

Look for issues labeled
[`good first issue`](https://github.com/aGallea/embedding-clusters/labels/good%20first%20issue)
for beginner-friendly tasks.
