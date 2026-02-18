# Revive Project Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Modernize embedding-clusters: migrate to uv, upgrade all deps, fix bugs, add strict typing, tests, CI, and update pre-commit.

**Architecture:** Incremental modernization - each task is independently committable. Start with build tooling (uv), then fix bugs, add quality tooling (ruff + mypy strict), add tests, add CI, update pre-commit.

**Tech Stack:** Python 3.13, uv, ruff, mypy (strict), pytest + pytest-asyncio + pytest-cov, GitHub Actions

---

### Task 1: Migrate to uv + Update Dependencies

**Files:**
- Delete: `requirements.txt`
- Delete: `mypy.ini`
- Modify: `pyproject.toml` (rewrite entirely)
- Create: `.python-version`

**Step 1: Create `.python-version`**

```
3.13
```

**Step 2: Rewrite `pyproject.toml`**

Replace the entire `pyproject.toml` with a proper uv project config:

```toml
[project]
name = "embedding-cluster"
version = "0.1.0"
description = "Generate, index, and visualize embedding clusters from CSV data"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.13"
dependencies = [
    "pydantic>=2.12,<3",
    "pydantic-settings>=2.13,<3",
    "torch>=2.10,<3",
    "chromadb>=0.6,<1",
    "transformers>=4.57,<5",
    "sentence-transformers>=3.4,<4",
    "dash>=3.4,<4",
    "plotly>=6.5,<7",
    "aiohttp>=3.13,<4",
    "openai>=1.109,<2",
    "scikit-learn>=1.8,<2",
    "numpy>=2.2,<3",
    "Pillow>=11,<12",
]

[project.optional-dependencies]
dev = [
    "pytest>=8,<9",
    "pytest-asyncio>=0.25,<1",
    "pytest-cov>=6,<7",
    "mypy>=1.14,<2",
    "ruff>=0.9,<1",
    "pre-commit>=4,<5",
]

[tool.ruff]
target-version = "py313"
line-length = 90

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "SIM",  # flake8-simplify
    "TCH",  # flake8-type-checking
    "RUF",  # ruff-specific
]

[tool.ruff.lint.isort]
known-first-party = ["embedding_cluster"]

[tool.mypy]
python_version = "3.13"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_generics = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true

[[tool.mypy.overrides]]
module = [
    "chromadb.*",
    "transformers.*",
    "sentence_transformers.*",
    "torch.*",
    "dash.*",
    "plotly.*",
    "sklearn.*",
    "PIL.*",
    "openai.*",
    "aiohttp.*",
]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.black]
line-length = 90
target-version = ['py313']

[tool.isort]
profile = "black"
known_first_party = ["embedding_cluster"]
```

**Step 3: Delete old files**

```bash
rm requirements.txt mypy.ini
```

**Step 4: Initialize uv and install**

```bash
uv sync
```

This creates `uv.lock` and installs all deps including dev extras.

**Step 5: Verify installation**

```bash
uv run python -c "import embedding_cluster; print('OK')"
```

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: migrate to uv, upgrade all dependencies to latest"
```

---

### Task 2: Fix All Bugs

**Files:**
- Modify: `embedding_cluster/utils.py`
- Modify: `embedding_cluster/indexer.py`
- Modify: `embedding_cluster/scatter_plot.py`
- Modify: `embedding_cluster/__main__.py`

**Bug fixes to apply (in order):**

**2a. `utils.py` - Fix `logger.warn()` -> `logger.warning()`**

Line 123: `logger.warn(log)` -> `logger.warning(log)`

**2b. `indexer.py` - Fix `logger.warn()` -> `logger.warning()`**

Lines 237, 249: Replace all `logger.warn(` with `logger.warning(`

**2c. `indexer.py` - Fix resource leak (CSV file never closed)**

Line 35: Wrap in context manager. Replace:
```python
csv_file = open(settings.local_csv_filename)
csv_iter = csv.DictReader(csv_file)
```
With pattern using `with` statement around the entire CSV processing block.

**2d. `indexer.py` - Fix `async_wrapper_build_and_encode` swallowing exceptions**

Lines 212-213: The bare `except Exception` returns `None`, which crashes `handle()` when unpacking. Fix by returning `None` explicitly and filtering in `handle()`:

In `async_wrapper_build_and_encode`, make it return `None` on error (already does implicitly).
In `handle()`, filter out None results:
```python
docs = await asyncio.gather(*tasks)
docs = [doc for doc in docs if doc is not None]
```

**2e. `indexer.py` - Fix missing None check for text embeddings**

Line 152: Add None guard like the image path has:
```python
curr_embedding = (
    embeddings.get(embedding_field_name).tolist()
    if embeddings.get(embedding_field_name) is not None
    else []
)
```

**2f. `indexer.py` - Fix type mismatch in `async_wrapper_build_and_encode`**

Lines 190, 192: Change `list[str]` to `Optional[list[str]]` for `image_embedding_fields` and `text_embedding_fields`.

**2g. `utils.py` - Fix `ImageDownloader` creating session outside async context**

Lines 66-67: Defer session creation to first use inside async context. Change `__init__` to not create session, add lazy init in `download_image_exp_backoff`.

**2h. `utils.py` - Fix unbound variables in error handler**

Lines 101-116: Initialize `status` and `reason` before the if-chain:
```python
status = 500
reason = "Unknown error"
```

**2i. `scatter_plot.py` - Fix potential None from GPT response**

Line 47: Guard against None content:
```python
content = completion.choices[0].message.content or ""
```

**2j. `indexer.py` - Remove f-string prefix on static strings**

Lines 325, 339: `f"failed to encode image"` -> `"failed to encode image"` (same for text).

**2k. `indexer.py` / `scatter_plot.py` - Remove duplicate Settings() instantiation**

Pass settings as parameter to `main_indexer(settings)` and `main_scatter_plot(settings)` from `__main__.py`.

**Step: Commit**

```bash
git add -A
git commit -m "fix: resolve resource leaks, deprecated APIs, error handling, and type issues"
```

---

### Task 3: Add ruff + Strict Typing

**Files:**
- Modify: `embedding_cluster/utils.py` (modernize type hints)
- Modify: `embedding_cluster/indexer.py` (modernize type hints)
- Modify: `embedding_cluster/scatter_plot.py` (modernize type hints)
- Modify: `embedding_cluster/settings.py` (modernize type hints)
- Modify: `embedding_cluster/__main__.py`
- Create: `embedding_cluster/py.typed`

**Step 1: Create `py.typed` marker**

Empty file at `embedding_cluster/py.typed`.

**Step 2: Modernize type hints across all files**

Replace all:
- `typing.List[X]` -> `list[X]`
- `typing.Dict[X, Y]` -> `dict[X, Y]`
- `typing.Optional[X]` -> `X | None`
- `typing.Union[X, Y]` -> `X | Y`
- `typing.Sequence[X]` -> `collections.abc.Sequence[X]`
- `typing.Mapping[X, Y]` -> `collections.abc.Mapping[X, Y]`
- `typing.Any` -> keep (still valid)
- Remove unused `typing` imports

**Step 3: Run ruff format and fix**

```bash
uv run ruff format embedding_cluster/
uv run ruff check --fix embedding_cluster/
```

**Step 4: Run mypy**

```bash
uv run mypy embedding_cluster/
```

Fix any remaining type errors. Add `# type: ignore[import-untyped]` only for third-party imports that lack stubs.

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add strict mypy typing, modernize to Python 3.13 type syntax, add ruff"
```

---

### Task 4: Add Tests

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_settings.py`
- Create: `tests/test_utils.py`
- Create: `tests/test_indexer.py`
- Create: `tests/test_scatter_plot.py`

**Step 1: Create test infrastructure**

`tests/__init__.py` - empty file.

`tests/conftest.py` - shared fixtures:
- `mock_settings()` - returns Settings with test defaults
- `mock_chromadb_client()` - mocked ChromaDB client
- `mock_image_model()` - mocked CLIPModel
- `mock_text_model()` - mocked SentenceTransformer

**Step 2: Write `tests/test_settings.py`**

Test cases:
- Default values are correct
- Environment variable overrides work
- List fields parse JSON strings correctly
- Invalid running_mode values
- Optional fields default to None

**Step 3: Write `tests/test_utils.py`**

Test cases:
- `id_generator` produces correct length and character set
- `Formatter._get_level_color` returns correct ANSI codes
- `ChromaDocsCollection` model validation
- `init_logger` sets up handlers correctly
- `get_or_create_chromadb_collections` creates correct collections
- `init_chroma_docs_collection` initializes empty collections
- `ImageDownloader.download_image_exp_backoff` with mocked aiohttp:
  - successful download
  - retry on 429
  - give up on 404
  - handle None URL

**Step 4: Write `tests/test_indexer.py`**

Test cases:
- `generate_embedding_field_name` returns correct format
- `encode_text` with mocked SentenceTransformer
- `encode_image` with mocked CLIPModel and processor
- `build_and_encode` with mocked models (image + text)
- `build_and_encode` handles missing image URL gracefully
- `handle` filters out None docs from failed encodings
- `main_indexer` with mocked CSV and models (integration-ish)

**Step 5: Write `tests/test_scatter_plot.py`**

Test cases:
- `get_field_as_list` extracts field correctly
- `create_collection_text_display` joins fields with separator
- `generate_cluster_props` creates correct cluster indices and names
- `gpt_get_cluster_name` with mocked OpenAI client
- `load_chromadb_collection` with mocked chromadb
- `display_hover` callback with mock hover data and None

**Step 6: Run tests with coverage**

```bash
uv run pytest tests/ -v --cov=embedding_cluster --cov-report=term-missing
```

Target: 70%+ coverage.

**Step 7: Commit**

```bash
git add -A
git commit -m "test: add comprehensive unit test suite with 70%+ coverage"
```

---

### Task 5: Add GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Create CI workflow**

```yaml
name: CI

on:
  push:
    branches: ["*"]
  pull_request:
    branches: [master]

permissions:
  contents: read

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen
      - run: uv run ruff check embedding_cluster/
      - run: uv run ruff format --check embedding_cluster/

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen
      - run: uv run mypy embedding_cluster/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen
      - run: uv run pytest tests/ -v --cov=embedding_cluster --cov-report=term-missing --cov-fail-under=70
```

**Step 2: Commit**

```bash
git add -A
git commit -m "ci: add GitHub Actions workflow for lint, typecheck, and test"
```

---

### Task 6: Update Pre-commit Config

**Files:**
- Modify: `.pre-commit-config.yaml`

**Step 1: Update hook versions and add ruff**

- Replace `black` and `isort` hooks with `ruff` (format + check)
- Update all hook `rev` values to latest
- Add mypy hook (local, runs via uv)
- Keep: commitizen, yamllint, markdownlint, gitleaks, shellcheck, pre-commit-hooks, check-jsonschema
- Remove: `black` hook, `isort` hook (replaced by ruff)

**Step 2: Test hooks**

```bash
uv run pre-commit run --all-files
```

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: modernize pre-commit hooks, replace black+isort with ruff"
```

---

### Task 7: Update AGENTS.md

**Files:**
- Modify: `AGENTS.md`

Update to reflect new tooling:
- uv instead of pip/venv
- ruff instead of black+isort
- pytest test commands
- CI workflow
- Python 3.13 target

**Step: Commit**

```bash
git add -A
git commit -m "docs: update AGENTS.md for new tooling and workflows"
```
