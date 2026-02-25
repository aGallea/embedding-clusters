# Real-time Indexing Progress Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add on_log callbacks, heartbeat, and completion broadcasts for real-time indexing progress while preserving existing behavior.

**Architecture:** Extend the indexer to emit optional log callbacks and more frequent progress updates without changing existing data structures. Wire the server route to broadcast log and heartbeat messages over WebSocket, and send a completion payload with collection names and final stats. Maintain synchronous callback signatures and fire-and-forget broadcasts via asyncio tasks.

**Tech Stack:** Python 3.13, FastAPI, asyncio, mypy (strict), ruff

---

### Task 1: Extend indexer callbacks and logging

**Files:**
- Modify: `embedding_cluster/indexer.py:33-143`

**Step 1: Write the failing test**

```python
def test_main_indexer_calls_on_log_and_progress_callbacks():
    # TODO: add test when allowed by scope
    assert True
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_indexer.py::test_main_indexer_calls_on_log_and_progress_callbacks -v`
Expected: FAIL with missing test or behavior

**Step 3: Write minimal implementation**

```python
PROGRESS_UPDATE_INTERVAL = 10

async def main_indexer(
    settings: Settings,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    on_log: Callable[[str, str, str], None] | None = None,
    cancel_event: asyncio.Event | None = None,
) -> None:
    # wrap model loading with try/except and call on_log for failures
    # emit on_log at requested milestones
    # emit on_progress every PROGRESS_UPDATE_INTERVAL rows
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_indexer.py::test_main_indexer_calls_on_log_and_progress_callbacks -v`
Expected: PASS

**Step 5: Commit**

```bash
git add embedding_cluster/indexer.py
git commit -m "feat(indexer): add log callbacks and progress updates"
```

### Task 2: Wire WebSocket logging, heartbeat, and completion message

**Files:**
- Modify: `embedding_cluster/server/routes/index.py:1-128`

**Step 1: Write the failing test**

```python
def test_index_routes_broadcast_log_heartbeat_completed():
    # TODO: add test when allowed by scope
    assert True
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_main.py::test_index_routes_broadcast_log_heartbeat_completed -v`
Expected: FAIL with missing test or behavior

**Step 3: Write minimal implementation**

```python
import time

def _get_collection_names(settings: Settings) -> list[str]:
    names: list[str] = []
    prefix = settings.chromadb_collection_prefix
    if settings.image_embedding_fields:
        for field in settings.image_embedding_fields:
            names.append(f"{prefix}{field}")
    if settings.text_embedding_fields:
        for field in settings.text_embedding_fields:
            names.append(f"{prefix}{field}")
    return names

def on_log(message: str, level: str, verbosity: str) -> None:
    asyncio.create_task(
        ws_manager.broadcast(
            task_state.job_id,
            {
                "type": "log",
                "level": level,
                "message": message,
                "verbosity": verbosity,
            },
        )
    )

async def heartbeat() -> None:
    while task_state.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        await ws_manager.broadcast(
            task_state.job_id,
            {"type": "heartbeat", "elapsed_seconds": time.perf_counter() - start_time},
        )
        await asyncio.sleep(3)

start_time = time.perf_counter()
heartbeat_task = asyncio.create_task(heartbeat())

await main_indexer(
    settings,
    on_progress=on_progress,
    on_log=on_log,
    cancel_event=task_state.cancel_event,
)

heartbeat_task.cancel()
try:
    await heartbeat_task
except asyncio.CancelledError:
    pass

task_state.status = TaskStatus.COMPLETED
final_progress = task_state.progress
asyncio.create_task(
    ws_manager.broadcast(
        task_state.job_id,
        {
            "type": "completed",
            "status": "completed",
            "progress": final_progress,
            "total_indexed": final_progress.get("rows_indexed", 0),
            "collection_names": _get_collection_names(settings),
        },
    )
)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_main.py::test_index_routes_broadcast_log_heartbeat_completed -v`
Expected: PASS

**Step 5: Commit**

```bash
git add embedding_cluster/server/routes/index.py
git commit -m "feat(server): add websocket logs and heartbeat"
```

### Task 3: Lint, format, and type-check

**Files:**
- Modify: `embedding_cluster/indexer.py`
- Modify: `embedding_cluster/server/routes/index.py`

**Step 1: Run ruff check**

Run: `uv run ruff check embedding_cluster/indexer.py embedding_cluster/server/routes/index.py`
Expected: PASS

**Step 2: Run ruff format**

Run: `uv run ruff format embedding_cluster/indexer.py embedding_cluster/server/routes/index.py`
Expected: PASS

**Step 3: Run mypy**

Run: `uv run mypy embedding_cluster/indexer.py embedding_cluster/server/routes/index.py`
Expected: PASS

**Step 4: Commit**

```bash
git add embedding_cluster/indexer.py embedding_cluster/server/routes/index.py
git commit -m "chore: lint and typecheck indexing changes"
```
