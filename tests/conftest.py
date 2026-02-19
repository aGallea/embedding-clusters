from __future__ import annotations

import pytest

from embedding_cluster.utils import Singleton


@pytest.fixture(autouse=True)
def _clear_singletons() -> None:
    """Clear Singleton instances between tests to avoid cross-contamination."""
    Singleton._instances.clear()
