from __future__ import annotations

import logging

import litellm

logger = logging.getLogger(__name__)

# Alias for testability (easy to mock)
litellm_completion = litellm.completion

SYSTEM_PROMPT_TOP_LEVEL = (
    "Your role is to find a very short (max 5 words), concise name "
    "for a group of items, one name to rule them all. "
    "The user will provide a list of item names. Do your best."
)

SYSTEM_PROMPT_SUB_CLUSTER = (
    "Your role is to find a very short (max 5 words), concise name "
    "for a sub-group of items within a larger group called "
    '"{parent_name}". '
    "The name should distinguish this sub-group from its siblings "
    "while relating to the parent theme. The user will provide a "
    "list of item names. Do your best."
)


def _normalize_base_url(model: str, base_url: str | None) -> str | None:
    """Strip /v1 suffix for Ollama models (litellm uses native API)."""
    if base_url and model.startswith("ollama/"):
        stripped = base_url.rstrip("/")
        if stripped.endswith("/v1"):
            return stripped[:-3]
    return base_url


def _call_llm(
    messages: list[dict[str, str]],
    api_key: str,
    model: str,
    base_url: str | None = None,
    temperature: float = 0.5,
) -> str:
    """Call LiteLLM and return the response content."""
    kwargs: dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if api_key:
        kwargs["api_key"] = api_key
    resolved_url = _normalize_base_url(model, base_url)
    if resolved_url:
        kwargs["api_base"] = resolved_url

    response = litellm_completion(**kwargs)
    content: str = response.choices[0].message.content or ""
    return (content[:30] + "..") if len(content) > 30 else content


def get_cluster_name(
    item_names: list[str],
    api_key: str,
    model: str,
    base_url: str | None = None,
    temperature: float = 0.5,
) -> str:
    """Generate a short name for a cluster of items."""
    user_content = "\n".join(f"name: {name}" for name in item_names)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TOP_LEVEL},
        {"role": "user", "content": user_content},
    ]
    return _call_llm(messages, api_key, model, base_url, temperature)


def get_sub_cluster_name(
    item_names: list[str],
    api_key: str,
    model: str,
    base_url: str | None = None,
    temperature: float = 0.5,
    parent_cluster_name: str | None = None,
) -> str:
    """Generate a short name for a sub-cluster."""
    if parent_cluster_name:
        system_content = SYSTEM_PROMPT_SUB_CLUSTER.format(
            parent_name=parent_cluster_name,
        )
    else:
        system_content = SYSTEM_PROMPT_TOP_LEVEL

    user_content = "\n".join(f"name: {name}" for name in item_names)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return _call_llm(messages, api_key, model, base_url, temperature)


def test_connection(
    api_key: str,
    model: str,
    base_url: str | None = None,
) -> tuple[bool, str | None]:
    """Test LLM connection. Returns (success, error)."""
    try:
        kwargs: dict[str, object] = {
            "model": model,
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 5,
        }
        if api_key:
            kwargs["api_key"] = api_key
        resolved_url = _normalize_base_url(model, base_url)
        if resolved_url:
            kwargs["api_base"] = resolved_url
        litellm_completion(**kwargs)
        return True, None
    except Exception as exc:
        error_msg = str(exc)
        if api_key and api_key in error_msg:
            error_msg = error_msg.replace(api_key, "***")
        return False, error_msg
