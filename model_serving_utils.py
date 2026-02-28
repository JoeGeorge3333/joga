"""Model serving utilities for querying Databricks chat endpoints."""

import os
from typing import Any

from mlflow.deployments import get_deploy_client


def _get_endpoint_task_type(endpoint_name: str) -> str:
    """Get the task type of a serving endpoint."""
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    ep = w.serving_endpoints.get(endpoint_name)
    return ep.task


def is_endpoint_supported(endpoint_name: str) -> bool:
    """Check if the endpoint has a supported task type."""
    if not endpoint_name:
        return False
    try:
        task_type = _get_endpoint_task_type(endpoint_name)
        supported = ["agent/v1/chat", "agent/v2/chat", "llm/v1/chat"]
        return task_type in supported
    except Exception:
        return False


def query_chat_endpoint(
    endpoint_name: str,
    messages: list[dict[str, str]],
    max_tokens: int = 1024,
) -> str:
    """
    Query a chat-completions or agent serving endpoint.
    Returns the text content of the assistant's response.
    """
    if not endpoint_name:
        raise ValueError("SERVING_ENDPOINT must be set")
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
    client = get_deploy_client(mlflow_tracking_uri)

    res = client.predict(
        endpoint=endpoint_name,
        inputs={"messages": messages, "max_tokens": max_tokens},
    )

    if "messages" in res:
        msgs = res["messages"]
        last = msgs[-1] if msgs else {}
        return last.get("content", "") or ""
    if "choices" in res:
        choice = res["choices"][0]["message"]
        content = choice.get("content")
        if isinstance(content, list):
            return "".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )
        return content or ""

    raise ValueError("Unexpected endpoint response format")
