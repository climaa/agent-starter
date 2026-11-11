"""Example tool: GET JSON from a configured API (base URL is fixed → no SSRF to arbitrary hosts)."""

import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent_starter.config import get_settings
from agent_starter.tools import register
from agent_starter.tools._http import request_with_retry

MAX_CHARS = 2000


class HttpGetArgs(BaseModel):
    path: str = Field(description="Path on the configured API, starting with '/', e.g. '/zen'")


async def http_get_json(path: str) -> str:
    if not path.startswith("/") or ".." in path:
        return "error: path must start with '/' and must not contain '..'"
    s = get_settings()
    resp = await request_with_retry(
        "GET",
        s.http_api_base_url.rstrip("/") + path,
        timeout_s=s.http_timeout_s,
        retries=s.http_retries,
    )
    try:
        body = json.dumps(resp.json())
    except ValueError:
        body = resp.text
    return f"status={resp.status_code} body={body[:MAX_CHARS]}"


register(
    StructuredTool.from_function(
        coroutine=http_get_json,
        name="http_get_json",
        description="Fetch live JSON data from the company API. Use for live data, not policy.",
        args_schema=HttpGetArgs,
    )
)
