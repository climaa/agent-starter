"""Shared async HTTP helper with timeout + bounded retry (explicit, no extra deps)."""

import asyncio
import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


async def request_with_retry(
    method: str,
    url: str,
    *,
    json: Any = None,
    timeout_s: float = 10.0,
    retries: int = 2,
    backoff_s: float = 0.5,
) -> httpx.Response:
    last_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        for attempt in range(retries + 1):
            try:
                resp = await client.request(method, url, json=json)
                if resp.status_code not in RETRYABLE_STATUS:
                    return resp
                last_exc = httpx.HTTPStatusError(
                    f"retryable status {resp.status_code}", request=resp.request, response=resp
                )
            except httpx.TransportError as exc:  # timeouts, connection errors
                last_exc = exc
            if attempt < retries:
                log.warning("http retry", extra={"url": url, "attempt": attempt + 1})
                await asyncio.sleep(backoff_s * 2**attempt)
    assert last_exc is not None
    raise last_exc
