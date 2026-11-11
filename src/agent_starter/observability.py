"""Structured JSON logging (with request/thread ids) and optional Langfuse tracing."""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from agent_starter.config import Settings

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
thread_id_var: ContextVar[str] = ContextVar("thread_id", default="-")

_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
            "thread_id": thread_id_var.get(),
        }
        # Anything passed via `extra={...}` is included as-is.
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "botocore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def langfuse_callbacks(settings: Settings) -> list[BaseCallbackHandler]:
    """Return a Langfuse callback only if keys are configured and the extra is installed."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return []
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
    except ImportError:
        logging.getLogger(__name__).warning(
            "langfuse keys set but package missing: uv sync --extra observability"
        )
        return []
    Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    handler: BaseCallbackHandler = CallbackHandler()
    return [handler]
