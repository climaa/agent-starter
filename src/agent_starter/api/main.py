"""FastAPI surface: POST /chat (SSE stream), POST /webhook/n8n (JSON), GET /health."""

import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, cast

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agent_starter.config import get_settings
from agent_starter.graph.build import Graph
from agent_starter.graph.state import AgentState, new_turn
from agent_starter.observability import (
    langfuse_callbacks,
    request_id_var,
    setup_logging,
    thread_id_var,
)
from agent_starter.runtime import create_agent, run_turn

log = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str | None = None


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    sources: list[dict[str, Any]]
    guardrail_notes: list[str]


def to_response(state: AgentState, thread_id: str) -> ChatResponse:
    return ChatResponse(
        thread_id=thread_id,
        answer=state.answer,
        sources=[
            {"id": s.id, "source": s.source, "score": round(s.score, 3)} for s in state.sources
        ],
        guardrail_notes=state.guardrail_notes,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.log_level)
    if not hasattr(app.state, "graph"):  # tests inject their own graph
        try:
            app.state.graph = create_agent(settings)
        except FileNotFoundError as exc:
            log.error("agent not ready", extra={"error": str(exc)})
            app.state.graph = None
    yield


app = FastAPI(title="agent-starter", lifespan=lifespan)


@app.middleware("http")
async def request_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request_id_var.set(rid)
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response


def _graph(request: Request) -> Graph:
    graph: Graph | None = request.app.state.graph
    if graph is None:
        raise HTTPException(503, "index not found — run `make ingest`")
    return graph


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    return {"status": "ok", "agent_ready": request.app.state.graph is not None}


@app.post("/chat", response_class=EventSourceResponse)
async def chat(req: ChatRequest, request: Request) -> AsyncIterator[ServerSentEvent]:
    """Events: `node` (progress), `token` (answer tokens), `final` (answer + sources)."""
    graph = _graph(request)
    thread_id = req.thread_id or uuid.uuid4().hex
    thread_id_var.set(thread_id)
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": langfuse_callbacks(get_settings()),
    }
    stream = graph.astream(new_turn(req.message), config, stream_mode=["updates", "messages"])
    async for mode, chunk in stream:
        if mode == "updates":
            for node in chunk:
                yield ServerSentEvent(event="node", data={"node": node})
        elif mode == "messages":
            msg, meta = cast(tuple[BaseMessage, dict[str, Any]], chunk)
            is_answer = meta.get("langgraph_node") == "answer"
            if is_answer and isinstance(msg, AIMessage) and msg.text:
                yield ServerSentEvent(event="token", data={"text": msg.text})
    state = AgentState.model_validate((await graph.aget_state(config)).values)
    yield ServerSentEvent(event="final", data=to_response(state, thread_id).model_dump())


@app.post("/webhook/n8n")
async def n8n_webhook(
    req: ChatRequest,
    request: Request,
    x_webhook_token: Annotated[str | None, Header()] = None,
) -> ChatResponse:
    """Synchronous JSON endpoint for n8n (or any automation tool)."""
    expected = get_settings().n8n_webhook_token
    if expected and x_webhook_token != expected:
        raise HTTPException(401, "invalid webhook token")
    thread_id = req.thread_id or uuid.uuid4().hex
    thread_id_var.set(thread_id)
    state = await run_turn(
        _graph(request), req.message, thread_id, langfuse_callbacks(get_settings())
    )
    return to_response(state, thread_id)
