"""Typed graph state. `messages` is the conversation (persisted per thread_id by the
checkpointer); every other field is per-turn scratch space reset by `new_turn()`."""

from typing import Annotated, Any

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from agent_starter.rag.store import Source


class AgentState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    question: str = ""
    plan: str = ""
    sources: list[Source] = Field(default_factory=list)
    scratchpad: list[AnyMessage] = Field(default_factory=list)  # act-loop AI/tool messages
    steps: int = 0
    blocked: bool = False
    answer: str = ""
    guardrail_notes: list[str] = Field(default_factory=list)


def new_turn(question: str) -> dict[str, Any]:
    """Graph input for one user turn: appends the question, resets per-turn fields."""
    return {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "plan": "",
        "sources": [],
        "scratchpad": [],
        "steps": 0,
        "blocked": False,
        "answer": "",
        "guardrail_notes": [],
    }
