"""Graph wiring — the whole agent in one picture:

START → guard_input ─blocked→ END
             └→ plan → retrieve → act ⟲ (tool calls, ≤ MAX_STEPS) → answer → guard_output → END
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_starter.config import Settings
from agent_starter.graph.nodes import AgentNodes
from agent_starter.graph.state import AgentState
from agent_starter.rag.store import VectorStore

Graph = CompiledStateGraph[AgentState, None, Any, Any]


def default_checkpointer() -> InMemorySaver:
    """In-memory, per-process conversation memory keyed by thread_id.

    Swap for a persistent saver (Postgres/DynamoDB) in production. Our own Pydantic
    types stored in state are allow-listed for the checkpoint serializer.
    """
    serde = JsonPlusSerializer(allowed_msgpack_modules=[("agent_starter.rag.store", "Source")])
    return InMemorySaver(serde=serde)


def build_graph(
    llm: BaseChatModel,
    store: VectorStore,
    tools: list[BaseTool],
    settings: Settings,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Graph:
    n = AgentNodes(llm, store, tools, settings)
    g = StateGraph(AgentState)
    g.add_node("guard_input", n.guard_input)
    g.add_node("plan", n.plan)
    g.add_node("retrieve", n.retrieve)
    g.add_node("act", n.act)
    g.add_node("answer", n.answer)
    g.add_node("guard_output", n.guard_output)

    g.add_edge(START, "guard_input")
    g.add_conditional_edges(
        "guard_input", lambda s: END if s.blocked else "plan", {END: END, "plan": "plan"}
    )
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "act")
    g.add_conditional_edges("act", n.route_after_act, {"act": "act", "answer": "answer"})
    g.add_edge("answer", "guard_output")
    g.add_edge("guard_output", END)
    return g.compile(checkpointer=checkpointer or default_checkpointer())
