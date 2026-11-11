"""Assemble the agent from settings (shared by API, CLI and evals)."""

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

from agent_starter.config import Settings
from agent_starter.graph.build import Graph, build_graph
from agent_starter.graph.state import AgentState, new_turn
from agent_starter.providers import get_chat_model, get_embeddings
from agent_starter.rag.store import VectorStore
from agent_starter.tools import get_tools


def create_agent(settings: Settings) -> Graph:
    store = VectorStore.load(settings.index_dir, get_embeddings(settings))
    return build_graph(get_chat_model(settings), store, get_tools(), settings)


async def run_turn(
    graph: Graph, question: str, thread_id: str, callbacks: list[BaseCallbackHandler] | None = None
) -> AgentState:
    """Run one user turn to completion and return the final state."""
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": callbacks or [],
    }
    await graph.ainvoke(new_turn(question), config)
    snapshot = await graph.aget_state(config)
    return AgentState.model_validate(snapshot.values)
