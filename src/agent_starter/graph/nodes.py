"""Graph nodes. Each is an async method returning a partial state update."""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from agent_starter import prompts
from agent_starter.config import Settings
from agent_starter.graph import guardrails
from agent_starter.graph.state import AgentState
from agent_starter.rag.store import Source, VectorStore

log = logging.getLogger(__name__)

REFUSAL = "Sorry, I can't help with that request."
SECRET_FALLBACK = "The answer was withheld because it appeared to contain a credential."
UNGROUNDED_NOTE = "\n\n(Note: this answer could not be fully verified against the sources.)"
HISTORY_TURNS = 6


def format_sources(sources: Sequence[Source]) -> str:
    return "\n".join(f"[S{i}] ({s.id})\n{s.text}" for i, s in enumerate(sources, start=1))


def format_tool_results(scratchpad: Sequence[BaseMessage]) -> str:
    return "\n".join(f"{m.name} -> {m.text}" for m in scratchpad if isinstance(m, ToolMessage))


class AgentNodes:
    def __init__(
        self, llm: BaseChatModel, store: VectorStore, tools: list[BaseTool], settings: Settings
    ) -> None:
        self.llm = llm
        self.store = store
        self.tools = {t.name: t for t in tools}
        self.llm_with_tools = llm.bind_tools(tools) if tools else llm
        self.settings = settings

    def _history(self, state: AgentState) -> list[BaseMessage]:
        """Previous turns only (the current question is passed explicitly)."""
        return list(state.messages[:-1][-HISTORY_TURNS:])

    # -- guard_input ---------------------------------------------------------
    async def guard_input(self, state: AgentState) -> dict[str, Any]:
        violations = guardrails.check_input(state.question, self.settings.max_input_chars)
        if not violations:
            return {}
        log.warning("input blocked", extra={"violations": violations})
        return {
            "blocked": True,
            "answer": REFUSAL,
            "guardrail_notes": violations,
            "messages": [AIMessage(content=REFUSAL)],
        }

    # -- plan ----------------------------------------------------------------
    async def plan(self, state: AgentState) -> dict[str, Any]:
        msgs = [
            SystemMessage(prompts.PLAN_SYSTEM),
            *self._history(state),
            HumanMessage(state.question),
        ]
        resp = await self.llm.ainvoke(msgs)
        return {"plan": resp.text}

    # -- retrieve ------------------------------------------------------------
    async def retrieve(self, state: AgentState) -> dict[str, Any]:
        sources = await self.store.search(state.question, k=self.settings.top_k)
        log.info("retrieved", extra={"source_ids": [s.id for s in sources]})
        return {"sources": sources}

    # -- act (tool loop) -----------------------------------------------------
    async def act(self, state: AgentState) -> dict[str, Any]:
        msgs: list[BaseMessage] = [
            SystemMessage(prompts.ACT_SYSTEM),
            HumanMessage(f"Request: {state.question}\n\nPlan:\n{state.plan}"),
            *state.scratchpad,
        ]
        resp = await self.llm_with_tools.ainvoke(msgs)
        if not isinstance(resp, AIMessage) or not resp.tool_calls:
            # Last scratchpad message becomes an AIMessage → route_after_act goes to answer.
            return {"scratchpad": [*state.scratchpad, resp]}
        results = await asyncio.gather(*(self._run_tool(c) for c in resp.tool_calls))
        return {"scratchpad": [*state.scratchpad, resp, *results], "steps": state.steps + 1}

    async def _run_tool(self, call: Any) -> ToolMessage:
        name, call_id = call["name"], call["id"]
        tool = self.tools.get(name)
        if tool is None:
            return ToolMessage(
                f"error: unknown tool {name}", tool_call_id=call_id, name=name, status="error"
            )
        try:
            out = await tool.ainvoke(call["args"])
            log.info("tool ok", extra={"tool": name})
            return ToolMessage(str(out), tool_call_id=call_id, name=name)
        except Exception as exc:  # tool failures are reported to the model, not raised
            log.warning("tool failed", extra={"tool": name, "error": str(exc)})
            return ToolMessage(f"error: {exc}", tool_call_id=call_id, name=name, status="error")

    def route_after_act(self, state: AgentState) -> str:
        last = state.scratchpad[-1] if state.scratchpad else None
        if isinstance(last, ToolMessage) and state.steps < self.settings.max_steps:
            return "act"
        return "answer"

    # -- answer --------------------------------------------------------------
    async def answer(self, state: AgentState) -> dict[str, Any]:
        notes = list(state.guardrail_notes)
        if state.steps >= self.settings.max_steps:
            notes.append("max_steps_reached")
        prompt = prompts.answer_user_prompt(
            state.question, format_sources(state.sources), format_tool_results(state.scratchpad)
        )
        msgs = [SystemMessage(prompts.ANSWER_SYSTEM), *self._history(state), HumanMessage(prompt)]
        resp = await self.llm.ainvoke(msgs)
        return {"answer": resp.text, "guardrail_notes": notes}

    # -- guard_output --------------------------------------------------------
    async def guard_output(self, state: AgentState) -> dict[str, Any]:
        violations = guardrails.check_output(state.answer, len(state.sources))
        answer = state.answer
        if "secret_in_output" in violations:
            answer = SECRET_FALLBACK
        elif violations:
            answer += UNGROUNDED_NOTE
        if violations:
            log.warning("output guardrail", extra={"violations": violations})
        return {
            "answer": answer,
            "guardrail_notes": [*state.guardrail_notes, *violations],
            "messages": [AIMessage(content=answer)],
        }
