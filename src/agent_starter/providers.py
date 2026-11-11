"""Model factories driven by env (LLM_PROVIDER / EMBEDDINGS_PROVIDER).

Real providers are imported lazily so the offline path (tests, CI, demo without
credentials) never needs AWS / Anthropic / OpenAI packages or network.
"""

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field

from agent_starter import prompts
from agent_starter.config import Settings

DEFAULT_MODELS = {
    # Bedrock: use a cross-region inference profile id if your region requires one.
    "bedrock": "anthropic.claude-opus-5-5",
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-4.1-mini",
}
DEFAULT_EMBEDDINGS = {"bedrock": "amazon.titan-embed-text-v2:0", "openai": "text-embedding-3-small"}


# --------------------------------------------------------------------------- chat
def get_chat_model(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider
    model = settings.llm_model or DEFAULT_MODELS.get(provider, "")
    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse

        return ChatBedrockConverse(
            model_id=model, region_name=settings.aws_region, max_tokens=settings.llm_max_tokens
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, max_tokens=settings.llm_max_tokens)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, max_tokens=settings.llm_max_tokens)
    return OfflineChatModel()


# ---------------------------------------------------------------------- embeddings
def get_embeddings(settings: Settings) -> Embeddings:
    provider = settings.embeddings_provider
    model = settings.embeddings_model or DEFAULT_EMBEDDINGS.get(provider, "")
    if provider == "bedrock":
        from langchain_aws import BedrockEmbeddings

        return BedrockEmbeddings(model_id=model, region_name=settings.aws_region)
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model)
    return HashingEmbeddings()


# ------------------------------------------------------------ offline fallbacks
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS_TEXT = """a an the and or of to in on for is are was were be by with what which how do
does can i we you it this that at as from our your my me if any"""
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embeddings: offline, no model download.

    Lexical only (no semantics) — good enough for tests, CI and a no-credentials demo.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OfflineChatModel(BaseChatModel):
    """Rule-based stand-in for an LLM so the whole graph runs offline.

    It recognises which node is calling by the system prompt and returns a
    deterministic, *extractive* response (it only copies sentences from the
    sources), so evals and tests are reproducible. Not meant to be smart.
    """

    bound_tool_names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "offline-fake"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        names = [t.name for t in tools if isinstance(t, BaseTool)]
        return self.model_copy(update={"bound_tool_names": names})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._respond(messages))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop)

    # -- behaviour per node -----------------------------------------------------
    def _respond(self, messages: list[BaseMessage]) -> AIMessage:
        system = next((m.text for m in messages if isinstance(m, SystemMessage)), "")
        human = [m.text for m in messages if isinstance(m, HumanMessage)]
        last_human = human[-1] if human else ""
        if system == prompts.PLAN_SYSTEM:
            return AIMessage(
                content="1. Search the knowledge base. 2. Call tools if an action "
                "is requested. 3. Answer with citations."
            )
        if system == prompts.ACT_SYSTEM:
            return self._act(messages, last_human)
        if system == prompts.ANSWER_SYSTEM:
            return AIMessage(content=self._extractive_answer(last_human))
        return AIMessage(content="OK")

    def _act(self, messages: list[BaseMessage], question: str) -> AIMessage:
        already_called = any(isinstance(m, ToolMessage) for m in messages)
        if (
            "create_ticket" in self.bound_tool_names
            and "ticket" in question.lower()
            and (not already_called)
        ):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_ticket",
                        "args": {
                            "title": question[:80],
                            "description": question,
                            "priority": "medium",
                        },
                        "id": "call_offline_1",
                    }
                ],
            )
        return AIMessage(content="NO_TOOLS")

    @staticmethod
    def _extractive_answer(prompt: str) -> str:
        question = prompt.split("\n", 1)[0].removeprefix("Question: ")
        q_tokens = set(tokenize(question))
        best: tuple[int, str, str] = (0, "", "")
        for sid, body in re.findall(
            r"\[(S\d+)\][^\n]*\n(.*?)(?=\n\[S\d+\]|\n\nTool results:|\Z)", prompt, flags=re.S
        ):
            text = " ".join(re.sub(r"^#.*$", "", body, flags=re.M).split())  # drop headings
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                score = len(q_tokens & set(tokenize(sentence)))
                if score > best[0]:
                    best = (score, sentence.strip(), sid)
        tools = prompt.split("Tool results:\n", 1)[-1].strip()
        tool_note = f" Action taken: {tools}" if tools and tools != "(none)" else ""
        if not best[0]:
            return "I don't know based on the available sources." + tool_note
        return f"{best[1]} [{best[2]}]{tool_note}"
