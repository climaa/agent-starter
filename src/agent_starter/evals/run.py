"""`make eval`: retrieval hit-rate + answer faithfulness over eval/evalset.yaml.

Metrics (deterministic, no LLM judge — works offline):
- hit@k        expected source file is among the top-k retrieved chunks
- faithfulness share of answer content words found in the retrieved sources / tool output
               (lexical grounding; swap in an LLM-as-judge for production)
- correct      every `must_include` keyword appears in the answer
"""

import argparse
import asyncio
import re
import sys
import uuid
from pathlib import Path

import yaml
from pydantic import BaseModel

from agent_starter.config import get_settings
from agent_starter.graph.nodes import format_tool_results
from agent_starter.graph.state import AgentState
from agent_starter.observability import setup_logging
from agent_starter.providers import tokenize
from agent_starter.runtime import create_agent, run_turn


class EvalCase(BaseModel):
    id: str
    question: str
    expected_source: str
    must_include: list[str] = []


class EvalResult(BaseModel):
    id: str
    hit: bool
    faithfulness: float
    correct: bool


def faithfulness(answer: str, context: str) -> float:
    answer_tokens = tokenize(re.sub(r"\[S\d+\]", "", answer))
    if not answer_tokens:
        return 0.0
    ctx = set(tokenize(context))
    return sum(t in ctx for t in answer_tokens) / len(answer_tokens)


def score(case: EvalCase, state: AgentState) -> EvalResult:
    context = "\n".join(s.text for s in state.sources) + format_tool_results(state.scratchpad)
    return EvalResult(
        id=case.id,
        hit=any(s.source == case.expected_source for s in state.sources),
        faithfulness=faithfulness(state.answer, context),
        correct=all(k.lower() in state.answer.lower() for k in case.must_include),
    )


def print_table(results: list[EvalResult]) -> None:
    print(f"{'case':<20} {'hit@k':>6} {'faithful':>9} {'correct':>8}")
    print("-" * 46)
    for r in results:
        print(
            f"{r.id:<20} {'✓' if r.hit else '✗':>6} {r.faithfulness:>9.2f} "
            f"{'✓' if r.correct else '✗':>8}"
        )
    n = len(results)
    print("-" * 46)
    print(
        f"{'TOTAL':<20} {sum(r.hit for r in results) / n:>6.0%} "
        f"{sum(r.faithfulness for r in results) / n:>9.2f} "
        f"{sum(r.correct for r in results) / n:>8.0%}"
    )


async def run(cases: list[EvalCase]) -> list[EvalResult]:
    graph = create_agent(get_settings())
    results = []
    for case in cases:
        state = await run_turn(graph, case.question, thread_id=f"eval-{uuid.uuid4().hex[:6]}")
        results.append(score(case, state))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evalset", type=Path, default=Path("eval/evalset.yaml"))
    parser.add_argument("--min-hit-rate", type=float, default=0.0)
    args = parser.parse_args()
    setup_logging("WARNING")
    cases = [EvalCase.model_validate(c) for c in yaml.safe_load(args.evalset.read_text())]
    results = asyncio.run(run(cases))
    print_table(results)
    hit_rate = sum(r.hit for r in results) / len(results)
    if hit_rate < args.min_hit_rate:
        sys.exit(f"hit-rate {hit_rate:.0%} below threshold {args.min_hit_rate:.0%}")


if __name__ == "__main__":
    main()
