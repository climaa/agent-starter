"""Pure guardrail checks (no LLM) — cheap, deterministic, unit-testable.

In production, layer Amazon Bedrock Guardrails (PII, denied topics, contextual
grounding) on top; these checks stay as the first, free line of defence.
"""

import re

INJECTION_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"ignore (all |any )?(previous|prior|above) instructions",
        r"disregard (the )?(system|previous) prompt",
        r"reveal (your|the) (system )?prompt",
        r"you are now (dan|in developer mode)",
    )
]
SECRET_PATTERNS = [
    re.compile(p)
    for p in (
        r"AKIA[0-9A-Z]{16}",  # AWS access key id
        r"sk-[A-Za-z0-9_-]{20,}",  # OpenAI/Anthropic-style keys
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    )
]
CITATION = re.compile(r"\[(S\d+)\]")


def check_input(text: str, max_chars: int) -> list[str]:
    violations: list[str] = []
    if not text.strip():
        violations.append("empty_input")
    if len(text) > max_chars:
        violations.append("input_too_long")
    if any(p.search(text) for p in INJECTION_PATTERNS):
        violations.append("prompt_injection")
    if any(p.search(text) for p in SECRET_PATTERNS):
        violations.append("secret_in_input")
    return violations


def check_output(answer: str, n_sources: int) -> list[str]:
    violations: list[str] = []
    if any(p.search(answer) for p in SECRET_PATTERNS):
        violations.append("secret_in_output")
    cited = set(CITATION.findall(answer))
    valid = {f"S{i}" for i in range(1, n_sources + 1)}
    if cited - valid:
        violations.append("unknown_citation")
    if n_sources and not cited and "don't know" not in answer.lower():
        violations.append("missing_citation")
    return violations
