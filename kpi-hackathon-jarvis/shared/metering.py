"""Token usage hook. Default is no-op; the tournament orchestrator can
override at runtime to capture per-team-per-round usage.

Agents call `record_llm_usage(input_tokens, output_tokens)` after each LLM
call. With no recorder set, it's a no-op (zero cost). Template code does
not depend on logfire or any external library.
"""
from __future__ import annotations

from typing import Callable


def _noop(input_tokens: int, output_tokens: int) -> None:
    pass


_recorder: Callable[[int, int], None] = _noop


def record_llm_usage(input_tokens: int, output_tokens: int) -> None:
    _recorder(input_tokens, output_tokens)


def set_recorder(fn: Callable[[int, int], None]) -> Callable[[int, int], None]:
    """Replace the recorder. Returns the previous one (for restoration)."""
    global _recorder
    prev = _recorder
    _recorder = fn
    return prev
