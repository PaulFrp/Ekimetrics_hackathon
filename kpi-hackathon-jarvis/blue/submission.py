"""Wire your blue agent here. The orchestrator imports `agent` from this module.

Two baselines are provided as examples (both in `baseline.py`):
  - `BaselineBlueAgent`  — rule-based, no API key needed, kind of dumb. Smoke test.
  - `LLMBlueAgent`       — Groq + a minimal prompt that returns JSON. Needs GROQ_API_KEY.

Swap which line is active to switch, or write your own agent in this folder.
"""
# from .baseline import BaselineBlueAgent
from .baseline import LLMBlueAgent

# agent = BaselineBlueAgent()
agent = LLMBlueAgent()
