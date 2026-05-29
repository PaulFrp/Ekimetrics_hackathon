"""Wire your red agent here. The orchestrator imports `agent` from this module.

Two baselines are provided as examples (both in `baseline.py`):
  - `BaselineRedAgent`  — modular code-only baseline, no API key needed.
  - `LLMRedAgent`       — Groq + a minimal prompt that returns JSON. Needs GROQ_API_KEY.

Swap which line is active to switch, or write your own agent in this folder.
"""
# from .baseline import BaselineRedAgent
from .baseline import LLMRedAgent

# agent = BaselineRedAgent()
agent = LLMRedAgent()
