"""Blue baselines — rule-based and LLM-driven.

Two reference agents in this file:
  - BaselineBlueAgent — heuristic, no API key. Flags as hallucinated any KPI
    whose value (as a string) doesn't appear verbatim in the document.
  - LLMBlueAgent     — minimal Groq-driven baseline using gpt-oss-120b.

Pick one in blue/submission.py. Both are intentionally weak — students
should outperform them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from shared.metering import record_llm_usage
from shared.types import BlueJudgment, KPI, RedExtraction, Verdict

from .base import BlueAgent


# ─────────────────────── Rule-based baseline ───────────────────────


class BaselineBlueAgent(BlueAgent):
    """If the KPI value (as a string) appears verbatim in the document,
    trust it; otherwise flag as hallucinated. Naive on purpose — it will
    miss unit swaps, scope/period corruption, and value-coincidence false
    positives.
    """

    name = "rule-blue"

    def judge(
        self, document_text: str, extraction: RedExtraction
    ) -> BlueJudgment:
        return BlueJudgment(
            verdicts={
                kpi.id: self._verdict_for(kpi, document_text)
                for kpi in extraction.kpis
            }
        )

    @staticmethod
    def _verdict_for(kpi: KPI, document: str) -> Verdict:
        if BaselineBlueAgent._value_appears_in(kpi.value, document):
            return Verdict.CORRECT
        return Verdict.HALLUCINATED

    @staticmethod
    def _value_appears_in(value: float | str, document: str) -> bool:
        if isinstance(value, str):
            return value.strip() != "" and value in document
        # Try a couple of stringifications: "12500", "12,500", "3.8", "3,8"
        candidates = {str(value)}
        if isinstance(value, float) and value.is_integer():
            candidates.add(str(int(value)))
        if isinstance(value, int):
            candidates.add(f"{value:,}")
        for c in list(candidates):
            candidates.add(c.replace(".", ","))
        return any(c in document for c in candidates)


# ─────────────────────── LLM baseline ───────────────────────


class LLMBlueAgent(BlueAgent):
    """Minimal Groq-driven blue agent using gpt-oss-120b.

    Intentionally barebones — single LLM call, weak prompt. Replace or
    extend in `blue/submission.py`. Missing verdicts default to CORRECT
    (benign — gives the KPI the benefit of the doubt).

    Setup: `pip install -e ".[llm]"`. For the tournament, set LLM_BASE_URL +
    LLM_API_KEY (your team's proxy virtual key) in template/.env; for local dev
    straight against Groq, set GROQ_API_KEY instead.
    """

    name = "llm-blue"

    _DEFAULT_MODEL = "openai/gpt-oss-120b"
    _BASE_URL = "https://api.groq.com/openai/v1"
    # gpt-oss-120b is a reasoning model: if max output is left at the small
    # provider default, reasoning eats the whole budget and the final answer
    # comes back empty (→ json_validate_failed). Give it generous room so a big
    # verdict list (one per KPI, on a large document) always fits.
    _MAX_OUTPUT_TOKENS = 16000

    # JSON mode (not strict json_schema): blue's output is long (one verdict per
    # KPI, 100+ on big docs), and strict mode is all-or-nothing — a single stray
    # token (e.g. a mistyped key) rejects the WHOLE response. JSON mode only
    # guarantees valid JSON; a malformed verdict is skipped by the parser and
    # backfilled to CORRECT, so the agent degrades per-item instead of per-call.

    _INSTRUCTIONS = """You audit KPIs that another agent extracted from a document.
The other agent was instructed to keep most KPIs faithful and plant
hallucinations in a minority of them (typically a fifth to a third).

For each KPI, return exactly one verdict — no abstention is allowed:

  "correct"       — the KPI is faithful to the document. Allow formatting
                    equivalences: 12,500 == 12500, 3.8B == 3,800M, "Revenue"
                    == "Total revenue", abbreviated or paraphrased names.
                    These are NOT hallucinations.

  "hallucinated"  — the document contradicts the KPI in name, value, unit,
                    period or scope, OR the KPI is not in the document at all.
                    Example: doc says "Group Revenue 2023 = 3.8 B€",
                    KPI says 3.5 B€ → hallucinated.

Scoring is symmetric: a false positive (flagging a real KPI) and a false
negative (missing a hallucination) cost the same. Commit to a call on
every KPI — when in doubt, lean on the prior that a minority (~20-33%) are hallucinated.

Reply with one JSON object: {"verdicts": [{"id": <int>, "verdict": "<verdict>"}, ...]}.
Use the exact integer ids you were given, one entry per KPI."""

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        from openai import OpenAI  # local import so the rule-based baseline stays dep-free

        self._load_dotenv()
        # Set LLM_BASE_URL + LLM_API_KEY (your team's proxy virtual key) to route
        # through the metering proxy; falls back to Groq directly for local dev.
        base_url = os.environ.get("LLM_BASE_URL", self._BASE_URL)
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No API key. Set LLM_API_KEY (proxy virtual key) or GROQ_API_KEY "
                "in template/.env or the environment."
            )
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def judge(
        self, document_text: str, extraction: RedExtraction
    ) -> BlueJudgment:
        kpi_payload = [
            {
                "id": kpi.id,
                "name": kpi.name,
                "value": kpi.value,
                "unit": kpi.unit,
                "period": kpi.period,
                "scope": kpi.scope,
            }
            for kpi in extraction.kpis
        ]
        user_input = (
            "DOCUMENT:\n"
            f"{document_text}\n\n"
            "KPIs TO AUDIT:\n"
            f"{json.dumps(kpi_payload, ensure_ascii=False, indent=2)}"
        )

        response = self._client.responses.create(
            model=self._model,
            instructions=self._INSTRUCTIONS,
            input=user_input,
            max_output_tokens=self._MAX_OUTPUT_TOKENS,
            text={"format": {"type": "json_object"}},
        )
        _record_usage(response)
        raw = (response.output_text or "").strip()
        verdicts = self._parse_response(raw)

        # Backfill any KPI the model forgot. CORRECT is the benign default —
        # with ~20% hallucination rate, defaulting to CORRECT is expected
        # to cost less than defaulting to HALLUCINATED.
        for kpi in extraction.kpis:
            verdicts.setdefault(kpi.id, Verdict.CORRECT)
        return BlueJudgment(verdicts=verdicts)

    @staticmethod
    def _load_dotenv() -> None:
        if os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY"):
            return
        here = Path(__file__).resolve()
        for candidate in (
            here.parent.parent / ".env",
            here.parent.parent.parent / ".env",
        ):
            if not candidate.exists():
                continue
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            if os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY"):
                return

    @staticmethod
    def _parse_response(raw: str) -> dict[int, Verdict]:
        # Strip markdown code fences if the model wrapped its JSON.
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        verdicts: dict[int, Verdict] = {}
        for item in payload.get("verdicts") or []:
            try:
                verdicts[int(item["id"])] = Verdict(item["verdict"])
            except (ValueError, KeyError, TypeError):
                continue
        return verdicts


def _record_usage(response) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    record_llm_usage(
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
    )
