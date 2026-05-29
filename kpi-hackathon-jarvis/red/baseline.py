"""Red baselines — rule-based and LLM-driven.

Two reference agents in this file:
  - BaselineRedAgent — deterministic, no API key. Composes four
    transformations on the ground truth (swap digits, perturb value,
    shift validation key, fabricate with existing value).
  - LLMRedAgent     — minimal Groq-driven baseline using gpt-oss-120b.

Pick one in red/submission.py. Both are intentionally weak — students
should outperform them.
"""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

from shared.metering import record_llm_usage
from shared.types import GroundTruth, KPI, RedExtraction

from .base import RedAgent


# ─────────────────────── Rule-based baseline ───────────────────────


class BaselineRedAgent(RedAgent):
    """Stochastic rule-based baseline.

    Per call:
      - Pick a random fraction of GT to use as base (70–100%), so red keeps
        well over half the GT and stays inside the coverage quota.
      - Pick random indices to corrupt so ~20% of the output is hallucinated
        (well under the 25%-of-GT addition cap).
      - For each picked index, pick a random transformation among:
        swap_two_digits, perturb_value (×1.01), shift_validation_key.
      - Append one fabricated KPI whose value is borrowed from GT.

    The randomness makes the agent harder to game even though everything
    is rule-based. Pass `seed` to make a run reproducible.
    """

    name = "rule-red"

    _BASE_FRACTION_RANGE = (0.70, 1.0)
    _HALLUC_RATE = 0.20  # target hallucination share of the output

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def extract(
        self, document_text: str, ground_truth: GroundTruth
    ) -> RedExtraction:
        gt = list(ground_truth.kpis)
        if not gt:
            return RedExtraction(kpis=[])

        # Random base size.
        frac = self._rng.uniform(*self._BASE_FRACTION_RANGE)
        n_base = min(max(3, int(len(gt) * frac)), len(gt))

        kpis: list[KPI] = [self._copy(i, k) for i, k in enumerate(gt[:n_base])]

        # Target ~20% hallucination rate on (n_base + 1 fabricated).
        target_halluc = max(1, int(round((n_base + 1) * self._HALLUC_RATE)))
        n_modify = max(0, min(target_halluc - 1, n_base))

        indices = self._rng.sample(range(n_base), n_modify) if n_modify else []
        for idx in indices:
            self._try_random_modification(kpis, idx, ground_truth)

        fab = self.fabricate_with_existing_value(
            ground_truth, len(kpis), used_names={k.name for k in kpis}
        )
        if fab is not None:
            kpis.append(fab)

        return RedExtraction(kpis=kpis)

    def _try_random_modification(
        self,
        kpis: list[KPI],
        idx: int,
        ground_truth: GroundTruth,
    ) -> None:
        options = [
            self.swap_two_digits,
            lambda k: self.perturb_value(k, 1.01),
            lambda k: self.shift_validation_key(k, ground_truth),
        ]
        self._rng.shuffle(options)
        for fn in options:
            modified = fn(kpis[idx])
            if modified is not None and modified != kpis[idx]:
                kpis[idx] = modified
                return

    @staticmethod
    def _copy(new_id: int, k: KPI) -> KPI:
        return KPI(
            id=new_id,
            name=k.name,
            value=k.value,
            unit=k.unit,
            period=k.period,
            scope=k.scope,
            source_span=k.source_span,
        )

    @staticmethod
    def swap_two_digits(kpi: KPI) -> KPI | None:
        """Swap two adjacent digits in the numeric value (e.g. 14876 → 14867)."""
        if not isinstance(kpi.value, (int, float)) or kpi.value == 0:
            return None
        chars = list(repr(kpi.value))
        for i in range(len(chars) - 1, 0, -1):
            if (
                chars[i].isdigit()
                and chars[i - 1].isdigit()
                and chars[i] != chars[i - 1]
            ):
                chars[i], chars[i - 1] = chars[i - 1], chars[i]
                try:
                    new_val = float("".join(chars))
                except ValueError:
                    return None
                if new_val != kpi.value:
                    return kpi.model_copy(update={"value": new_val})
                return None
        return None

    @staticmethod
    def perturb_value(kpi: KPI, factor: float = 1.01) -> KPI | None:
        """Multiply the numeric value by `factor` (default +1%)."""
        if not isinstance(kpi.value, (int, float)):
            return None
        new_val = kpi.value * factor
        if new_val == kpi.value:
            return None
        return kpi.model_copy(update={"value": new_val})

    @staticmethod
    def shift_validation_key(kpi: KPI, ground_truth: GroundTruth) -> KPI | None:
        """Shift the period to a nearby year that doesn't exist in GT."""
        if not kpi.period:
            return None
        try:
            year = int(kpi.period)
        except (ValueError, TypeError):
            return None
        gt_keys = {(k.name, k.period, k.scope) for k in ground_truth.kpis}
        for delta in (-1, 1, -2, 2):
            new_period = str(year + delta)
            if (kpi.name, new_period, kpi.scope) not in gt_keys:
                return kpi.model_copy(update={"period": new_period})
        return None

    @staticmethod
    def fabricate_with_existing_value(
        ground_truth: GroundTruth, new_id: int, used_names: set[str]
    ) -> KPI | None:
        """New KPI whose value is borrowed from GT (so it's in the document)
        but whose name is a slight variation not present in GT."""
        gt_keys = {(k.name, k.period, k.scope) for k in ground_truth.kpis}
        for donor in ground_truth.kpis:
            if not isinstance(donor.value, (int, float)):
                continue
            new_name = f"{donor.name} (adjusted)"
            if (new_name, donor.period, donor.scope) in gt_keys:
                continue
            if new_name in used_names:
                continue
            return KPI(
                id=new_id,
                name=new_name,
                value=donor.value,
                unit=donor.unit,
                period=donor.period,
                scope=donor.scope,
            )
        return None


# ─────────────────────── LLM baseline ───────────────────────


class LLMRedAgent(RedAgent):
    """Hybrid Groq-driven red agent using gpt-oss-120b.

    The LLM is only used for what it is good at — inventing a handful of
    stealthy HALLUCINATIONS. The faithful majority of the extraction is copied
    verbatim from the ground truth in plain code. So the model emits ~10 KPIs
    regardless of document size (no giant structured output, no decode/empty
    failures, cheap + fast), while code guarantees the >=50% coverage quota.

    Setup: `pip install -e ".[llm]"`. For the tournament, set LLM_BASE_URL +
    LLM_API_KEY (your team's proxy virtual key) in template/.env; for local dev
    straight against Groq, set GROQ_API_KEY instead.
    """

    name = "llm-red"

    _DEFAULT_MODEL = "openai/gpt-oss-120b"
    _BASE_URL = "https://api.groq.com/openai/v1"

    _KEEP_FRACTION = 0.70   # share of GT copied verbatim → comfortably >=50% coverage
    _MAX_HALLUC = 12        # upper bound on LLM-invented hallucinations (keeps output tiny)
    _SEED_SAMPLE = 15       # real KPIs shown to the LLM to seed plausible fakes

    # Strict schema for the SMALL hallucination batch only (no id — code assigns
    # ids when assembling the final extraction). value is number-or-string;
    # unit/scope are nullable.
    _SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": ["kpis"],
        "properties": {
            "kpis": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "value", "unit", "period", "scope"],
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": ["number", "string"]},
                        "unit": {"type": ["string", "null"]},
                        "period": {"type": "string"},
                        "scope": {"type": ["string", "null"]},
                    },
                },
            }
        },
    }

    _INSTRUCTIONS = """You build adversarial test cases for a KPI verifier.

You are given a few REAL KPIs from a document. Invent new, intentionally WRONG
KPIs ("hallucinations"), each in one of two ways:
  - corrupt a real one: change its value, unit, scope, or period (small,
    stealthy deltas are best);
  - fabricate a plausible KPI that is NOT in the list.

Every KPI you return must be wrong — do NOT reproduce any real KPI unchanged.
Make them realistic so a verifier finds them hard to spot."""

    def __init__(self, model: str = _DEFAULT_MODEL, seed: int | None = None) -> None:
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
        self._rng = random.Random(seed)

    def extract(
        self, document_text: str, ground_truth: GroundTruth
    ) -> RedExtraction:
        gt = list(ground_truth.kpis)
        if not gt:
            return RedExtraction(kpis=[])
        n = len(gt)

        # Rule layer: copy a faithful majority of the GT verbatim (distinct KPIs
        # → distinct coverage). This alone clears the >=50% coverage quota.
        n_keep = max(1, min(n, math.ceil(self._KEEP_FRACTION * n)))
        kept = self._rng.sample(gt, n_keep)

        # LLM layer: a small, bounded batch of hallucinations (<=25% of GT).
        n_halluc = max(1, min(self._MAX_HALLUC, n // 4))
        fakes = self._invent_hallucinations(gt, n_halluc)[:n_halluc]

        gt_keys = {self._key(k) for k in gt}
        kpis: list[KPI] = []
        for k in kept:
            kpis.append(k.model_copy(update={"id": len(kpis)}))
        for f in fakes:
            if self._key(f) in gt_keys:  # accidental exact-real → not a hallucination
                continue
            kpis.append(f.model_copy(update={"id": len(kpis)}))
        return RedExtraction(kpis=kpis)

    def _invent_hallucinations(self, gt: list[KPI], n: int) -> list[KPI]:
        seed = self._rng.sample(gt, min(self._SEED_SAMPLE, len(gt)))
        seed_payload = [
            {"name": k.name, "value": k.value, "unit": k.unit, "period": k.period, "scope": k.scope}
            for k in seed
        ]
        user_input = (
            f"REAL KPIs:\n{json.dumps(seed_payload, ensure_ascii=False, indent=2)}\n\n"
            f"Return exactly {n} hallucinated KPIs."
        )
        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=self._INSTRUCTIONS,
                input=user_input,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "hallucinations",
                        "strict": True,
                        "schema": self._SCHEMA,
                    }
                },
            )
        except Exception:  # noqa: BLE001 — degrade to no hallucinations, never crash
            return []
        _record_usage(response)
        return self._parse_kpis((response.output_text or "").strip())

    @staticmethod
    def _key(k: KPI) -> tuple:
        return (k.name, k.period, k.scope, k.value, k.unit)

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
    def _parse_kpis(raw: str) -> list[KPI]:
        # Strip markdown code fences if the model wrapped its JSON.
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []

        kpis: list[KPI] = []
        for i, raw_kpi in enumerate(payload.get("kpis") or []):
            try:
                kpis.append(KPI(id=i, **raw_kpi))  # temp id; reassigned on assembly
            except (TypeError, ValueError):
                continue
        return kpis


def _record_usage(response) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    record_llm_usage(
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
    )
