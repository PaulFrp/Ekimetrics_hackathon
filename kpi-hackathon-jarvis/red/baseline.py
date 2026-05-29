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
from shared.scoring import classify_kpi
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
            # Log fabricated KPI
            self._debug_new_kpi("fabricate_with_existing_value", fab)
            kpis.append(fab)

        return RedExtraction(kpis=kpis)

    def _try_random_modification(
        self,
        kpis: list[KPI],
        idx: int,
        ground_truth: GroundTruth,
    ) -> None:
        options = [
            ("swap_two_digits", self.swap_two_first_nonzero_digits),
            ("perturb_value", lambda k: self.perturb_value_gaussian(k, 1.01, len(kpis))),
            ("shift_validation_key", lambda k: self.shift_period(k, ground_truth)),
        ]
        self._rng.shuffle(options)
        for name, fn in options:
            original = kpis[idx]
            modified = fn(kpis[idx])
            if modified is not None and modified != original:
                # Log field-level changes
                for field in ("name", "value", "unit", "period", "scope"):
                    old = getattr(original, field, None)
                    new = getattr(modified, field, None)
                    if old != new:
                        self._debug_field_change(name, original, field, old, new)
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
    def swap_two_first_nonzero_digits(kpi: KPI) -> KPI | None:
        """Swap the first two non-zero digits in an integer-like KPI value."""
        if not isinstance(kpi.value, (int, float)):
            return None

        try:
            s = str(int(kpi.value))
        except Exception:
            return None

        digits = list(s)
        first = -1
        second = -1
        for i, ch in enumerate(digits):
            if ch != "0":
                if first == -1:
                    first = i
                elif second == -1:
                    second = i
                    break

        if first == -1 or second == -1:
            return None

        digits[first], digits[second] = digits[second], digits[first]
        try:
            new_val = int("".join(digits))
        except Exception:
            return None

        if new_val == int(kpi.value):
            return None

        return kpi.model_copy(update={"value": float(new_val)})

    @staticmethod
    def perturb_value_gaussian(kpi: KPI, rng: random.Random, n_gt: int) -> KPI | None:
        """Perturb a KPI numeric value with Gaussian noise and round to thousands."""
        if not isinstance(kpi.value, (int, float)):
            return None

        base = float(kpi.value)
        sigma = 1.0 / max(n_gt, 1)
        noise = rng.gauss(0, sigma)

        new_val = base * (1 + noise)
        new_val = round(new_val / 1000.0) * 1000.0

        if new_val == base:
            new_val = base + (1000.0 if rng.random() < 0.5 else -1000.0)

        if isinstance(kpi.value, int):
            new_val = int(new_val)

        return kpi.model_copy(update={"value": new_val})

    @staticmethod
    def shift_period(kpi: KPI, rng: random.Random) -> KPI | None:
        """Shift a KPI period string by one month or one year depending on format."""
        if not kpi.period:
            return None

        p = kpi.period.strip()

        if p == "2019-01-01—2019-12-31":
            return None

        if len(p) == 10 and p[4] == "-" and p[7] == "-":
            try:
                year = int(p[0:4])
                month = int(p[5:7])
                day = int(p[8:10])
            except Exception:
                return None

            delta = 1 if rng.random() < 0.5 else -1
            month += delta

            if month < 1:
                month = 12
                year -= 1
            elif month > 12:
                month = 1
                year += 1

            new_period = f"{year:04d}-{month:02d}-{day:02d}"
            return kpi.model_copy(update={"period": new_period})

        if len(p) == 7 and p[4] == "-":
            try:
                year = int(p[0:4])
                month = int(p[5:7])
            except Exception:
                return None

            delta = 1 if rng.random() < 0.5 else -1
            month += delta

            if month < 1:
                month = 12
                year -= 1
            elif month > 12:
                month = 1
                year += 1

            new_period = f"{year:04d}-{month:02d}"
            return kpi.model_copy(update={"period": new_period})

        try:
            year = int(p)
            delta = 1 if rng.random() < 0.5 else -1
            return kpi.model_copy(update={"period": str(year + delta)})
        except Exception:
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

    # Enable/disable the misleading perturbation and its magnitude (fraction).
    _MISLEAD_ENABLED = True
    _MISLEAD_PCT = 0.05
    # When True, allow mislead to create new hallucinations (for testing).
    _MISLEAD_FORCE = False

    _DEFAULT_MODEL = "openai/gpt-oss-120b"
    _BASE_URL = "https://api.groq.com/openai/v1"

    _KEEP_FRACTION = 0.70   # share of GT copied verbatim → comfortably >=50% coverage
    _MAX_HALLUC = 12        # upper bound on LLM-invented hallucinations (keeps output tiny)
    _SEED_SAMPLE = 40       # real KPIs shown to the LLM to seed plausible fakes

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

    _FINANCE_LONG_INSTRUCTIONS = """You are a financial KPI red-team agent for the finance_long round.

Given a KPI JSON from a long-form financial filing, create a modified version containing subtle, realistic hallucinations.

Rules:
- Modify at most 25% of KPI entries.
- Keep at least 75% unchanged.
- Preserve the exact JSON schema.
- Return only valid JSON.
- No explanations.

Allowed modifications:
- Swap scopes using existing scopes from the long dataset.
- Swap periods using existing periods from the long dataset.
- Change units (e.g. M€ <-> B€) and convert values accordingly.
- Replace KPI names with realistic GAAP/business synonyms.
- Slightly perturb values (<1%).
- Modify ratios, margins, growth rates, or derived KPIs while keeping them plausible.
- Change KPI types/classes when realistic.
- Reuse existing values, periods, scopes, and labels whenever possible.
- If a naming/unit/scope pattern exists in most records, propagate it consistently.
- Keep all changes coherent with each other.

Priority:
1. Scope swaps
2. Period swaps
3. Ratio / derived KPI modifications
4. Unit conversions
5. Pattern propagation
6. Small value changes
7. Synonym substitutions
8. Type changes

Goal:
Create realistic, internally consistent errors that are difficult to detect and require document-level reasoning rather than simple value matching.

Operational notes:
- Obey the tournament quotas: do not exceed 25% added hallucinations overall.
- When modifying derived KPIs, keep internal consistency (adjust bases or derived entries together).
- Preserve units, formatting, and significant digits unless the modification explicitly changes them (and adjust values accordingly).
- Output must be strictly valid JSON conforming to the `_SCHEMA` provided to the model.
- Do not include commentary or explanation — return raw JSON only.

Use the above as your instructions when inventing hallucinations for the long-form KPI batch provided.
"""

    # Runtime prompt used by the active LLM agent.
    _INSTRUCTIONS = _FINANCE_LONG_INSTRUCTIONS

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

        # LLM layer: a bounded batch of hallucinations (<=25% of GT).
        n_halluc = max(1, min(self._MAX_HALLUC, n // 4))
        fakes = self._invent_hallucinations(gt, n_halluc)[:n_halluc]

        gt_keys = {self._key(k) for k in gt}
        kpis: list[KPI] = []
        for k in kept:
            kpis.append(k.model_copy(update={"id": len(kpis)}))
        for f in fakes:
            if self._key(f) in gt_keys:  # accidental exact-real → not a hallucination
                continue
            # Log LLM-provided hallucination
            self._debug_new_kpi("llm_hallucination", f)
            kpis.append(f.model_copy(update={"id": len(kpis)}))
        # Optionally apply cascading/misleading perturbations so derived KPIs
        # become inconsistent with their bases and may trick the blue agent.
        try:
            if getattr(self, "_MISLEAD_ENABLED", True):
                pct = getattr(self, "_MISLEAD_PCT", 0.05)
                before = [k.model_dump() for k in kpis]
                kpis = self.mislead_ratio_by_changing_bases(
                    kpis, pct=pct, allow_random_fallback=True, ground_truth=ground_truth
                )
                after = [k.model_dump() for k in kpis]
                # Optional debug output when RED_DEBUG is set in the env.
                if os.environ.get("RED_DEBUG"):
                    print("RED_DEBUG: mislead changes:")
                    print(json.dumps({"before": before, "after": after}, ensure_ascii=False, indent=2))
        except Exception:
            # Never fail the match because of the mislead routine.
            pass

        return RedExtraction(kpis=kpis)

    @staticmethod
    def swap_two_first_nonzero_digits(kpi: KPI) -> KPI | None:
        """Swap the first two non-zero digits in an integer-like KPI value."""
        if not isinstance(kpi.value, (int, float)):
            return None

        try:
            s = str(int(kpi.value))
        except Exception:
            return None

        digits = list(s)
        first = -1
        second = -1
        for i, ch in enumerate(digits):
            if ch != "0":
                if first == -1:
                    first = i
                elif second == -1:
                    second = i
                    break

        if first == -1 or second == -1:
            return None

        digits[first], digits[second] = digits[second], digits[first]
        try:
            new_val = int("".join(digits))
        except Exception:
            return None

        if new_val == int(kpi.value):
            return None

        return kpi.model_copy(update={"value": float(new_val)})

    @staticmethod
    def perturb_value_gaussian(kpi: KPI, rng: random.Random, n_gt: int) -> KPI | None:
        """Perturb a KPI numeric value with Gaussian noise and round to thousands."""
        if not isinstance(kpi.value, (int, float)):
            return None

        base = float(kpi.value)
        sigma = 1.0 / max(n_gt, 1)
        noise = rng.gauss(0, sigma)

        new_val = base * (1 + noise)
        new_val = round(new_val / 1000.0) * 1000.0

        if new_val == base:
            new_val = base + (1000.0 if rng.random() < 0.5 else -1000.0)

        if isinstance(kpi.value, int):
            new_val = int(new_val)

        return kpi.model_copy(update={"value": new_val})

    @staticmethod
    def shift_period(kpi: KPI, rng: random.Random) -> KPI | None:
        """Shift a KPI period string by one month or one year depending on format."""
        if not kpi.period:
            return None

        p = kpi.period.strip()

        if p == "2019-01-01—2019-12-31":
            return None

        if len(p) == 10 and p[4] == "-" and p[7] == "-":
            try:
                year = int(p[0:4])
                month = int(p[5:7])
                day = int(p[8:10])
            except Exception:
                return None

            delta = 1 if rng.random() < 0.5 else -1
            month += delta

            if month < 1:
                month = 12
                year -= 1
            elif month > 12:
                month = 1
                year += 1

            new_period = f"{year:04d}-{month:02d}-{day:02d}"
            return kpi.model_copy(update={"period": new_period})

        if len(p) == 7 and p[4] == "-":
            try:
                year = int(p[0:4])
                month = int(p[5:7])
            except Exception:
                return None

            delta = 1 if rng.random() < 0.5 else -1
            month += delta

            if month < 1:
                month = 12
                year -= 1
            elif month > 12:
                month = 1
                year += 1

            new_period = f"{year:04d}-{month:02d}"
            return kpi.model_copy(update={"period": new_period})

        try:
            year = int(p)
            delta = 1 if rng.random() < 0.5 else -1
            return kpi.model_copy(update={"period": str(year + delta)})
        except Exception:
            return None

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

    def mislead_ratio_by_changing_bases(
        self,
        kpis: list[KPI],
        pct: float | None = None,
        prefer_small: bool = True,
        allow_random_fallback: bool = True,
        ground_truth: GroundTruth | None = None,
    ) -> list[KPI]:
        """LLM-specific override for mislead behaviour.

        This is a copy of the generic method but hosted on the LLM agent to
        keep LLM-specific tuning co-located with the agent. It delegates
        debugging to `RedAgent` helpers already available via inheritance.
        """
        # Default to the agent's configured pct if not provided
        pct = (pct if pct is not None else getattr(self, "_MISLEAD_PCT", 0.05))

        # Reuse the generic implementation from red.base where possible —
        # but keep a self-contained copy here to allow per-agent tuning.
        # Build simple name->KPI lookup
        lookup = {k.name.lower(): k for k in kpis}

        def parse_ratio_name(name: str) -> tuple[str | None, str | None]:
            lower = name.lower()
            if " per " in lower:
                a, b = lower.split(" per ", 1)
                return a.strip(), b.strip()
            if "/" in lower:
                a, b = lower.split("/", 1)
                return a.strip(), b.strip()
            return None, None

        changed = 0
        for k in list(kpis):
            num_name, den_name = parse_ratio_name(k.name)
            if not num_name or not den_name:
                continue
            def find_candidate(target: str):
                if target in lookup:
                    return lookup[target]
                for name, kp in lookup.items():
                    if target in name or name in target:
                        return kp
                return None

            num_k = find_candidate(num_name)
            den_k = find_candidate(den_name)
            if num_k is None or den_k is None:
                continue
            try:
                num_val = float(num_k.value)
                den_val = float(den_k.value)
            except Exception:
                continue

            # Only perturb if at least one base is already a hallucination,
            # unless the agent is forcing mislead for testing.
            if ground_truth is not None and not getattr(self, "_MISLEAD_FORCE", False):
                num_is_hall = classify_kpi(num_k, ground_truth) is not None
                den_is_hall = classify_kpi(den_k, ground_truth) is not None
                if not (num_is_hall or den_is_hall):
                    continue

            delta = pct if prefer_small else max(0.01, pct)
            new_num = round(num_val * (1 + delta), 6)
            new_den = round(den_val * (1 - delta), 6)
            # Log via inherited helper
            self._debug_field_change("mislead", num_k, "value", num_val, new_num)
            self._debug_field_change("mislead", den_k, "value", den_val, new_den)
            num_k.value = new_num
            den_k.value = new_den
            changed += 1

        # Fallback: only perturb numeric KPIs that are already hallucinated
        if changed == 0 and allow_random_fallback and ground_truth is not None:
            numeric_hall = []
            for kp in kpis:
                try:
                    float(kp.value)
                except Exception:
                    continue
                if classify_kpi(kp, ground_truth) is not None:
                    numeric_hall.append(kp)
            if len(numeric_hall) >= 2:
                a, b = numeric_hall[0], numeric_hall[1]
                try:
                    a_val = float(a.value)
                    b_val = float(b.value)
                    delta = pct if prefer_small else max(0.01, pct)
                    new_a = round(a_val * (1 + delta), 6)
                    new_b = round(b_val * (1 - delta), 6)
                    self._debug_field_change("mislead", a, "value", a_val, new_a)
                    self._debug_field_change("mislead", b, "value", b_val, new_b)
                    a.value = new_a
                    b.value = new_b
                    changed += 1
                except Exception:
                    pass

        # Extra long-round tactics: always try the new helper methods on a few
        # candidate KPIs so their techniques are visible in the summary.
        if ground_truth is not None:
            candidates = [kp for kp in kpis if isinstance(kp.value, (int, float))]
            for idx, kp in enumerate(candidates[: min(3, len(candidates))]):
                if idx == 0:
                    mod = self.swap_two_first_nonzero_digits(kp)
                    if mod is not None:
                        self._debug_field_change("swap_two_first_nonzero_digits", kp, "value", kp.value, mod.value)
                        kp.value = mod.value
                        changed += 1
                elif idx == 1:
                    mod = self.perturb_value_gaussian(kp, self._rng, len(kpis))
                    if mod is not None:
                        self._debug_field_change("perturb_value_gaussian", kp, "value", kp.value, mod.value)
                        kp.value = mod.value
                        changed += 1
                else:
                    mod = self.shift_period(kp, self._rng)
                    if mod is not None:
                        self._debug_field_change("shift_period", kp, "period", kp.period, mod.period)
                        kp.period = mod.period
                        changed += 1

        return kpis

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
