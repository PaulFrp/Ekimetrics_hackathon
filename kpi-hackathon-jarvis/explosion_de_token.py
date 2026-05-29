"""Advanced Blue Agent — multi-agent pipeline.

Architecture:
  Agent 0   — Text Cleaner: noisy raw txt → clean markdown
  Agent 1   — Extractor: clean doc → reference KPI JSON
  Agent 2a  — Numeric checker:   P1 digit-swap, P2 ×1.01, P3 gaussian, P6 ratio-mislead
  Agent 2b  — Metadata checker:  P4 period-shift, P7 unit-scale-swap
  Agent 2c  — Name/scope checker: P5 fabricated "(adjusted)", P7 synonym, P7b scope-swap
  Agent 2d  — Aggregator: merges 2a/2b/2c votes → correct / suspicious / hallucinated
  Agent 2e  — Internal consistency:
                Heuristic layer: intra-element checks (value vs unit type)
                                 + cross-element checks (percentages, subtotals, magnitudes)
                LLM layer: subtle semantic inconsistencies
  Agent 3   — Advocate:   argues suspicious KPI is CORRECT
  Agent 4   — Prosecutor: argues suspicious KPI is HALLUCINATED
  Agent 5   — Judge:      reads both arguments → final verdict

Agents 2a/2b/2c/2e run in parallel.
Agents 3/4 run in parallel per suspicious KPI.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from shared.metering import record_llm_usage
from shared.types import BlueJudgment, KPI, RedExtraction, Verdict

from .base import BlueAgent


# ══════════════════════════════════════════════════════════════
#  Prompts
# ══════════════════════════════════════════════════════════════

_CLEANER_INSTRUCTIONS = """You are a document pre-processor. Your job is to clean a raw,
poorly-formatted text extracted from a PDF and convert it into clean Markdown.

RULES — what to REMOVE:
  - Standalone page numbers (a line containing only a number, e.g. "127", "128")
  - Repeated document headers/footers that appear on every page
    (e.g. "Bucher Industries", "Annual report 2022", "Environmental, social and ethics report")
  - Decorative separators (lines of dashes, underscores, dots with no content)
  - Artefacts from PDF extraction: mid-word line breaks, hyphenation artefacts (­),
    stray single characters on their own line

RULES — what to KEEP (NEVER remove):
  - ALL numerical values, even if they appear on a line alone — only remove a number
    if it is CLEARLY a page number (surrounded by document title / footer context)
  - All table data, KPI values, percentages, dates, financial figures
  - All section titles and paragraph text
  - Footnote markers like "1)" or "n.a." — keep them

RULES — formatting:
  - Use ## for main section titles, ### for subsections
  - Preserve tables as Markdown tables where possible
  - Merge hyphenated words split across lines (e.g. "environ-\nment" → "environment")
  - Normalise whitespace: collapse multiple blank lines to one

Return ONLY the cleaned Markdown. No commentary, no preamble."""

_EXTRACTOR_INSTRUCTIONS = """You are a financial document parser.

Read the document carefully and extract EVERY numerical KPI you find.
Return them as a JSON object — one entry per KPI:

{
  "kpis": [
    {
      "name":   "<metric name, e.g. Revenue, EBITDA, Net Income>",
      "value":  <number — use the exact value from the document>,
      "unit":   "<unit, e.g. M€, k$, % — null if absent>",
      "period": "<year or quarter, e.g. 2023, Q1 2024 — null if absent>",
      "scope":  "<entity/segment — null if absent>"
    }
  ]
}

Rules:
- Do NOT invent values. Only extract what is explicitly in the document.
- Normalise numeric strings to numbers where unambiguous (e.g. "3.8B€" → 3800 with unit "M€").
- If the same metric appears for multiple periods or scopes, create one entry per combination.
- Return ONLY the JSON object, no commentary."""

# ── Agent 2a: numeric values only ─────────────────────────────
_FLAGGER_NUMERIC_INSTRUCTIONS = """You are a KPI numeric value auditor.

CONTEXT: Between 15% and 25% of candidates are hallucinated.

You receive:
  - REFERENCE: trusted KPIs from the document.
  - CANDIDATES: KPIs to audit (already pre-filtered by heuristics).

Your task: check ONLY the numeric VALUE field for these patterns:

  P1 — DIGIT SWAP: two digits transposed (e.g. 14876 → 14867).
       Sorted digits of candidate == sorted digits of reference, but 2 positions differ.

  P2 — ×1.01 PERTURB: candidate value ≈ reference × 1.01 (exactly +1%).

  P3 — GAUSSIAN NOISE: candidate differs from reference by a small amount
       rounded to the nearest 1000 (e.g. 245000 → 246000, diff = 1000).

  P6 — RATIO MISLEAD: candidate ≈ reference × 1.05 or × 0.95 (exactly ±5%).

Do NOT check names, units, periods, or scopes — only raw numeric values.

Calibration: expect 15–25% hallucinated. The `heuristic_label` field is a hint.

Reply ONLY with:
{"verdicts": [{"id": <int>, "label": "<correct|suspicious|hallucinated>", "pattern": "<P1|P2|P3|P6|none>"}, ...]}"""

# ── Agent 2b: metadata (period + unit) ────────────────────────
_FLAGGER_METADATA_INSTRUCTIONS = """You are a KPI metadata auditor.

CONTEXT: Between 15% and 25% of candidates are hallucinated.

You receive:
  - REFERENCE: trusted KPIs from the document.
  - CANDIDATES: KPIs to audit.

Your task: check ONLY the PERIOD and UNIT fields for these patterns:

  P4 — PERIOD SHIFT: period is ±1 or ±2 years/months from reference
       (e.g. 2023 → 2022, 2024-03 → 2024-04).

  P7 — UNIT SCALE SWAP: value is numerically equivalent after unit conversion
       but unit changed scale (e.g. 3.8 B€ vs 3800 M€, or 1234 M€ vs 1.234 B€).
       The value×unit product is preserved — that is the hallucination.

Do NOT check names, scopes, or raw numeric values — only period and unit.

Calibration: expect 15–25% hallucinated. The `heuristic_label` field is a hint.

Reply ONLY with:
{"verdicts": [{"id": <int>, "label": "<correct|suspicious|hallucinated>", "pattern": "<P4|P7|none>"}, ...]}"""

# ── Agent 2c: name + scope ─────────────────────────────────────
_FLAGGER_NAME_INSTRUCTIONS = """You are a KPI name and scope auditor.

CONTEXT: Between 15% and 25% of candidates are hallucinated.

You receive:
  - REFERENCE: trusted KPIs from the document.
  - CANDIDATES: KPIs to audit.

Your task: check ONLY the NAME and SCOPE fields for these patterns:

  P5 — FABRICATED NAME: name ends with "(adjusted)" and no such KPI exists
       in the reference → always hallucinated.

  P7b — SCOPE SWAP: value and name match a reference KPI, but the scope
        field has been changed to a different entity/segment.

  P7c — SYNONYM SUBSTITUTION: name is a plausible financial synonym of a
        reference KPI but not present verbatim (e.g. "Net Profit" vs "Net Income",
        "Turnover" vs "Revenue"). Flag as suspicious.

Allow standard abbreviations (e.g. "Rev." == "Revenue", "Op. Inc." == "Operating Income").

Do NOT check numeric values, units, or periods.

Calibration: expect 15–25% hallucinated. The `heuristic_label` field is a hint.

Reply ONLY with:
{"verdicts": [{"id": <int>, "label": "<correct|suspicious|hallucinated>", "pattern": "<P5|P7b|P7c|none>"}, ...]}"""

# ── Agent 2d: aggregator ───────────────────────────────────────
_AGGREGATOR_INSTRUCTIONS = """You are a verdict aggregator for a KPI hallucination detector.

You receive the outputs of three independent specialist auditors for each KPI:
  - numeric_verdict  (checked value patterns: digit swap, ×1.01, gaussian, ±5%)
  - metadata_verdict (checked period shift, unit scale swap)
  - name_verdict     (checked fabricated name, scope swap, synonym)

Each verdict is one of: correct / suspicious / hallucinated.

Aggregation rules:
  1. If ANY specialist says "hallucinated" → final = "hallucinated"
  2. If ALL specialists say "correct"       → final = "correct"
  3. Otherwise                              → final = "suspicious"

CONTEXT: Between 15% and 25% of KPIs are hallucinated total.
Do not over-flag — only mark "hallucinated" if at least one specialist found a clear pattern.

Reply ONLY with:
{"verdicts": [{"id": <int>, "label": "<correct|suspicious|hallucinated>"}, ...]}"""

# ── Agent 2e: internal consistency ────────────────────────────
_CONSISTENCY_INSTRUCTIONS = """You are a financial consistency auditor.

You receive a list of KPIs from a single document. Some values may have been
corrupted. Your job is to detect internal inconsistencies WITHOUT comparing to
any external reference — only using the KPIs themselves.

Check for:

  C1 — PERCENTAGE PARTS DON'T SUM TO 100:
       e.g. Male 79% + Female 31% = 110% → flag both as suspicious.
       Tolerance: ±1% for rounding.

  C2 — SUBTOTALS DON'T MATCH TOTAL:
       e.g. Segment A + B + C ≠ Total Revenue → flag the outlier subtotal.

  C3 — DERIVED KPI INCONSISTENCY:
       e.g. EBITDA Margin = EBITDA / Revenue × 100. If all three are present,
       verify the formula. Flag if error > 1%.

  C4 — IMPOSSIBLE MAGNITUDE:
       e.g. EBITDA > Revenue, Margin > 100%, negative equity without explanation.

  C5 — SIGN INCONSISTENCY:
       e.g. Net Income positive but Operating Income deeply negative for same period,
       with no exceptional items mentioned.

  C6 — TEMPORAL INCONSISTENCY:
       e.g. Growth rate says +15% but N-1 and N values imply a different growth.

For each flagged KPI, explain which rule was violated and which other KPI it conflicts with.
Only flag KPIs where you are confident — do NOT flag based on vague suspicion.

Reply ONLY with:
{
  "flags": [
    {
      "id": <int>,
      "rule": "<C1|C2|C3|C4|C5|C6>",
      "reason": "<one sentence explaining the inconsistency>",
      "conflicting_ids": [<list of other KPI ids involved>]
    }
  ]
}
If no inconsistencies found, return {"flags": []}."""

# ── Agents 3/4/5: batch debate (1 call each, all suspicious KPIs at once) ────
_ADVOCATE_INSTRUCTIONS = """You are a defense lawyer for a batch of suspicious KPIs.

You receive:
  - The source document (trusted reference).
  - REFERENCE: KPIs extracted from the document.
  - SUSPICIOUS_KPIS: a list of KPIs flagged as suspicious, each with its best
    reference match (ref_match field, null if none found).

For EACH KPI, examine it INDIVIDUALLY and build the STRONGEST possible argument
that it is CORRECT. Consider formatting equivalences, rounding, abbreviations,
unit aliases, and scope aliases. Cite the document specifically.

Reply ONLY with:
{
  "arguments": [
    {"id": <int>, "argument": "<2-3 sentences defending this specific KPI>"},
    ...
  ]
}
One entry per KPI id. Do not skip any id."""

_PROSECUTOR_INSTRUCTIONS = """You are a prosecutor for a batch of suspicious KPIs.

You receive:
  - The source document (trusted reference).
  - REFERENCE: KPIs extracted from the document.
  - SUSPICIOUS_KPIS: a list of KPIs flagged as suspicious, each with its best
    reference match (ref_match field, null if none found).

For EACH KPI, examine it INDIVIDUALLY and build the STRONGEST possible argument
that it is HALLUCINATED. Point to the exact discrepancy: value, unit, period,
scope, or name. Cite the document specifically.

Reply ONLY with:
{
  "arguments": [
    {"id": <int>, "argument": "<2-3 sentences prosecuting this specific KPI>"},
    ...
  ]
}
One entry per KPI id. Do not skip any id."""

_JUDGE_INSTRUCTIONS = """You are an impartial judge ruling on a batch of suspicious KPIs.

You receive:
  - The source document.
  - SUSPICIOUS_KPIS: the KPIs under review.
  - ADVOCATE_ARGS: defense arguments (one per KPI id).
  - PROSECUTOR_ARGS: prosecution arguments (one per KPI id).

For EACH KPI id, examine the advocate and prosecutor arguments INDIVIDUALLY
and weigh them carefully against the document. Then deliver a verdict.

Calibration: expect 15–25% of ALL KPIs in the document to be hallucinated.
Do not over-flag — only mark "hallucinated" when the prosecutor's argument
is clearly stronger and supported by the document.

Reply ONLY with:
{
  "verdicts": [
    {"id": <int>, "verdict": "<correct|hallucinated>"},
    ...
  ]
}
One entry per KPI id. Do not skip any id."""


# ══════════════════════════════════════════════════════════════
#  Heuristic helpers — pattern-specific detectors (Python only)
# ══════════════════════════════════════════════════════════════

def _normalize_value(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        clean = v.replace(",", "").replace(" ", "").upper()
        for suffix, mult in (("B", 1_000), ("M", 1), ("K", 0.001)):
            if clean.endswith(suffix):
                try:
                    return float(clean[:-1]) * mult
                except ValueError:
                    return None
        try:
            return float(clean)
        except ValueError:
            return None
    return None


def _values_close(a: Any, b: Any, rtol: float = 0.005) -> bool:
    fa, fb = _normalize_value(a), _normalize_value(b)
    if fa is None or fb is None:
        return str(a).strip().lower() == str(b).strip().lower()
    if fa == 0 and fb == 0:
        return True
    return abs(fa - fb) / max(abs(fa), abs(fb)) <= rtol


def _is_digit_swap(candidate: Any, reference: Any) -> bool:
    try:
        cs = str(int(float(candidate)))
        rs = str(int(float(reference)))
    except (ValueError, TypeError):
        return False
    if len(cs) != len(rs) or sorted(cs) != sorted(rs):
        return False
    return sum(1 for a, b in zip(cs, rs) if a != b) == 2


def _is_perturb_1pct(candidate: Any, reference: Any) -> bool:
    fc, fr = _normalize_value(candidate), _normalize_value(reference)
    if fc is None or fr is None or fr == 0:
        return False
    return abs(fc / fr - 1.01) < 0.001


def _is_gaussian_perturb(candidate: Any, reference: Any) -> bool:
    fc, fr = _normalize_value(candidate), _normalize_value(reference)
    if fc is None or fr is None or fr == 0:
        return False
    diff = abs(fc - fr)
    if diff == 0:
        return False
    rel = diff / abs(fr)
    is_rounded = (diff % 1.0 < 0.01) or (diff % 0.001 < 1e-6)
    return is_rounded and 0.0005 <= rel <= 0.05


def _is_period_shifted(kpi_period: str | None, ref_period: str | None) -> bool:
    if not kpi_period or not ref_period:
        return False
    kp, rp = str(kpi_period).strip(), str(ref_period).strip()
    if re.fullmatch(r"\d{4}", kp) and re.fullmatch(r"\d{4}", rp):
        return abs(int(kp) - int(rp)) in (1, 2)
    def _to_months(s: str) -> int | None:
        parts = s.split("-")
        try:
            return int(parts[0]) * 12 + int(parts[1]) if len(parts) >= 2 else None
        except ValueError:
            return None
    km, rm = _to_months(kp), _to_months(rp)
    if km is not None and rm is not None:
        return abs(km - rm) in (1, 2)
    return False


def _is_fabricated_adjusted(name: str) -> bool:
    return name.strip().lower().endswith("(adjusted)")


def _is_ratio_mislead(candidate: Any, reference: Any, pct: float = 0.05) -> bool:
    fc, fr = _normalize_value(candidate), _normalize_value(reference)
    if fc is None or fr is None or fr == 0:
        return False
    ratio = fc / fr
    return abs(ratio - (1 + pct)) < 0.005 or abs(ratio - (1 - pct)) < 0.005


_UNIT_SCALE_MAP: dict[str, float] = {
    "B€": 1_000, "B$": 1_000, "BN€": 1_000, "BN$": 1_000,
    "M€": 1,     "M$": 1,     "MN€": 1,     "MN$": 1,
    "K€": 0.001, "K$": 0.001,
}

def _is_unit_scale_swap(kpi_value, kpi_unit, ref_value, ref_unit) -> bool:
    ku = (kpi_unit or "").strip().upper()
    ru = (ref_unit or "").strip().upper()
    if ku == ru:
        return False
    ks, rs = _UNIT_SCALE_MAP.get(ku), _UNIT_SCALE_MAP.get(ru)
    if ks is None or rs is None:
        return False
    fk, fr = _normalize_value(kpi_value), _normalize_value(ref_value)
    if fk is None or fr is None or fr == 0:
        return False
    return abs(fk * ks - fr * rs) / abs(fr * rs) < 0.01


def _heuristic_flag(kpi: KPI, reference: list[dict]) -> tuple[str, str]:
    """Fast rule-based pre-filter. Returns (label, reason)."""
    if _is_fabricated_adjusted(kpi.name):
        return "hallucinated", "P5: name ends with '(adjusted)'."

    name_lower = kpi.name.lower().strip()
    candidates = [
        r for r in reference
        if name_lower in r["name"].lower() or r["name"].lower() in name_lower
    ]
    if not candidates:
        return "suspicious", "P7c: no name match in reference — possible synonym."

    def _period_ok(r: dict) -> bool:
        rp = str(r.get("period") or "").strip()
        kp = str(kpi.period or "").strip()
        return rp == kp or not rp or not kp

    def _scope_ok(r: dict) -> bool:
        rs = str(r.get("scope") or "").strip().lower()
        ks = str(kpi.scope or "").strip().lower()
        return rs == ks or not rs or not ks

    contextual = [r for r in candidates if _period_ok(r) and _scope_ok(r)]
    pool = contextual if contextual else candidates
    best = pool[0]
    ref_val = best.get("value")
    ref_period = str(best.get("period") or "")

    if not _period_ok(best):
        if _is_period_shifted(kpi.period, ref_period):
            return "hallucinated", f"P4: period shifted {ref_period} → {kpi.period}."
        return "suspicious", f"P4?: period mismatch ({kpi.period} vs {ref_period})."

    if _is_unit_scale_swap(kpi.value, kpi.unit, ref_val, best.get("unit")):
        return "hallucinated", f"P7: unit scale swap ({best.get('unit')} → {kpi.unit})."

    if _values_close(kpi.value, ref_val):
        all_scopes = {str(r.get("scope") or "").lower() for r in candidates}
        ks = str(kpi.scope or "").lower()
        if ks and len(all_scopes) > 1 and ks not in all_scopes:
            return "suspicious", f"P7b: value matches but scope '{kpi.scope}' differs."
        return "correct", f"Matches reference: {best['name']} = {ref_val}."

    if _is_digit_swap(kpi.value, ref_val):
        return "hallucinated", f"P1: digit swap ({ref_val} → {kpi.value})."
    if _is_perturb_1pct(kpi.value, ref_val):
        return "hallucinated", f"P2: ×1.01 perturb ({ref_val} → {kpi.value})."
    if _is_gaussian_perturb(kpi.value, ref_val):
        return "hallucinated", f"P3: gaussian noise ({ref_val} → {kpi.value})."
    if _is_ratio_mislead(kpi.value, ref_val):
        return "hallucinated", f"P6: ±5% ratio mislead ({ref_val} → {kpi.value})."

    fk, fr = _normalize_value(kpi.value), _normalize_value(ref_val)
    if fk is not None and fr is not None and fr != 0:
        dev = abs(fk - fr) / abs(fr)
        if dev > 0.10:
            return "hallucinated", f"Value deviates {dev:.1%} from reference."
        return "suspicious", f"Value differs {dev:.1%} from reference — needs review."

    return "suspicious", "Could not confirm value against reference."


# ══════════════════════════════════════════════════════════════
#  Internal consistency checks (Agent 2e — Python layer)
# ══════════════════════════════════════════════════════════════

def _consistency_heuristic(kpis: list[KPI]) -> dict[int, str]:
    """
    Rule-based internal consistency checks on the biased JSON alone.
    Returns {kpi.id: reason} for flagged KPIs.

    Two layers:
      I — Intra-element: value vs declared unit type (single KPI)
      X — Cross-element: arithmetic/logic between multiple KPIs
    """
    flags: dict[int, str] = {}

    # ════════════════════════════════════════════════
    # LAYER I — Intra-element checks (value vs unit)
    # ════════════════════════════════════════════════

    # Integer unit types: value must be a whole number
    _INTEGER_UNITS = {"integer", "decimal", "count", "headcount"}
    # Units that imply value ∈ [0, 100]
    _PERCENT_UNITS = {"percent", "%", "percentage"}
    # Units that imply value > 0
    _POSITIVE_UNITS = {"hour", "MWh", "tCO2e", "metric_ton", "year", "day"}
    # Date units: value must look like a date string, never a float
    _DATE_UNITS = {"date"}

    for kpi in kpis:
        unit = str(kpi.unit or "").strip().lower()
        val = kpi.value

        # I1 — integer unit but non-integer value
        if unit in _INTEGER_UNITS and isinstance(val, float):
            if val != int(val):
                flags[kpi.id] = (
                    f"I1: unit='{kpi.unit}' implies integer but value={val} "
                    f"is fractional (e.g. 0.26 can't be a headcount)."
                )
                continue

        # I2 — percent unit but value clearly out of [0, 100]
        if unit in _PERCENT_UNITS:
            fv = _normalize_value(val)
            if fv is not None and (fv < 0 or fv > 100):
                flags[kpi.id] = (
                    f"I2: unit='{kpi.unit}' but value={val} is outside [0, 100]."
                )
                continue

        # I3 — physical unit but negative value (should be ≥ 0)
        if unit in _POSITIVE_UNITS:
            fv = _normalize_value(val)
            if fv is not None and fv < 0:
                flags[kpi.id] = (
                    f"I3: unit='{kpi.unit}' implies non-negative but value={val} < 0."
                )
                continue

        # I4 — date unit but value looks like a number (not a date string)
        if unit in _DATE_UNITS and isinstance(val, (int, float)):
            flags[kpi.id] = (
                f"I4: unit='date' but value={val} is numeric, not a date string."
            )
            continue

        # I5 — zero value for a KPI where zero is suspicious
        #       (employee counts, energy, emissions — 0 is almost always wrong)
        _NONZERO_UNITS = {"MWh", "tCO2e", "metric_ton", "hour"}
        if unit in {u.lower() for u in _NONZERO_UNITS}:
            fv = _normalize_value(val)
            if fv is not None and fv == 0.0:
                flags[kpi.id] = (
                    f"I5: unit='{kpi.unit}' — value=0 is suspicious for this metric type."
                )
                continue

        # I6 — extremely large headcount (> 10M employees is almost certainly wrong)
        if unit in _INTEGER_UNITS and "employee" in kpi.name.lower():
            fv = _normalize_value(val)
            if fv is not None and fv > 10_000_000:
                flags[kpi.id] = (
                    f"I6: employee count={val} exceeds 10M — implausible magnitude."
                )
                continue

        # I7 — duration in hours but value looks like a date (> 100k hours ≈ 11 years)
        if unit == "hour":
            fv = _normalize_value(val)
            if fv is not None and fv > 100_000:
                flags[kpi.id] = (
                    f"I7: unit='hour' but value={val} > 100,000h — likely wrong unit or value."
                )
                continue

    # ════════════════════════════════════════════════
    # LAYER X — Cross-element checks
    # ════════════════════════════════════════════════

    # ── C1: percentage parts should sum to ~100 ────────────────
    # Group % KPIs by (period, scope) and detect breakdowns that don't sum to 100
    # Strategy: only flag if name pattern suggests complementary categories
    # (Male/Female, Segment A/B/C etc.)

    _PCT_UNITS_CROSS = {"percent", "%", "percentage"}
    pct_groups: dict[str, list[KPI]] = defaultdict(list)
    for kpi in kpis:
        unit = str(kpi.unit or "").strip().lower()
        if unit in _PCT_UNITS_CROSS:
            group_key = f"{kpi.period}|{kpi.scope}"
            pct_groups[group_key].append(kpi)

    for group_key, group in pct_groups.items():
        if len(group) < 2:
            continue
        # Only check groups where names share a common prefix (same breakdown)
        # e.g. "% employees Female" + "% employees Male" → same breakdown
        def _common_prefix(names: list[str]) -> str:
            if not names:
                return ""
            prefix = names[0]
            for n in names[1:]:
                while not n.startswith(prefix):
                    prefix = prefix[:-1]
                    if not prefix:
                        return ""
            return prefix

        # Group by shared name prefix within the pct_group
        sub_groups: dict[str, list[KPI]] = defaultdict(list)
        for kpi in group:
            # Extract base name (strip last word as the "category" suffix)
            parts = kpi.name.rsplit(" ", 1)
            base = parts[0] if len(parts) > 1 else kpi.name
            sub_groups[base].append(kpi)

        for base, sub in sub_groups.items():
            if len(sub) < 2:
                continue
            total = sum(_normalize_value(k.value) or 0 for k in sub)
            # Only flag if it looks like a partition of 100% (total between 50 and 200)
            if 50 < total < 200 and abs(total - 100.0) > 1.5:
                outlier = max(sub, key=lambda k: abs((_normalize_value(k.value) or 0)))
                flags[outlier.id] = (
                    f"C1: '{base}' breakdown sums to {total:.1f}% ≠ 100% "
                    f"(ids: {[k.id for k in sub]})."
                )

    # ── C4: impossible magnitudes ──────────────────────────────
    # Build name→value lookup for same period
    by_period: dict[str, dict[str, KPI]] = defaultdict(dict)
    for kpi in kpis:
        by_period[str(kpi.period or "")][kpi.name.lower().strip()] = kpi

    for period, lookup in by_period.items():
        revenue = next(
            (lookup[n] for n in lookup if "revenue" in n or "turnover" in n), None
        )
        ebitda = next(
            (lookup[n] for n in lookup if "ebitda" in n), None
        )
        net_income = next(
            (lookup[n] for n in lookup if "net income" in n or "net profit" in n), None
        )

        if revenue and ebitda:
            rv = _normalize_value(revenue.value)
            ev = _normalize_value(ebitda.value)
            if rv is not None and ev is not None and rv != 0 and ev > rv:
                flags[ebitda.id] = f"C4: EBITDA ({ev}) > Revenue ({rv}) — impossible."

        if revenue and net_income:
            rv = _normalize_value(revenue.value)
            nv = _normalize_value(net_income.value)
            if rv is not None and nv is not None and rv > 0 and nv > rv:
                flags[net_income.id] = (
                    f"C4: Net Income ({nv}) > Revenue ({rv}) — impossible."
                )

    # ── C3: margin consistency ─────────────────────────────────
    for period, lookup in by_period.items():
        margin = next(
            (lookup[n] for n in lookup
             if ("margin" in n or "rate" in n) and "ebitda" in n), None
        )
        revenue = next(
            (lookup[n] for n in lookup if "revenue" in n or "turnover" in n), None
        )
        ebitda = next(
            (lookup[n] for n in lookup
             if "ebitda" in n and "margin" not in n), None
        )
        if margin and revenue and ebitda:
            mv = _normalize_value(margin.value)
            rv = _normalize_value(revenue.value)
            ev = _normalize_value(ebitda.value)
            if mv is not None and rv is not None and ev is not None and rv != 0:
                implied = (ev / rv) * 100
                if abs(implied - mv) / max(abs(mv), 1e-9) > 0.02:
                    flags[margin.id] = (
                        f"C3: EBITDA margin {mv:.1f}% inconsistent with "
                        f"EBITDA/Revenue = {implied:.1f}%."
                    )

    return flags


# ══════════════════════════════════════════════════════════════
#  Main Agent
# ══════════════════════════════════════════════════════════════

class LLMBlueAgent(BlueAgent):
    """Multi-agent pipeline blue agent.

    Agents 2a/2b/2c/2e run in parallel.
    Agent 2d aggregates their votes.
    Suspicious KPIs go through the Advocate/Prosecutor/Judge debate.
    """

    name = "advanced-blue"

    _DEFAULT_MODEL = "openai/gpt-oss-120b"
    _BASE_URL = "https://api.groq.com/openai/v1"
    _MAX_OUTPUT_TOKENS = 16_000

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        from openai import OpenAI
        self._load_dotenv()
        base_url = os.environ.get("LLM_BASE_URL", self._BASE_URL)
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("No API key. Set LLM_API_KEY or GROQ_API_KEY in .env.")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    # ── Entry point ────────────────────────────────────────────

    def judge(self, document_text: str, extraction: RedExtraction) -> BlueJudgment:
        kpis = list(extraction.kpis)
        if not kpis:
            return BlueJudgment(verdicts={})

        # Agent 0: clean noisy raw text → clean markdown
        clean_text = self._clean_document(document_text)

        # Agent 1: reconstruct reference from clean document
        reference = self._extract_reference(clean_text)

        # Heuristic pre-filter (free, instant)
        pre_labels = {kpi.id: _heuristic_flag(kpi, reference) for kpi in kpis}

        # Consistency heuristic on the biased JSON alone (free, instant)
        consistency_flags = _consistency_heuristic(kpis)

        # KPIs already settled by heuristics
        definitive: dict[int, Verdict] = {}
        to_review: list[KPI] = []
        for kpi in kpis:
            label, _ = pre_labels[kpi.id]
            if label == "hallucinated":
                definitive[kpi.id] = Verdict.HALLUCINATED
            elif label == "correct" and kpi.id not in consistency_flags:
                definitive[kpi.id] = Verdict.CORRECT
            else:
                to_review.append(kpi)

        if not to_review:
            return BlueJudgment(verdicts=definitive)

        # Agents 2a / 2b / 2c / 2e — run in parallel
        with ThreadPoolExecutor(max_workers=4) as pool:
            fut_num  = pool.submit(self._llm_flag_numeric,  to_review, reference, pre_labels)
            fut_meta = pool.submit(self._llm_flag_metadata, to_review, reference, pre_labels)
            fut_name = pool.submit(self._llm_flag_names,    to_review, reference, pre_labels)
            fut_cons = pool.submit(self._llm_consistency,   kpis, consistency_flags)

            votes_num  = fut_num.result()
            votes_meta = fut_meta.result()
            votes_name = fut_name.result()
            cons_extra = fut_cons.result()   # {kpi.id: Verdict} for newly found issues

        # Agent 2d: aggregate votes → correct / suspicious / hallucinated
        aggregated = self._aggregate(to_review, votes_num, votes_meta, votes_name)

        # Merge consistency flags (any flag from 2e → suspicious at minimum)
        for kpi_id, verdict in cons_extra.items():
            if kpi_id not in definitive:
                # Upgrade suspicious→hallucinated if both 2d and 2e agree
                current = aggregated.get(kpi_id, "suspicious")
                if verdict == Verdict.HALLUCINATED and current in ("suspicious", "hallucinated"):
                    aggregated[kpi_id] = "hallucinated"
                elif current == "correct":
                    aggregated[kpi_id] = "suspicious"

        # Split aggregated into definitive / suspicious
        suspicious_kpis: list[KPI] = []
        for kpi in to_review:
            label = aggregated.get(kpi.id, "suspicious")
            if label == "hallucinated":
                definitive[kpi.id] = Verdict.HALLUCINATED
            elif label == "correct":
                definitive[kpi.id] = Verdict.CORRECT
            else:
                suspicious_kpis.append(kpi)

        # Agents 3+4+5: debate on remaining suspicious KPIs
        if suspicious_kpis:
            debate_verdicts = self._debate_batch(clean_text, suspicious_kpis, reference)
            definitive.update(debate_verdicts)

        # Backfill → CORRECT (benign default)
        for kpi in kpis:
            definitive.setdefault(kpi.id, Verdict.CORRECT)

        return BlueJudgment(verdicts=definitive)

    # ── Agent 0: Text Cleaner ──────────────────────────────────

    def _clean_document(self, raw_text: str) -> str:
        """
        Convert noisy PDF-extracted text to clean Markdown.
        Falls back to the raw text if the LLM call fails.
        Also applies a fast regex pre-pass before the LLM to strip
        the most obvious artefacts cheaply.
        """
        # Fast regex pre-pass (free, instant)
        text = self._regex_preclean(raw_text)

        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=_CLEANER_INSTRUCTIONS,
                input=f"RAW DOCUMENT:\n{text}",
                max_output_tokens=self._MAX_OUTPUT_TOKENS,
            )
            _record_usage(response)
            cleaned = (response.output_text or "").strip()
            return cleaned if cleaned else text
        except Exception:
            return text

    @staticmethod
    def _regex_preclean(text: str) -> str:
        """
        Fast rule-based pre-cleaning before the LLM pass.
        Removes:
          - Lines that are ONLY a page number (1–4 digits, optionally surrounded
            by whitespace). We require the line to be isolated (blank lines above/below
            OR at start/end of document) to avoid stripping real standalone values.
          - Soft-hyphen artefacts (­ U+00AD)
          - Excessive blank lines (collapse 3+ to 2)
        """
        lines = text.split("\n")
        cleaned_lines: list[str] = []
        n = len(lines)
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Page number heuristic: line is 1–4 digits, and neighbours are
            # blank or document-title-like (short, no punctuation)
            if re.fullmatch(r"\d{1,4}", stripped):
                prev_blank = (i == 0) or (lines[i - 1].strip() == "")
                next_blank = (i == n - 1) or (lines[i + 1].strip() == "")
                if prev_blank or next_blank:
                    continue  # skip — looks like a page number

            # Remove soft hyphens
            line = line.replace("\u00ad", "")

            cleaned_lines.append(line)

        # Collapse 3+ consecutive blank lines to 2
        result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines))
        return result.strip()

    # ── Agent 1: Extractor ─────────────────────────────────────

    def _extract_reference(self, document_text: str) -> list[dict]:
        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=_EXTRACTOR_INSTRUCTIONS,
                input=f"DOCUMENT:\n{document_text}",
                max_output_tokens=self._MAX_OUTPUT_TOKENS,
                text={"format": {"type": "json_object"}},
            )
        except Exception:
            return []
        _record_usage(response)
        raw = _strip_fences((response.output_text or "").strip())
        try:
            return json.loads(raw).get("kpis") or []
        except json.JSONDecodeError:
            return []

    # ── Agents 2a / 2b / 2c ───────────────────────────────────

    def _llm_flag_numeric(
        self, kpis: list[KPI], reference: list[dict],
        pre_labels: dict[int, tuple[str, str]],
    ) -> dict[int, str]:
        return self._llm_flag_specialist(
            kpis, reference, pre_labels, _FLAGGER_NUMERIC_INSTRUCTIONS
        )

    def _llm_flag_metadata(
        self, kpis: list[KPI], reference: list[dict],
        pre_labels: dict[int, tuple[str, str]],
    ) -> dict[int, str]:
        return self._llm_flag_specialist(
            kpis, reference, pre_labels, _FLAGGER_METADATA_INSTRUCTIONS
        )

    def _llm_flag_names(
        self, kpis: list[KPI], reference: list[dict],
        pre_labels: dict[int, tuple[str, str]],
    ) -> dict[int, str]:
        return self._llm_flag_specialist(
            kpis, reference, pre_labels, _FLAGGER_NAME_INSTRUCTIONS
        )

    def _llm_flag_specialist(
        self, kpis: list[KPI], reference: list[dict],
        pre_labels: dict[int, tuple[str, str]], instructions: str,
    ) -> dict[int, str]:
        """Generic specialist flagger — one LLM call, returns {id: label}."""
        payload = [
            {"id": k.id, "name": k.name, "value": k.value,
             "unit": k.unit, "period": k.period, "scope": k.scope,
             "heuristic_label": pre_labels.get(k.id, ("suspicious", ""))[0],
             "heuristic_reason": pre_labels.get(k.id, ("", ""))[1]}
            for k in kpis
        ]
        user_input = (
            f"REFERENCE:\n{json.dumps(reference, ensure_ascii=False, indent=2)}\n\n"
            f"CANDIDATES:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=instructions,
                input=user_input,
                max_output_tokens=self._MAX_OUTPUT_TOKENS,
                text={"format": {"type": "json_object"}},
            )
        except Exception:
            return {k.id: pre_labels.get(k.id, ("suspicious",))[0] for k in kpis}
        _record_usage(response)
        raw = _strip_fences((response.output_text or "").strip())
        try:
            items = json.loads(raw).get("verdicts") or []
        except json.JSONDecodeError:
            return {}
        return {
            int(item["id"]): item["label"]
            for item in items
            if item.get("label") in ("correct", "suspicious", "hallucinated")
        }

    # ── Agent 2d: Aggregator ───────────────────────────────────

    def _aggregate(
        self,
        kpis: list[KPI],
        votes_num: dict[int, str],
        votes_meta: dict[int, str],
        votes_name: dict[int, str],
    ) -> dict[int, str]:
        """
        Merge specialist votes.
        Rule: any "hallucinated" → hallucinated; all "correct" → correct; else suspicious.
        Falls back to LLM aggregator if votes conflict on many KPIs.
        """
        result: dict[int, str] = {}
        for kpi in kpis:
            votes = [
                votes_num.get(kpi.id, "suspicious"),
                votes_meta.get(kpi.id, "suspicious"),
                votes_name.get(kpi.id, "suspicious"),
            ]
            if "hallucinated" in votes:
                result[kpi.id] = "hallucinated"
            elif all(v == "correct" for v in votes):
                result[kpi.id] = "correct"
            else:
                result[kpi.id] = "suspicious"
        return result

    # ── Agent 2e: Internal consistency (LLM layer) ─────────────

    def _llm_consistency(
        self, kpis: list[KPI], heuristic_flags: dict[int, str]
    ) -> dict[int, Verdict]:
        """
        LLM consistency check on the full KPI list.
        Returns {kpi.id: Verdict} for any newly flagged KPIs.
        """
        payload = [
            {"id": k.id, "name": k.name, "value": k.value,
             "unit": k.unit, "period": k.period, "scope": k.scope,
             "heuristic_consistency_flag": heuristic_flags.get(k.id)}
            for k in kpis
        ]
        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=_CONSISTENCY_INSTRUCTIONS,
                input=f"KPIs:\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
                max_output_tokens=4096,
                text={"format": {"type": "json_object"}},
            )
        except Exception:
            # Fallback: promote heuristic flags to HALLUCINATED
            return {kid: Verdict.HALLUCINATED for kid in heuristic_flags}
        _record_usage(response)
        raw = _strip_fences((response.output_text or "").strip())
        try:
            flags = json.loads(raw).get("flags") or []
        except json.JSONDecodeError:
            return {}

        result: dict[int, Verdict] = {}
        # Heuristic flags → always at least suspicious
        for kid in heuristic_flags:
            result[kid] = Verdict.HALLUCINATED
        # LLM flags → hallucinated
        for item in flags:
            try:
                result[int(item["id"])] = Verdict.HALLUCINATED
            except (KeyError, ValueError, TypeError):
                continue
        return result

    # ── Agents 3+4+5: Batch Debate (3 calls total, regardless of N) ──

    def _debate_batch(
        self, document_text: str, suspicious_kpis: list[KPI], reference: list[dict]
    ) -> dict[int, Verdict]:
        """
        3 API calls total for the entire suspicious batch:
          1. Advocate   — 1 call for all suspicious KPIs
          2. Prosecutor — 1 call for all suspicious KPIs (parallel with advocate)
          3. Judge      — 1 call receiving all args, returns all verdicts
        """
        if not suspicious_kpis:
            return {}

        # Build enriched payload: each KPI + its best reference match
        kpi_payload = [
            {
                "id": kpi.id,
                "name": kpi.name,
                "value": kpi.value,
                "unit": kpi.unit,
                "period": kpi.period,
                "scope": kpi.scope,
                "ref_match": _find_best_ref(kpi, reference),
            }
            for kpi in suspicious_kpis
        ]

        base_input = (
            f"DOCUMENT:\n{document_text}\n\n"
            f"REFERENCE:\n{json.dumps(reference, ensure_ascii=False)}\n\n"
            f"SUSPICIOUS_KPIS:\n{json.dumps(kpi_payload, ensure_ascii=False, indent=2)}"
        )

        # Agents 3 & 4 — parallel, 1 call each
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_adv = ex.submit(
                self._call_json, _ADVOCATE_INSTRUCTIONS, base_input, "arguments"
            )
            fut_pro = ex.submit(
                self._call_json, _PROSECUTOR_INSTRUCTIONS, base_input, "arguments"
            )
            adv_args = fut_adv.result()   # list[{"id": int, "argument": str}]
            pro_args = fut_pro.result()

        # Agent 5 — Judge, 1 call
        judge_input = (
            f"DOCUMENT:\n{document_text}\n\n"
            f"SUSPICIOUS_KPIS:\n{json.dumps(kpi_payload, ensure_ascii=False, indent=2)}\n\n"
            f"ADVOCATE_ARGS:\n{json.dumps(adv_args, ensure_ascii=False, indent=2)}\n\n"
            f"PROSECUTOR_ARGS:\n{json.dumps(pro_args, ensure_ascii=False, indent=2)}"
        )
        verdicts_raw = self._call_json(_JUDGE_INSTRUCTIONS, judge_input, "verdicts")

        # Parse judge output → {kpi.id: Verdict}
        results: dict[int, Verdict] = {}
        for item in verdicts_raw:
            try:
                kpi_id = int(item["id"])
                v = item.get("verdict", "correct")
                results[kpi_id] = Verdict.HALLUCINATED if v == "hallucinated" else Verdict.CORRECT
            except (KeyError, ValueError, TypeError):
                continue

        # Backfill missing → CORRECT (benign default)
        for kpi in suspicious_kpis:
            results.setdefault(kpi.id, Verdict.CORRECT)

        return results

    # ── LLM call helpers ───────────────────────────────────────

    def _call_json(self, instructions: str, user_input: str, key: str) -> list:
        """
        Generic JSON call. Returns the list at `key` in the response,
        or [] on failure. Always uses json_object mode.
        """
        try:
            r = self._client.responses.create(
                model=self._model,
                instructions=instructions,
                input=user_input,
                max_output_tokens=self._MAX_OUTPUT_TOKENS,
                text={"format": {"type": "json_object"}},
            )
            _record_usage(r)
            raw = _strip_fences((r.output_text or "").strip())
            return json.loads(raw).get(key) or []
        except Exception:
            return []

    # ── Dotenv loader ──────────────────────────────────────────

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


# ══════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════

def _strip_fences(raw: str) -> str:
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


def _find_best_ref(kpi: KPI, reference: list[dict]) -> dict | None:
    name_lower = kpi.name.lower().strip()
    candidates = [
        r for r in reference
        if name_lower in r["name"].lower() or r["name"].lower() in name_lower
    ]
    if not candidates:
        return None
    for r in candidates:
        if str(r.get("period") or "") == str(kpi.period or ""):
            return r
    return candidates[0]


def _record_usage(response) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    record_llm_usage(
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
    )