from __future__ import annotations

from abc import ABC, abstractmethod

import os
import json

from shared.types import GroundTruth, RedExtraction

# In-memory buffer of debug events (appended when RED_DEBUG is set). Each
# event is a serializable dict with at least a `technique` key. Scripts can
# read this buffer after extraction to build a human summary.
_RED_DEBUG_EVENTS: list[dict] = []


class RedAgent(ABC):
    """Base class for a red (adversarial) agent.

    You receive the document text and the clean ground-truth KPIs. Your job
    is adversarial: decide which KPIs to pass through unchanged, which to
    corrupt, and which extra KPIs to fabricate. Return a `RedExtraction`
    containing only your final KPI list — do NOT mark which entries you
    tampered with. The scorer derives each KPI's truth from the ground
    truth (see `shared.scoring.classify_kpi`).

    Quotas (GT-relative):
    - Reproduce at least 50% of the ground-truth KPIs exactly.
    - Add at most 25% of the ground-truth count as hallucinations
      (corrupted + fabricated). See `shared.scoring.py` for details.

    Cascading (dependency-aware) edits:
    Many KPIs are derived values (ratios, percentages, subtotals). If you
    change a base KPI (for example, "Total Revenue"), you should also
    adjust dependent KPIs (for example, "Revenue per Customer" or any
    ratios that use that base) so the document stays internally
    consistent and the tampering is less obvious. Small, plausible drifts
    are more deceptive than large, contradictory changes.

    Implementation guidance for subclasses:
    - Prefer small perturbations (e.g. ±1–5%, or subtle rounding) to avoid
      easy detection.
    - When fabricating KPIs, match units and typical magnitudes in the
      document (currency, percentage points, counts).
    - Preserve formatting, significant digits and units where possible.
    - Centralize dependency updates by overriding `cascade_kpi_changes()`.

    The default `cascade_kpi_changes()` implementation is a no-op; real
    agents should override it with domain-aware logic.
    """

    name: str = "unnamed-red"

    def cascade_kpi_changes(self, kpis: list, ground_truth: GroundTruth) -> list:
        """Apply dependency-aware adjustments to a KPI list.

        Many KPI sets include derived values (ratios, percentages, growth
        rates). When a base value is altered, subclasses should update any
        dependent KPIs so the final extraction remains internally
        consistent. This method should accept and return the KPI list in
        the same format the subclass will place in the returned
        `RedExtraction`.

        Default behaviour: no-op (returns `kpis` unchanged). Override this
        to implement cascading updates specific to the dataset or domain.
        """
        return kpis

    def _debug_field_change(self, technique: str, kpi, field: str, old, new) -> None:
        if not os.environ.get("RED_DEBUG"):
            return
        loc = f"{getattr(kpi, 'id', '?')}:{getattr(kpi, 'name', '')} ({getattr(kpi,'period', '')},{getattr(kpi,'scope', '')})"
        ev = {
            "technique": technique,
            "type": "field_change",
            "kpi": loc,
            "field": field,
            "old": old,
            "new": new,
        }
        _RED_DEBUG_EVENTS.append(ev)
        print(json.dumps(ev, ensure_ascii=False))

    def _debug_new_kpi(self, technique: str, kpi) -> None:
        if not os.environ.get("RED_DEBUG"):
            return
        ev = {
            "technique": technique,
            "type": "kpi_new",
            "kpi_new": kpi.model_dump() if hasattr(kpi, 'model_dump') else str(kpi),
        }
        _RED_DEBUG_EVENTS.append(ev)
        print(json.dumps(ev, ensure_ascii=False))

    # mislead implementation moved to agent-local override in red/baseline.py

    @abstractmethod
    def extract(self, document_text: str, ground_truth: GroundTruth) -> RedExtraction: ...
