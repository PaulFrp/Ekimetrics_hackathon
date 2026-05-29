from __future__ import annotations

from abc import ABC, abstractmethod

from shared.types import GroundTruth, RedExtraction


class RedAgent(ABC):
    """Implement this to participate as a red agent.

    You receive the document and the clean ground-truth KPIs. Your job is
    purely adversarial: decide which of those KPIs to pass through
    unchanged, which to corrupt, and what extra KPIs to fabricate. Return
    a `RedExtraction` with just your KPI list — you do NOT label which ones
    you tampered with. The scorer derives each KPI's truth from the ground
    truth (see `shared.scoring.classify_kpi`).

    Quota (GT-relative): reproduce at least 50% of the ground-truth KPIs
    faithfully, and add at most 25% of the ground-truth count as
    hallucinations (corrupted + fabricated). See `shared/scoring.py`.
    """

    name: str = "unnamed-red"

    @abstractmethod
    def extract(
        self, document_text: str, ground_truth: GroundTruth
    ) -> RedExtraction: ...
