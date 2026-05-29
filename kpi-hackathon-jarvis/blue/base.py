from __future__ import annotations

from abc import ABC, abstractmethod

from shared.types import BlueJudgment, RedExtraction


class BlueAgent(ABC):
    """Implement this to participate as a blue agent.

    The orchestrator passes the original document and the red agent's
    public extraction (KPIs only, no hallucination metadata). Return one
    verdict per KPI id.
    """

    name: str = "unnamed-blue"

    @abstractmethod
    def judge(
        self, document_text: str, extraction: RedExtraction
    ) -> BlueJudgment: ...
