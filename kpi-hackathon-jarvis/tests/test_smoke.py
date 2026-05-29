from pathlib import Path

from blue.baseline import BaselineBlueAgent
from red.baseline import BaselineRedAgent
from shared.scoring import score_match
from shared.types import GroundTruth

EXAMPLE = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "finance_short"
    / "example_01"
)


def test_baseline_match_runs_end_to_end():
    document = (EXAMPLE / "document.txt").read_text()
    ground_truth = GroundTruth.model_validate_json(
        (EXAMPLE / "ground_truth.json").read_text()
    )

    red = BaselineRedAgent()
    blue = BaselineBlueAgent()

    extraction = red.extract(document, ground_truth)
    judgment = blue.judge(document, extraction.public_view())
    result = score_match(extraction, judgment, ground_truth)

    assert 0.0 <= result["red_score"] <= 1.0
    assert 0.0 <= result["blue_score"] <= 1.0
    assert result["kpis_extracted"] == len(extraction.kpis)


def test_public_view_exposes_only_kpis():
    """Red no longer self-labels — the extraction carries nothing but the KPI
    list, so blue structurally cannot see any truth signal."""
    ground_truth = GroundTruth.model_validate_json(
        (EXAMPLE / "ground_truth.json").read_text()
    )
    red = BaselineRedAgent()
    extraction = red.extract("doc", ground_truth)
    public = extraction.public_view()
    assert set(public.model_dump().keys()) == {"kpis"}
    assert public.kpis == extraction.kpis
