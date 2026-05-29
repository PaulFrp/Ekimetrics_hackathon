"""Quota semantics: coverage counts distinct GT facts, additions count
distinct hallucinations. Duplicating real or fake KPIs is allowed but does
not game either bound. Truth (correct / value / metadata / fabricated) is
derived from the ground truth by the scorer — red declares nothing."""
from __future__ import annotations

from shared.scoring import (
    classify_kpi,
    distinct_hallucination_count,
    gt_coverage_count,
    quota_ok,
    score_match,
)
from shared.types import (
    BlueJudgment,
    GroundTruth,
    HallucinationType,
    KPI,
    RedExtraction,
    Verdict,
)


def _gt(n: int) -> GroundTruth:
    return GroundTruth(
        kpis=[
            KPI(id=i, name=f"k{i}", value=float(i), unit="x", period="2020", scope="S")
            for i in range(n)
        ]
    )


def _real(kpi_id: int, gt_index: int) -> KPI:
    """A red KPI that faithfully reproduces GT row `gt_index`."""
    return KPI(
        id=kpi_id, name=f"k{gt_index}", value=float(gt_index),
        unit="x", period="2020", scope="S",
    )


def _fake(kpi_id: int, tag: int) -> KPI:
    return KPI(
        id=kpi_id, name=f"fake{tag}", value=-1.0 - tag,
        unit="x", period="2099", scope="S",
    )


# ─────────────────────── classify_kpi ───────────────────────


def test_classify_correct_when_all_fields_match():
    gt = _gt(3)
    assert classify_kpi(_real(99, 1), gt) is None


def test_classify_value_when_only_value_differs():
    gt = _gt(3)
    kpi = KPI(id=99, name="k1", value=999.0, unit="x", period="2020", scope="S")
    assert classify_kpi(kpi, gt) == HallucinationType.VALUE


def test_classify_metadata_when_value_present_but_context_differs():
    gt = _gt(3)
    # value 2.0 exists in GT (row 2), but no GT row carries this metadata.
    kpi = KPI(id=99, name="k1", value=2.0, unit="x", period="2099", scope="S")
    assert classify_kpi(kpi, gt) == HallucinationType.METADATA


def test_classify_fabricated_when_neither_value_nor_metadata_present():
    gt = _gt(3)
    kpi = KPI(id=99, name="zzz", value=999.0, unit="q", period="2099", scope="Z")
    assert classify_kpi(kpi, gt) == HallucinationType.FABRICATED


def test_classify_value_takes_precedence_over_metadata():
    """Metadata matches one row and the value matches another → VALUE wins."""
    gt = _gt(3)
    kpi = KPI(id=99, name="k1", value=2.0, unit="x", period="2020", scope="S")
    assert classify_kpi(kpi, gt) == HallucinationType.VALUE


# ─────────────────────── quota ───────────────────────


def test_duplicating_one_real_does_not_reach_coverage_floor():
    """Spamming a single real GT KPI 60× covers only 1 distinct GT row."""
    gt = _gt(100)
    kpis = [_real(i, 0) for i in range(60)]  # 60 copies of GT row 0
    ext = RedExtraction(kpis=kpis)
    assert gt_coverage_count(ext, gt) == 1
    assert not quota_ok(ext, gt)


def test_fifty_distinct_reals_meets_floor():
    gt = _gt(100)
    kpis = [_real(i, i) for i in range(50)]  # 50 distinct GT rows
    ext = RedExtraction(kpis=kpis)
    assert gt_coverage_count(ext, gt) == 50
    assert quota_ok(ext, gt)


def test_duplicate_reals_allowed_on_top_of_a_valid_base():
    """Doubling real KPIs is fine — base of 50 distinct + duplicates stays legal."""
    gt = _gt(100)
    base = [_real(i, i) for i in range(50)]
    dups = [_real(100 + i, i) for i in range(50)]  # duplicate the same 50
    ext = RedExtraction(kpis=base + dups)
    assert gt_coverage_count(ext, gt) == 50
    assert quota_ok(ext, gt)


def test_doubling_hallucinations_does_not_breach_addition_cap():
    """25 distinct fakes duplicated many times = still 25 distinct fakes."""
    gt = _gt(100)
    reals = [_real(i, i) for i in range(50)]
    fakes = [_fake(1000 + i, i % 25) for i in range(200)]  # 25 distinct, 200 entries
    ext = RedExtraction(kpis=reals + fakes)
    assert distinct_hallucination_count(ext, gt) == 25
    assert quota_ok(ext, gt)


def test_twenty_six_distinct_fakes_breaches_addition_cap():
    gt = _gt(100)
    reals = [_real(i, i) for i in range(50)]
    fakes = [_fake(1000 + i, i) for i in range(26)]  # 26 distinct fakes
    ext = RedExtraction(kpis=reals + fakes)
    assert distinct_hallucination_count(ext, gt) == 26
    assert not quota_ok(ext, gt)


def test_quota_breach_penalizes_red_score():
    """The duplicate-real loophole is caught by the scorer: coverage of 1
    triggers an additive quota penalty with a positive amount."""
    gt = _gt(100)
    kpis = [_real(i, 0) for i in range(60)]  # coverage 1 → quota breach
    ext = RedExtraction(kpis=kpis)
    judgment = BlueJudgment(verdicts={k.id: Verdict.CORRECT for k in kpis})
    result = score_match(ext, judgment, gt)
    quota = next(p for p in result["red_penalties"] if p["name"] == "quota")
    assert quota["penalty"] > 0
