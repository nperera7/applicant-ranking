from __future__ import annotations

import random
from collections.abc import AsyncIterator, Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from applicant_ranking.backtest import (
    BIAS_CAVEAT,
    Outcome,
    auc_mann_whitney,
    compute_metrics,
    label_outcomes,
    run_backtest,
    unmatched_stage_names,
)
from applicant_ranking.domain import Evaluation, RankingReport, RankingRow, RunMode, Status, Tier
from applicant_ranking.lever.client import LeverClient
from applicant_ranking.lever.models import ArchiveReason, Opportunity, Stage

STAGES = [
    Stage(id="s-new", text="New applicant"),
    Stage(id="s-screen", text="Phone Screen"),
    Stage(id="s-onsite", text="On-site  Interview"),
    Stage(id="s-offer", text="Offer"),
]
REASONS = [
    ArchiveReason(id="r-hired", text="Hired", type="hired"),
    ArchiveReason(id="r-other", text="Underqualified", type="non-hired"),
    ArchiveReason(id="r-hired-2", text="Hired (contractor)", type="HIRED"),
]


def opp(opp_id: str, **data: Any) -> Opportunity:
    return Opportunity.model_validate({"id": opp_id, **data})


def ev(opp_id: str, score: int | None, tier: Tier | None = None,
       status: Status = Status.SCORED) -> Evaluation:
    return Evaluation(evaluation_id=f"ev-{opp_id}", mode=RunMode.EVALUATION,
                      opportunity_id=opp_id, status=status, score=score, tier=tier)


# --------------------------------------------------------------------------- labels


def test_label_outcomes() -> None:
    opportunities = [
        # went to phone screen and back to new applicant: still a positive
        opp("o1", stage="s-new", stageChanges=[{"toStageId": "s-screen"},
                                                {"toStageId": "s-new"}]),
        # current stage only (no history)
        opp("o2", stage={"id": "s-onsite", "text": "On-site Interview"}),
        # archived as hired without reaching a positive stage
        opp("o3", stage="s-new", archived={"archivedAt": 1, "reason": "r-hired"}),
        # archived for another reason
        opp("o4", stage="s-new", archived={"archivedAt": "2", "reason": "r-other"}),
        # older shape: top-level archiveReason; "HIRED" type compared case-insensitively
        opp("o5", archivedAt=3, archiveReason="r-hired-2"),
        # application-level archive
        opp("o6", applications=[{"id": "a6", "archived": {"archivedAt": 4,
                                                          "reason": "r-hired"}}]),
        # unknown stage id is kept by id
        opp("o7", stage="s-gone", stageChanges=[{"toStageId": "s-new"}]),
        opp("o8"),
    ]
    out = label_outcomes(opportunities, stages=STAGES, archive_reasons=REASONS,
                         positive_stage_names=["phone screen", "ON-SITE INTERVIEW", " "])

    assert {k: (v.positive, v.hired) for k, v in out.items()} == {
        "o1": (True, False), "o2": (True, False), "o3": (True, True), "o4": (False, False),
        "o5": (True, True), "o6": (True, True), "o7": (False, False), "o8": (False, False),
    }
    assert out["o1"].reached_stages == ["Phone Screen", "New applicant"]
    assert out["o2"].reached_stages == ["On-site  Interview"]
    assert out["o7"].reached_stages == ["New applicant", "s-gone"]
    assert out["o8"].reached_stages == []


def test_label_outcomes_offer_stage_only() -> None:
    out = label_outcomes([opp("o1", stage="s-screen"), opp("o2", stage="s-offer")],
                         stages=STAGES, archive_reasons=[], positive_stage_names=["Offer"])
    assert [out["o1"].positive, out["o2"].positive] == [False, True]


def test_unmatched_stage_names() -> None:
    assert unmatched_stage_names(STAGES, ["offer", "Hired", "phone  screen"]) == ["Hired"]


# -------------------------------------------------------------------------- metrics


def outcomes_for(labels: dict[str, bool]) -> dict[str, Outcome]:
    return {k: Outcome(opportunity_id=k, positive=v) for k, v in labels.items()}


def test_compute_metrics_hand_computed() -> None:
    evaluations = [
        ev("e1", 90, Tier.A), ev("e2", 80, Tier.A), ev("e3", 80, Tier.A),
        ev("e4", 70, Tier.B), ev("e5", 60, Tier.C), ev("e6", 50, Tier.C),
        ev("e7", 40, Tier.D),
        ev("e8", None, status=Status.ERROR),
        ev("e9", None, status=Status.SKIPPED),
        ev("e1", 10, Tier.D),  # duplicate opportunity: counted once (first wins)
        ev("e10", 99, Tier.A),  # ranked but no outcome label: left out
    ]
    outcomes = outcomes_for({"e1": True, "e2": False, "e3": True, "e4": False, "e5": False,
                             "e6": True, "e7": False, "e8": True, "e9": False})
    m = compute_metrics(evaluations, outcomes)

    assert (m.n, m.positives, m.unranked) == (7, 3, 2)
    # Positives {90, 80, 50} vs negatives {80, 70, 60, 40}:
    # 90 beats 4; 80 beats 3 and ties 1 (3.5); 50 beats 1 -> 8.5 / 12
    assert m.auc == pytest.approx(8.5 / 12)
    # Order: e1 90 P, e2 80 N, e3 80 P (input order breaks the tie), e4, e5 -> 2 of 5
    assert m.precision_at == {5: pytest.approx(0.4)}
    assert m.tier_positive_rate == {
        "A": (3, pytest.approx(2 / 3)), "B": (1, 0.0), "C": (2, 0.5), "D": (1, 0.0)}
    # Positive ranks 1, 3, 6
    assert m.mean_rank_positive == pytest.approx(10 / 3)
    assert m.median_rank_positive == 3.0


def test_compute_metrics_precision_keys_and_even_median() -> None:
    evaluations = [ev(f"o{i}", 100 - i, Tier.A) for i in range(25)]
    positives = {"o0", "o3", "o9", "o19"}
    m = compute_metrics(evaluations, outcomes_for({e.opportunity_id: e.opportunity_id in
                                                   positives for e in evaluations}))
    assert m.precision_at == {5: pytest.approx(2 / 5), 10: pytest.approx(3 / 10),
                              20: pytest.approx(4 / 20)}
    assert m.mean_rank_positive == pytest.approx((1 + 4 + 10 + 20) / 4)
    assert m.median_rank_positive == pytest.approx(7.0)
    # 21 negatives; positives beat those ranked below them: o0 21, o3 19, o9 14, o19 5
    assert m.auc == pytest.approx(59 / 84)


@pytest.mark.parametrize("label", [True, False])
def test_auc_undefined_without_both_classes(label: bool) -> None:
    evaluations = [ev("a", 90), ev("b", 50)]
    m = compute_metrics(evaluations, outcomes_for({"a": label, "b": label}))
    assert m.auc is None
    if label:
        assert m.mean_rank_positive == 1.5
    else:
        assert m.mean_rank_positive is None
        assert m.median_rank_positive is None


def test_auc_all_tied_is_half() -> None:
    assert auc_mann_whitney([50, 50, 50, 50], [True, False, True, False]) == 0.5


def test_auc_perfect_and_inverted() -> None:
    assert auc_mann_whitney([9, 8, 2, 1], [True, True, False, False]) == 1.0
    assert auc_mann_whitney([9, 8, 2, 1], [False, False, True, True]) == 0.0


def test_auc_matches_pairwise_definition() -> None:
    rng = random.Random(1234)
    for _ in range(50):
        scores = [rng.randint(1, 10) for _ in range(rng.randint(2, 30))]
        labels = [rng.random() < 0.4 for _ in scores]
        pos = [s for s, lab in zip(scores, labels, strict=True) if lab]
        neg = [s for s, lab in zip(scores, labels, strict=True) if not lab]
        if not pos or not neg:
            assert auc_mann_whitney(scores, labels) is None
            continue
        pairs = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
        assert auc_mann_whitney(scores, labels) == pytest.approx(pairs / (len(pos) * len(neg)))


def test_compute_metrics_empty() -> None:
    m = compute_metrics([], {})
    assert (m.n, m.positives, m.unranked, m.auc, m.precision_at) == (0, 0, 0, None, {})


# ------------------------------------------------------------------------- backtest


class FakeRanker:
    def __init__(self, report: RankingReport) -> None:
        self.report = report
        self.calls: list[dict[str, Any]] = []

    async def rank_posting(self, posting_id: str, **kwargs: Any) -> RankingReport:
        self.calls.append({"posting_id": posting_id, **kwargs})
        return self.report


async def _aiter(items: Sequence[Opportunity]) -> AsyncIterator[Opportunity]:
    for item in items:
        yield item


def make_lever(opportunities: Sequence[Opportunity]) -> AsyncMock:
    lever = AsyncMock(spec=LeverClient)
    lever.list_stages.return_value = STAGES
    lever.list_archive_reasons.return_value = REASONS
    lever.list_opportunities = MagicMock(side_effect=lambda **kw: _aiter(opportunities))
    return lever


def ranking(rows: list[RankingRow]) -> RankingReport:
    return RankingReport(posting_id="posting-1", mode=RunMode.EVALUATION, rows=rows)


async def test_run_backtest() -> None:
    rows = [RankingRow(rank=1, evaluation=ev("o1", 90, Tier.A)),
            RankingRow(rank=2, evaluation=ev("o2", 70, Tier.B)),
            RankingRow(rank=3, evaluation=ev("o3", 40, Tier.D)),
            RankingRow(rank=None, evaluation=ev("o4", None, status=Status.ERROR))]
    opportunities = [opp("o1", stage="s-onsite"), opp("o2", stage="s-new"),
                     opp("o3", archived={"archivedAt": 1, "reason": "r-hired"}),
                     opp("o4", stage="s-offer"), opp("other", stage="s-offer")]
    ranker = FakeRanker(ranking(rows))
    lever = make_lever(opportunities)

    result = await run_backtest(ranker, lever, "posting-1",  # type: ignore[arg-type]
                                positive_stage_names=["on-site interview", "Typo Stage"],
                                limit=50, force=True)

    assert ranker.calls == [{"posting_id": "posting-1", "mode": RunMode.EVALUATION,
                             "limit": 50, "include_archived": True, "force": True}]
    lever.list_opportunities.assert_called_once_with(posting_id="posting-1", archived=None)
    assert result.posting_id == "posting-1"
    assert result.positive_stage_names == ["on-site interview", "Typo Stage"]
    assert result.ranking is ranker.report
    m = result.metrics
    assert (m.n, m.positives, m.unranked) == (3, 2, 1)
    # positives 90 and 40 vs negative 70 -> (1 + 0) / 2
    assert m.auc == pytest.approx(0.5)
    assert m.mean_rank_positive == pytest.approx(2.0)

    caveats = " ".join(result.caveats)
    assert result.caveats[0] == BIAS_CAVEAT
    assert "Typo Stage" in caveats
    assert "1 application(s) could not be ranked" in caveats
    assert "Small sample (n=3)" in caveats
    assert "Limited to 50" in caveats


async def test_run_backtest_reports_unlabelled_and_undefined_auc() -> None:
    rows = [RankingRow(rank=1, evaluation=ev("o1", 90, Tier.A)),
            RankingRow(rank=2, evaluation=ev("gone", 80, Tier.A))]
    lever = make_lever([opp("o1", stage="s-new")])
    result = await run_backtest(FakeRanker(ranking(rows)), lever,  # type: ignore[arg-type]
                                "posting-1", positive_stage_names=["Offer"])
    caveats = " ".join(result.caveats)
    assert "1 ranked application(s) were not found" in caveats
    assert "AUC is undefined" in caveats
    assert "Limited to" not in caveats
    assert (result.metrics.n, result.metrics.positives) == (1, 0)


@pytest.mark.parametrize("names", [[], ["  "], ["Nope", "Also nope"]])
async def test_run_backtest_checks_stage_names_before_ranking(names: list[str]) -> None:
    ranker = FakeRanker(ranking([]))
    lever = make_lever([])
    with pytest.raises(ValueError, match="stage"):
        await run_backtest(ranker, lever, "posting-1",  # type: ignore[arg-type]
                           positive_stage_names=names)
    assert ranker.calls == []
    lever.list_opportunities.assert_not_called()
