"""Measure the ranking against what recruiters actually did on past postings, before any
live use.

A candidate is a positive if they ever reached one of the ``positive_stage_names`` (from
``stageChanges`` or the current stage, matched case-insensitively to Lever stage names) or
were archived with a "hired"-type archive reason. Metrics on ranked evaluations only:
- AUC (Mann-Whitney U, ties count 0.5), None if no positives or no negatives;
- precision@k for k in (5, 10, 20) where k <= n;
- positive rate per tier (count, rate);
- mean and median rank of positives;
- n, positives, unranked count.
Caveat (reported): past recruiter decisions carry their own biases; agreement is not
fairness. Run the adverse-impact export alongside.

Ranks are 1..n by score, highest first; equal scores keep the order the evaluations were
given in (the ranking report's order), so precision@k and the positives' ranks match
what a recruiter sees. AUC is order-independent (ties count half).
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Sequence

from pydantic import BaseModel, Field

from applicant_ranking.domain import Evaluation, RankingReport, RunMode, Tier
from applicant_ranking.lever.client import LeverClient
from applicant_ranking.lever.models import ArchiveReason, Opportunity, Stage
from applicant_ranking.pipeline import Ranker

logger = logging.getLogger(__name__)

PRECISION_KS: tuple[int, ...] = (5, 10, 20)
SMALL_SAMPLE = 30

BIAS_CAVEAT = (
    "Past recruiter decisions carry their own biases; agreement with them is not fairness. "
    "Run the adverse-impact audit export (export-audit) alongside this backtest."
)
LABEL_CAVEAT = (
    "Labels come from Lever stage history: candidates who were never reviewed, withdrew, or "
    "were archived because the role was filled count as negatives."
)
HINDSIGHT_CAVEAT = (
    "The ranking uses today's criteria, model and prompt; recruiters decided with what they "
    "knew then. Evaluation-mode output is not for hiring decisions."
)


class Outcome(BaseModel):
    opportunity_id: str
    positive: bool
    hired: bool = False
    reached_stages: list[str] = Field(default_factory=list)


class BacktestMetrics(BaseModel):
    n: int
    positives: int
    unranked: int = 0
    auc: float | None = None
    precision_at: dict[int, float] = Field(default_factory=dict)
    tier_positive_rate: dict[str, tuple[int, float]] = Field(default_factory=dict)
    mean_rank_positive: float | None = None
    median_rank_positive: float | None = None


class BacktestReport(BaseModel):
    posting_id: str
    positive_stage_names: list[str]
    metrics: BacktestMetrics
    ranking: RankingReport
    caveats: list[str] = Field(default_factory=list)


def _norm(name: str) -> str:
    return " ".join(name.split()).casefold()


def unmatched_stage_names(stages: Sequence[Stage], positive_stage_names: Sequence[str],
                          ) -> list[str]:
    """The positive stage names that match no Lever stage (case-insensitive)."""
    known = {_norm(stage.text) for stage in stages}
    return [name for name in positive_stage_names if _norm(name) not in known]


def label_outcomes(opportunities: Sequence[Opportunity], *, stages: Sequence[Stage],
                   archive_reasons: Sequence[ArchiveReason],
                   positive_stage_names: Sequence[str]) -> dict[str, Outcome]:
    wanted = {_norm(name) for name in positive_stage_names if name.strip()}
    positive_ids = {stage.id for stage in stages if _norm(stage.text) in wanted}
    stage_text = {stage.id: stage.text or stage.id for stage in stages}
    hired_reasons = {reason.id for reason in archive_reasons
                     if (reason.type or "").strip().casefold() == "hired"}

    outcomes: dict[str, Outcome] = {}
    for opp in opportunities:
        reached_ids = [change.toStageId for change in opp.stageChanges if change.toStageId]
        if opp.stage:
            reached_ids.append(opp.stage)
        reached_ids = list(dict.fromkeys(reached_ids))

        reason_ids = {opp.archive_reason_id}
        app = opp.application
        if app is not None and app.archived is not None:
            reason_ids.add(app.archived.reason)
        hired = bool(reason_ids & hired_reasons)

        outcomes[opp.id] = Outcome(
            opportunity_id=opp.id,
            positive=hired or any(stage_id in positive_ids for stage_id in reached_ids),
            hired=hired,
            reached_stages=[stage_text.get(stage_id, stage_id) for stage_id in reached_ids],
        )
    return outcomes


def auc_mann_whitney(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """P(score of a random positive > score of a random negative), ties counting 0.5."""
    n_pos = sum(1 for label in labels if label)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[start]]:
            end += 1
        average = (start + end) / 2 + 1  # 1-based average rank of the tie group
        for k in range(start, end + 1):
            ranks[order[k]] = average
        start = end + 1
    rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    u_statistic = rank_sum - n_pos * (n_pos + 1) / 2
    return u_statistic / (n_pos * n_neg)


def compute_metrics(evaluations: Sequence[Evaluation],
                    outcomes: dict[str, Outcome]) -> BacktestMetrics:
    """Evaluations without an outcome label are left out (and logged); duplicates of an
    opportunity count once (first wins)."""
    seen: set[str] = set()
    ranked: list[tuple[Evaluation, bool]] = []
    unranked = 0
    unlabelled = 0
    for evaluation in evaluations:
        if evaluation.opportunity_id in seen:
            continue
        seen.add(evaluation.opportunity_id)
        if not evaluation.is_ranked:
            unranked += 1
            continue
        outcome = outcomes.get(evaluation.opportunity_id)
        if outcome is None:
            unlabelled += 1
            continue
        ranked.append((evaluation, outcome.positive))
    if unlabelled:
        logger.warning("%d ranked evaluations have no outcome label and were left out",
                       unlabelled)

    ranked.sort(key=lambda item: -(item[0].score or 0))  # stable: ties keep input order
    labels = [positive for _, positive in ranked]
    n = len(ranked)
    positives = sum(labels)

    tier_counts: dict[str, list[int]] = {}
    for evaluation, positive in ranked:
        key = evaluation.tier.value if evaluation.tier is not None else "none"
        counts = tier_counts.setdefault(key, [0, 0])
        counts[0] += 1
        counts[1] += int(positive)
    tier_order = [t.value for t in Tier] + ["none"]
    tier_rate = {key: (tier_counts[key][0], tier_counts[key][1] / tier_counts[key][0])
                 for key in tier_order if key in tier_counts}

    positive_ranks = [rank for rank, label in enumerate(labels, 1) if label]
    return BacktestMetrics(
        n=n,
        positives=positives,
        unranked=unranked,
        auc=auc_mann_whitney([float(e.score or 0) for e, _ in ranked], labels),
        precision_at={k: sum(labels[:k]) / k for k in PRECISION_KS if k <= n},
        tier_positive_rate=tier_rate,
        mean_rank_positive=statistics.fmean(positive_ranks) if positive_ranks else None,
        median_rank_positive=float(statistics.median(positive_ranks)) if positive_ranks
        else None,
    )


async def run_backtest(ranker: Ranker, lever: LeverClient, posting_id: str, *,
                       positive_stage_names: Sequence[str], limit: int | None = None,
                       force: bool = False) -> BacktestReport:
    """rank_posting in EVALUATION mode with include_archived=True, then label outcomes
    from the same opportunities (re-listed with archived=None) and compute metrics.

    Stage names are checked before any (paid) model call: ValueError if none is given
    or none matches a Lever stage; a partial match is reported as a caveat."""
    names = [name for name in positive_stage_names if name.strip()]
    if not names:
        raise ValueError("at least one positive stage name is required")
    stages = await lever.list_stages()
    missing = unmatched_stage_names(stages, names)
    if len(missing) == len(names):
        available = ", ".join(sorted({stage.text for stage in stages if stage.text}))
        raise ValueError(f"no positive stage name matches a Lever stage: {', '.join(missing)}"
                         f" (available: {available or 'none'})")
    archive_reasons = await lever.list_archive_reasons()

    ranking = await ranker.rank_posting(posting_id, mode=RunMode.EVALUATION, limit=limit,
                                        include_archived=True, force=force)
    ranked_ids = {row.evaluation.opportunity_id for row in ranking.rows}
    opportunities = [opp async for opp in lever.list_opportunities(posting_id=posting_id,
                                                                   archived=None)
                     if opp.id in ranked_ids]
    outcomes = label_outcomes(opportunities, stages=stages, archive_reasons=archive_reasons,
                              positive_stage_names=names)
    evaluations = [row.evaluation for row in ranking.rows]
    metrics = compute_metrics(evaluations, outcomes)

    caveats = [BIAS_CAVEAT, LABEL_CAVEAT, HINDSIGHT_CAVEAT]
    if missing:
        caveats.append(f"Positive stage name(s) not found in Lever and ignored: "
                       f"{', '.join(missing)}.")
    unlabelled = sum(1 for e in evaluations if e.is_ranked and e.opportunity_id not in outcomes)
    if unlabelled:
        caveats.append(f"{unlabelled} ranked application(s) were not found when re-listing "
                       "the posting and are left out of the metrics.")
    if metrics.unranked:
        caveats.append(f"{metrics.unranked} application(s) could not be ranked (errors or "
                       "skipped) and are left out of the metrics.")
    if metrics.auc is None:
        caveats.append("AUC is undefined: the ranked set has no positives or no negatives.")
    if metrics.n < SMALL_SAMPLE:
        caveats.append(f"Small sample (n={metrics.n}): the metrics are noisy; treat them as "
                       "indicative only.")
    if limit is not None:
        caveats.append(f"Limited to {limit} application(s); the sample may not be "
                       "representative of the posting.")

    return BacktestReport(posting_id=posting_id, positive_stage_names=names, metrics=metrics,
                          ranking=ranking, caveats=caveats)
