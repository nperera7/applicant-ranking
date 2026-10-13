"""Deterministic score from per-requirement ratings (the model never sets the score).

score = round(100 * sum(weight_i * value(rating_i)) / sum(weight_i)), clamped to 1..100.
Ratings after evidence grounding are used (an ungrounded "meets" became "unknown").
"""

from __future__ import annotations

from collections.abc import Sequence

from applicant_ranking.config import TierThresholds
from applicant_ranking.domain import CriterionResult, Tier

RATING_VALUES: dict[str, float] = {
    "exceeds": 1.0,
    "meets": 0.85,
    "partial": 0.45,
    "not_met": 0.0,
    "unknown": 0.0,
}


def compute_score(results: Sequence[CriterionResult]) -> int | None:
    """None when there are no results or all weights are zero."""
    total_weight = sum(r.weight for r in results)
    if not results or total_weight <= 0:
        return None
    earned = sum(r.weight * RATING_VALUES[r.rating] for r in results)
    return max(1, min(100, round(100 * earned / total_weight)))


def tier_for(score: int, thresholds: TierThresholds) -> Tier:
    if score >= thresholds.a:
        return Tier.A
    if score >= thresholds.b:
        return Tier.B
    if score >= thresholds.c:
        return Tier.C
    return Tier.D


def score_band(score: int) -> tuple[int, int]:
    """Decile band for tags: 87 -> (80, 89); 100 -> (90, 100); 5 -> (1, 9)."""
    if score >= 90:
        return (90, 100)
    low = (score // 10) * 10
    return (max(1, low), low + 9)
