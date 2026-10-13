from applicant_ranking.config import TierThresholds
from applicant_ranking.domain import CriterionResult, Tier
from applicant_ranking.scoring.compute import compute_score, score_band, tier_for


def _r(rating: str, weight: float) -> CriterionResult:
    return CriterionResult(requirement_id="x", description="d", importance="must_have",
                           weight=weight, rating=rating, model_rating=rating)


def test_weighted_score() -> None:
    # (3*0.85 + 1*0.45) / 4 = 0.75
    assert compute_score([_r("meets", 3), _r("partial", 1)]) == 75


def test_all_unknown_clamps_to_one() -> None:
    assert compute_score([_r("unknown", 3), _r("not_met", 1)]) == 1


def test_empty_is_none() -> None:
    assert compute_score([]) is None


def test_exceeds_everything_is_100() -> None:
    assert compute_score([_r("exceeds", 3), _r("exceeds", 1)]) == 100


def test_tiers() -> None:
    t = TierThresholds()
    assert [tier_for(s, t) for s in (80, 79, 65, 64, 45, 44)] == [
        Tier.A, Tier.B, Tier.B, Tier.C, Tier.C, Tier.D]


def test_bands() -> None:
    assert score_band(87) == (80, 89)
    assert score_band(90) == (90, 100)
    assert score_band(100) == (90, 100)
    assert score_band(5) == (1, 9)
    assert score_band(10) == (10, 19)
