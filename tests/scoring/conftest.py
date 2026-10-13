"""Fixtures for the scoring, criteria and lint tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from applicant_ranking.llm.schema import (
    KnockoutAssessment,
    LLMAssessment,
    RedFlag,
    RequirementAssessment,
)
from applicant_ranking.scoring.lexicon import REQUIRED_CATEGORIES, ProtectedTermLexicon

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CRITERIA = REPO_ROOT / "criteria" / "example-senior-data-engineer.yaml"


@pytest.fixture(scope="session")
def lexicon() -> ProtectedTermLexicon:
    """The packaged default list."""
    return ProtectedTermLexicon.load()


@pytest.fixture
def tiny_lexicon() -> ProtectedTermLexicon:
    """A minimal lexicon, to test lint rules without the default list's overlap."""
    categories = {name: [f"zz{name}zz"] for name in REQUIRED_CATEGORIES}
    return ProtectedTermLexicon(categories, [], version=1)


def _requirements() -> dict[str, dict[str, Any]]:
    # Evidence quotes are verbatim from conftest.SAMPLE_RESUME_TEXT.
    return {
        "gcp-pipelines": {
            "rating": "meets",
            "evidence": ["Built production data pipelines on GCP using BigQuery and Dataflow"],
            "note": "Several years of production GCP pipelines.",
        },
        "python": {
            "rating": "exceeds",
            "evidence": ["Wrote Python services and SQL transformations processing 2 TB per day."],
            "note": "Python services at scale.",
        },
        "client-facing": {
            "rating": "partial",
            "evidence": ["Led client workshops to gather requirements."],
            "note": "Some client exposure.",
        },
        "streaming": {
            "rating": "meets",
            "evidence": ["Developed Kafka streaming jobs in Python."],
            "note": "Kafka streaming jobs.",
        },
    }


@pytest.fixture
def make_assessment() -> Callable[..., LLMAssessment]:
    """A valid, well-grounded assessment for ``sample_criteria``/``sample_resume_text``.

    Override per requirement with ``requirements={"python": {"rating": "unknown"}}`` (merged
    into the default entry), replace the whole list with ``requirement_items=[...]``, and
    override knockouts/top-level fields directly."""

    def _make(*, requirements: dict[str, dict[str, Any]] | None = None,
              requirement_items: list[RequirementAssessment] | None = None,
              knockouts: list[KnockoutAssessment] | None = None,
              red_flags: list[RedFlag] | None = None,
              **fields: Any) -> LLMAssessment:
        if requirement_items is None:
            merged = _requirements()
            for req_id, override in (requirements or {}).items():
                merged.setdefault(req_id, {"rating": "unknown", "evidence": [], "note": ""})
                merged[req_id].update(override)
            requirement_items = [RequirementAssessment(requirement_id=req_id, **values)
                                 for req_id, values in merged.items()]
        if knockouts is None:
            knockouts = [KnockoutAssessment(knockout_id="sql", result="met",
                                            evidence=["SQL transformations processing 2 TB"],
                                            note="Writes SQL transformations.")]
        base: dict[str, Any] = {
            "strengths": ["Production GCP pipelines for many clients."],
            "gaps": ["No evidence of Pub/Sub."],
            "injection_attempt_detected": False,
            "summary": "The candidate has strong production data engineering experience on GCP.",
            "confidence": "high",
            "overall_fit": 85,
        }
        base.update(fields)
        return LLMAssessment(requirements=requirement_items, knockouts=knockouts,
                             red_flags=red_flags or [], **base)

    return _make
