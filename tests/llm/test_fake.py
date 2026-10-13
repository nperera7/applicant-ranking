from __future__ import annotations

import pytest
from pydantic import ValidationError

from applicant_ranking.domain import Requirement, RoleCriteria
from applicant_ranking.llm.base import (
    AssessmentRequest,
    Evaluator,
    LLMBlockedError,
    LLMError,
    LLMOutputError,
    LLMUnavailableError,
)
from applicant_ranking.llm.fake import FakeEvaluator, default_assessment
from applicant_ranking.llm.schema import LLMAssessment


@pytest.fixture
def request_(sample_criteria: RoleCriteria, sample_resume_text: str) -> AssessmentRequest:
    return AssessmentRequest(criteria=sample_criteria, resume_text=sample_resume_text)


def test_default_assessment_covers_every_id_in_order(request_: AssessmentRequest) -> None:
    assessment = default_assessment(request_)
    criteria = request_.criteria
    assert [r.requirement_id for r in assessment.requirements] == [
        r.id for r in criteria.requirements]
    assert [k.knockout_id for k in assessment.knockouts] == [k.id for k in criteria.knockouts]
    assert {r.rating for r in assessment.requirements} == {"meets"}
    assert {k.result for k in assessment.knockouts} == {"met"}
    assert all(r.evidence == [] for r in assessment.requirements)
    assert all(k.evidence == [] for k in assessment.knockouts)
    assert assessment.overall_fit == 70
    assert assessment.injection_attempt_detected is False
    assert assessment.red_flags == []
    # schema-valid: survives the JSON round trip the real evaluator does
    assert LLMAssessment.model_validate_json(assessment.model_dump_json()) == assessment


def test_default_assessment_rating_and_fit(request_: AssessmentRequest) -> None:
    assessment = default_assessment(request_, rating="partial", overall_fit=33)
    assert {r.rating for r in assessment.requirements} == {"partial"}
    assert assessment.overall_fit == 33


@pytest.mark.parametrize(("rating", "fit"), [("great", 70), ("meets", 0), ("meets", 101)])
def test_default_assessment_rejects_invalid_values(request_: AssessmentRequest, rating: str,
                                                   fit: int) -> None:
    with pytest.raises(ValidationError):
        default_assessment(request_, rating=rating, overall_fit=fit)


def test_default_assessment_without_knockouts() -> None:
    criteria = RoleCriteria(role_id="role", title="Engineer", version=1,
                            requirements=[Requirement(id="python", description="Python")])
    assessment = default_assessment(AssessmentRequest(criteria=criteria, resume_text="x"))
    assert assessment.knockouts == []
    assert [r.requirement_id for r in assessment.requirements] == ["python"]


async def test_fake_records_requests_and_uses_default(request_: AssessmentRequest) -> None:
    fake = FakeEvaluator()
    result = await fake.assess(request_)
    second = request_.model_copy(update={"corrective_note": "fix it"})
    await fake.assess(second)
    assert fake.requests == [request_, second]
    assert result.assessment == default_assessment(request_)
    assert result.model_id == "fake-model"
    assert result.prompt_version == "p1-fake"
    assert result.finish_reason == "STOP"
    assert LLMAssessment.model_validate_json(result.raw_text) == result.assessment


async def test_fake_custom_ids_and_responder(request_: AssessmentRequest) -> None:
    fake = FakeEvaluator(lambda r: default_assessment(r, rating="exceeds", overall_fit=95),
                         model_id="gemini-test", prompt_version="p1-test")
    result = await fake.assess(request_)
    assert {r.rating for r in result.assessment.requirements} == {"exceeds"}
    assert (fake.model_id, fake.prompt_version) == ("gemini-test", "p1-test")
    assert (result.model_id, result.prompt_version) == ("gemini-test", "p1-test")


@pytest.mark.parametrize("error", [
    LLMBlockedError("blocked"), LLMOutputError("bad"), LLMUnavailableError("down"),
])
async def test_fake_raises_returned_errors(request_: AssessmentRequest, error: LLMError) -> None:
    fake = FakeEvaluator(lambda _: error)
    with pytest.raises(type(error)) as info:
        await fake.assess(request_)
    assert info.value is error
    assert fake.requests == [request_]


async def test_fake_scripted_responses_per_call(request_: AssessmentRequest) -> None:
    script: list[LLMAssessment | LLMError] = [
        LLMUnavailableError("try later"), default_assessment(request_, rating="partial")]
    fake = FakeEvaluator(lambda _: script.pop(0))
    with pytest.raises(LLMUnavailableError):
        await fake.assess(request_)
    result = await fake.assess(request_)
    assert result.assessment.requirements[0].rating == "partial"


async def test_fake_transcripts_in_order_then_error() -> None:
    fake = FakeEvaluator(transcripts=["page one text", "second resume"])
    assert await fake.transcribe([b"png-1"]) == "page one text"
    assert await fake.transcribe([b"png-2", b"png-3"]) == "second resume"
    with pytest.raises(LLMOutputError):
        await fake.transcribe([b"png-4"])
    assert fake.transcribe_calls == [[b"png-1"], [b"png-2", b"png-3"], [b"png-4"]]


async def test_fake_transcripts_list_is_copied() -> None:
    transcripts = ["only"]
    fake = FakeEvaluator(transcripts=transcripts)
    await fake.transcribe([b"png"])
    assert transcripts == ["only"]


async def test_fake_transcriber_callable() -> None:
    fake = FakeEvaluator(transcriber=lambda pages: f"{len(pages)} pages")
    assert await fake.transcribe([b"a", b"b"]) == "2 pages"
    failing = FakeEvaluator(transcriber=lambda _: LLMOutputError("illegible"))
    with pytest.raises(LLMOutputError, match="illegible"):
        await failing.transcribe([b"a"])


async def test_fake_is_an_evaluator_and_closes() -> None:
    fake = FakeEvaluator()
    assert isinstance(fake, Evaluator)
    assert fake.closed is False
    await fake.aclose()
    assert fake.closed is True
