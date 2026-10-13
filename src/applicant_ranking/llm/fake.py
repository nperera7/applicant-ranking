"""Scripted evaluator for tests (never used by the CLI or service)."""

from __future__ import annotations

from collections.abc import Callable

from applicant_ranking.domain import TokenUsage
from applicant_ranking.llm.base import AssessmentRequest, LLMCallResult, LLMError, LLMOutputError
from applicant_ranking.llm.schema import KnockoutAssessment, LLMAssessment, RequirementAssessment

Responder = Callable[[AssessmentRequest], LLMAssessment | LLMError]
Transcriber = Callable[[list[bytes]], str | LLMError]


class FakeEvaluator:
    """``responder`` maps a request to an assessment (or an LLMError to raise).
    ``default_assessment(criteria, rating="meets")`` builds a valid assessment that rates
    every requirement and quotes nothing. Records every request in ``requests``.

    Without a responder every request gets ``default_assessment(request)``.
    ``transcribe`` returns the scripted ``transcripts`` in order (LLMOutputError once they
    run out), or delegates to ``transcriber`` when one is given (it may return an
    LLMError to raise). Page-image lists are recorded in ``transcribe_calls``.
    """

    def __init__(self, responder: Responder | None = None, *,
                 transcripts: list[str] | None = None,
                 model_id: str = "fake-model", prompt_version: str = "p1-fake",
                 transcriber: Transcriber | None = None) -> None:
        self.model_id = model_id
        self.prompt_version = prompt_version
        self._responder: Responder = responder or default_assessment
        self._transcripts = list(transcripts or [])
        self._transcriber = transcriber
        self.requests: list[AssessmentRequest] = []
        self.transcribe_calls: list[list[bytes]] = []
        self.closed = False

    async def assess(self, request: AssessmentRequest) -> LLMCallResult:
        self.requests.append(request)
        outcome = self._responder(request)
        if isinstance(outcome, LLMError):
            raise outcome
        return LLMCallResult(
            assessment=outcome,
            raw_text=outcome.model_dump_json(),
            model_id=self.model_id,
            model_version=self.model_id,
            prompt_version=self.prompt_version,
            finish_reason="STOP",
            usage=TokenUsage(),
            generation_config={"model": self.model_id, "prompt_version": self.prompt_version,
                               "fake": True},
        )

    async def transcribe(self, page_images: list[bytes]) -> str:
        self.transcribe_calls.append(list(page_images))
        if self._transcriber is not None:
            outcome = self._transcriber(list(page_images))
            if isinstance(outcome, LLMError):
                raise outcome
            return outcome
        if not self._transcripts:
            raise LLMOutputError("FakeEvaluator has no scripted transcript left")
        return self._transcripts.pop(0)

    async def aclose(self) -> None:
        self.closed = True


def default_assessment(request: AssessmentRequest, rating: str = "meets",
                       overall_fit: int = 70) -> LLMAssessment:
    """Every requirement rated ``rating`` and every knockout "met", in criteria order,
    with no evidence. Raises pydantic ValidationError for an invalid rating or fit."""
    criteria = request.criteria
    return LLMAssessment.model_validate({
        "requirements": [
            RequirementAssessment.model_validate({
                "requirement_id": req.id, "evidence": [], "rating": rating,
                "note": f"Scripted rating: {rating}.",
            })
            for req in criteria.requirements
        ],
        "knockouts": [
            KnockoutAssessment(knockout_id=ko.id, evidence=[], result="met",
                               note="Scripted result: met.")
            for ko in criteria.knockouts
        ],
        "strengths": [],
        "gaps": [],
        "red_flags": [],
        "injection_attempt_detected": False,
        "summary": "Scripted assessment for tests.",
        "confidence": "high",
        "overall_fit": overall_fit,
    })
