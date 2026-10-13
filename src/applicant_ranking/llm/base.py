"""The evaluator interface the pipeline depends on. ``GeminiEvaluator`` is the real one;
``FakeEvaluator`` scripts responses for tests. Nothing outside ``llm/`` imports the SDK."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from applicant_ranking.domain import RoleCriteria, TokenUsage
from applicant_ranking.llm.schema import LLMAssessment


class AssessmentRequest(BaseModel):
    criteria: RoleCriteria
    resume_text: str  # normalised + redacted; the only resume content sent
    # Set on the single corrective retry after output validation failed: a short,
    # system-authored explanation of what to fix (never resume text).
    corrective_note: str | None = None
    # Cache key component; evaluator implementations may log it.
    evaluation_id: str | None = None


class LLMCallResult(BaseModel):
    assessment: LLMAssessment
    raw_text: str = ""
    model_id: str
    model_version: str | None = None  # response.model_version
    prompt_version: str
    finish_reason: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    generation_config: dict = Field(default_factory=dict)  # for the audit log (no secrets)


class LLMError(RuntimeError):
    """Base class. ``retryable`` tells the pipeline whether a later re-run may succeed."""

    retryable: bool = False

    def __init__(self, message: str, *, retryable: bool | None = None,
                 code: int | None = None, status: str | None = None) -> None:
        super().__init__(message)
        if retryable is not None:
            self.retryable = retryable
        self.code = code
        self.status = status


class LLMBlockedError(LLMError):
    """Prompt or response blocked (prompt_feedback.block_reason, or finish_reason
    SAFETY/PROHIBITED_CONTENT/SPII/BLOCKLIST/RECITATION)."""


class LLMOutputError(LLMError):
    """Response present but unusable: MAX_TOKENS truncation, invalid JSON, schema mismatch."""


class LLMUnavailableError(LLMError):
    """Quota/server/network failure after the SDK's own retries (429/5xx/timeouts)."""

    retryable = True


class LLMConfigError(LLMError):
    """4xx that is our fault (400 INVALID_ARGUMENT, 403 permission, 404 model)."""


@runtime_checkable
class Evaluator(Protocol):
    model_id: str
    prompt_version: str

    async def assess(self, request: AssessmentRequest) -> LLMCallResult:
        """Score one resume against criteria. Raises LLMError subclasses."""
        ...

    async def transcribe(self, page_images: list[bytes]) -> str:
        """Vision step for scanned/image resumes: return the visible text verbatim
        (no judgement, no scoring). Raises LLMError subclasses."""
        ...

    async def aclose(self) -> None: ...
