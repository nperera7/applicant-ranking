"""Domain types shared across modules. This file is the contract between the subsystems:
Lever ingestion, resume extraction, the LLM call, scoring/validation, storage and write-back.

Design rules that these types encode:
- The service ranks; it never decides. There is no status meaning "rejected", and a
  knockout that looks unmet is a flag for a human (``KNOCKOUT_FLAGGED``), not a filter.
- The final score is computed in code from per-requirement ratings (``scoring.compute``),
  not taken from the model. The model's own holistic number is kept only as a diagnostic.
- Deterministic checks (hidden text, injection phrases, validators) can force human review
  regardless of what the model says.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class Tier(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class Status(StrEnum):
    """Outcome of one evaluation. Ordered by precedence when several apply (see ``worst``)."""

    SCORED = "scored"
    KNOCKOUT_FLAGGED = "knockout_flagged"  # a knockout looks unmet; a human must confirm
    NEEDS_REVIEW = "needs_review"  # a check forced human review (see review_reasons)
    ERROR = "error"  # could not be evaluated (no resume, LLM failure, ...)
    SKIPPED = "skipped"  # deliberately not evaluated (deleted, anonymized, no criteria, policy)

    @staticmethod
    def worst(*statuses: Status) -> Status:
        order = [Status.SCORED, Status.KNOCKOUT_FLAGGED, Status.NEEDS_REVIEW, Status.ERROR,
                 Status.SKIPPED]
        return max(statuses, key=order.index)


class RunMode(StrEnum):
    """How results may be used.

    - ``DRY_RUN``: full evaluation behind the policy/approval gates; write-back is planned
      and reported but never sent. The default.
    - ``LIVE``: as DRY_RUN, and the planned write-back is sent to Lever (also needs
      ``AR_WRITE_ENABLED=true``).
    - ``EVALUATION``: offline evaluation of the tool itself (criteria tuning, backtests,
      fairness checks). Gates are relaxed (unapproved criteria allowed), every output is
      watermarked "evaluation only", and write-back is impossible.
    """

    DRY_RUN = "dry_run"
    LIVE = "live"
    EVALUATION = "evaluation"


# ----------------------------------------------------------------------------------------
# Per-role criteria (authored by the hiring manager, approved by TA/HR; YAML on disk)
# ----------------------------------------------------------------------------------------

Importance = Literal["must_have", "nice_to_have"]
SLUG_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,39}$"


class Requirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=SLUG_PATTERN)
    description: str = Field(min_length=3, max_length=500)
    importance: Importance = "must_have"
    # Optional override; defaults: must_have=3, nice_to_have=1 (see effective_weight).
    weight: float | None = Field(default=None, ge=0.5, le=10)

    @property
    def effective_weight(self) -> float:
        if self.weight is not None:
            return self.weight
        return 3.0 if self.importance == "must_have" else 1.0


class Knockout(BaseModel):
    """A hard requirement that is flagged (never applied) when it looks unmet.

    ``justification`` documents why it is job-related; the linter requires it.
    Knockouts can only be judged "unmet" on explicit evidence in the resume.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=SLUG_PATTERN)
    description: str = Field(min_length=3, max_length=500)
    justification: str = Field(min_length=10, max_length=1000)


class ApprovedTerm(BaseModel):
    """A protected-term lexicon hit that Legal/HR allow in these criteria because it names
    the role's domain, not a person ("credit history data" for a lender's risk engineer).
    The linter reports a matching hit as a warning that names the approver instead of an
    error; nothing else changes (the model output scan is unaffected)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    term: str = Field(min_length=2, max_length=100)
    justification: str = Field(min_length=10, max_length=1000)
    approved_by: str = Field(min_length=2, max_length=200)
    approved_at: date | datetime


class RoleCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role_id: str = Field(pattern=SLUG_PATTERN)
    title: str = Field(min_length=2, max_length=200)
    version: int = Field(ge=1)
    posting_ids: list[str] = Field(default_factory=list)
    summary: str = Field(default="", max_length=3000)
    requirements: list[Requirement] = Field(min_length=1, max_length=25)
    knockouts: list[Knockout] = Field(default_factory=list, max_length=10)
    guardrails: list[str] = Field(default_factory=list, max_length=20)
    # "job_description" marks a skeleton drafted from the Lever posting, not yet curated.
    source: Literal["hiring_manager", "job_description"] = "hiring_manager"
    # Domain vocabulary the linter may let through (see ApprovedTerm). Part of the content
    # hash: it changes what the criteria may say.
    approved_terms: list[ApprovedTerm] = Field(default_factory=list, max_length=20)
    approved_by: str | None = None
    approved_at: datetime | None = None

    @property
    def approved(self) -> bool:
        """Signed off by TA/HR: a non-blank approver and an approval time. Whether that
        time is plausible needs a clock; see ``approval_issue``."""
        return bool(self.approved_by and self.approved_by.strip() and self.approved_at)

    def approval_issue(self, now: datetime) -> str | None:
        """Why the approval cannot be relied on at ``now``: not approved, or ``approved_at``
        in the future (the policy-file gate makes the same two checks). None when it can.
        A naive ``approved_at`` or ``now`` is read as UTC."""
        if not self.approved:
            return "not approved (approved_by and approved_at are required)"
        assert self.approved_at is not None
        approved_at = self.approved_at
        if approved_at.tzinfo is None:
            approved_at = approved_at.replace(tzinfo=UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        if approved_at > now:
            return f"approved_at {approved_at.isoformat()} is in the future"
        return None

    def content_hash(self) -> str:
        """sha256 over the canonical JSON of everything that affects evaluation.

        Approval metadata is excluded so approving criteria doesn't invalidate results.
        """
        payload = self.model_dump(mode="json", exclude={"approved_by", "approved_at"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()


# ----------------------------------------------------------------------------------------
# Flags: deterministic (system) and model-reported
# ----------------------------------------------------------------------------------------

FlagSeverity = Literal["info", "review"]


class FlagCode(StrEnum):
    # resume integrity / security (system)
    HIDDEN_TEXT = "hidden_text"
    INJECTION_PHRASE = "injection_phrase"
    UNICODE_TAG_PAYLOAD = "unicode_tag_payload"
    INVISIBLE_CHARACTERS = "invisible_characters"
    MIXED_SCRIPT = "mixed_script"
    HIDDEN_TEXT_CHECK_UNAVAILABLE = "hidden_text_check_unavailable"
    SCANNED_RESUME = "scanned_resume"
    TRUNCATED = "truncated"
    UNSUPPORTED_FILE = "unsupported_file"
    UNREADABLE_FILE = "unreadable_file"
    NO_RESUME = "no_resume"
    RESUME_FROM_PARSED_DATA = "resume_from_parsed_data"
    DUPLICATE_RESUME = "duplicate_resume"
    CONTACT_REUSE = "contact_reuse"
    JD_COPY = "jd_copy"
    # output validation (system)
    UNGROUNDED_EVIDENCE = "ungrounded_evidence"
    PROTECTED_TERM_IN_OUTPUT = "protected_term_in_output"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    MODEL_BLOCKED = "model_blocked"
    SCORE_DISAGREEMENT = "score_disagreement"
    LOW_CONFIDENCE = "low_confidence"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    # reported by the model
    MODEL_INJECTION_REPORT = "model_injection_report"
    MODEL_RED_FLAG = "model_red_flag"


class Flag(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: FlagCode
    source: Literal["system", "model"] = "system"
    # "review" forces Status.NEEDS_REVIEW; "info" is shown but changes nothing.
    severity: FlagSeverity = "info"
    detail: str = ""


# ----------------------------------------------------------------------------------------
# Resume processing results
# ----------------------------------------------------------------------------------------

FileKind = Literal["pdf", "docx", "doc", "rtf", "txt", "odt", "image", "unknown"]


class ExtractionResult(BaseModel):
    """Output of ``resumes.extract.extract_resume_sync`` / ``ResumeExtractor.extract``:
    visible text plus what was hidden."""

    kind: FileKind
    visible_text: str = ""
    hidden_snippets: list[str] = Field(default_factory=list)  # short samples, for the note
    hidden_char_count: int = 0
    pages: int | None = None
    # Image-only/scanned document: visible_text is (nearly) empty; use vision transcription.
    needs_vision: bool = False
    # PNG bytes of rendered pages (scans) or a re-encoded image, EXIF stripped. Never the PDF.
    page_images: list[bytes] = Field(default_factory=list, repr=False)
    flags: list[Flag] = Field(default_factory=list)
    error: str | None = None  # set when the file could not be read at all


class NormalizedText(BaseModel):
    """Output of ``resumes.normalize.normalize_text``."""

    text: str
    removed: dict[str, int] = Field(default_factory=dict)  # category -> characters removed
    tag_payload: str = ""  # decoded Unicode tag-character ("ASCII smuggling") content
    truncated: bool = False
    original_length: int = 0  # length of the raw input, before normalisation


class SecurityScan(BaseModel):
    """Output of ``resumes.security.scan_text``."""

    injection_phrases: list[str] = Field(default_factory=list)  # matched snippets
    mixed_script_words: list[str] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)


class RedactionResult(BaseModel):
    """Output of ``resumes.redact.redact``. The text is what the LLM sees."""

    text: str
    # Bracketless placeholder name ("CANDIDATE", "EMAIL", "REDACTED PERSONAL DETAIL", ...;
    # see redact.PLACEHOLDERS) -> number of replacements.
    counts: dict[str, int] = Field(default_factory=dict)
    version: str  # e.g. "redact-v1"; part of the evaluation cache key


class PreparedResume(BaseModel):
    """Everything known about one candidate's resume before the LLM call."""

    opportunity_id: str
    resume_id: str | None = None
    file_name: str | None = None
    file_hmac: str | None = None  # HMAC-SHA256 of the original bytes (audit)
    source: Literal["file", "parsed_data", "local_file", "none"] = "none"
    kind: FileKind = "unknown"
    redacted_text: str = ""  # the only resume text sent for scoring
    redacted_text_hmac: str | None = None
    fingerprint: str | None = None  # sha256 of normalised redacted text (duplicate check)
    redaction_version: str = ""
    needs_vision: bool = False
    page_images: list[bytes] = Field(default_factory=list, repr=False)
    flags: list[Flag] = Field(default_factory=list)
    error: str | None = None


# ----------------------------------------------------------------------------------------
# Final, validated evaluation
# ----------------------------------------------------------------------------------------

Rating = Literal["exceeds", "meets", "partial", "not_met", "unknown"]
KnockoutResult = Literal["met", "unmet", "unclear"]
Confidence = Literal["high", "medium", "low"]


class CriterionResult(BaseModel):
    requirement_id: str
    description: str
    importance: Importance
    weight: float
    rating: Rating
    model_rating: Rating  # before evidence grounding (may be downgraded to "unknown")
    evidence: list[str] = Field(default_factory=list)  # quotes found in the resume text
    ungrounded_evidence: list[str] = Field(default_factory=list)  # quotes not found
    note: str = ""
    points: float = 0.0  # weight * rating value


class KnockoutCheck(BaseModel):
    knockout_id: str
    description: str
    result: KnockoutResult
    evidence: list[str] = Field(default_factory=list)
    note: str = ""


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None


class ValidationReport(BaseModel):
    quotes_total: int = 0
    quotes_grounded: int = 0
    # Of quotes_grounded, how many matched only by the fuzzy (formatting-noise) rule.
    quotes_grounded_fuzzy: int = 0
    downgraded_requirements: list[str] = Field(default_factory=list)
    protected_term_hits: list[str] = Field(default_factory=list)  # "category:term"
    corrective_retries: int = 0
    # Model-written ids that matched no requirement or knockout: at most a sample is kept
    # (``scoring.validate.MAX_UNKNOWN_IDS_KEPT``), and the audit row stores only the count.
    unknown_ids_dropped: list[str] = Field(default_factory=list)
    unknown_ids_dropped_count: int = 0
    missing_ids_filled: list[str] = Field(default_factory=list)


class Evaluation(BaseModel):
    evaluation_id: str
    created_at: datetime = Field(default_factory=utcnow)
    mode: RunMode
    posting_id: str | None = None
    opportunity_id: str
    application_id: str | None = None
    resume_id: str | None = None
    candidate_name: str | None = None  # for the recruiter-facing report only; never sent to the LLM
    profile_url: str | None = None

    criteria_role_id: str | None = None
    criteria_version: int | None = None
    criteria_hash: str | None = None
    prompt_version: str | None = None
    redaction_version: str | None = None
    model_id: str | None = None
    model_version: str | None = None  # as reported by the API response

    status: Status
    score: int | None = Field(default=None, ge=1, le=100)
    tier: Tier | None = None
    holistic_score: int | None = None  # the model's own 1-100, diagnostic only
    confidence: Confidence | None = None
    criteria_results: list[CriterionResult] = Field(default_factory=list)
    knockout_results: list[KnockoutCheck] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    summary: str = ""
    flags: list[Flag] = Field(default_factory=list)
    # Why a human should look (NEEDS_REVIEW), or why the row was SKIPPED (one fixed sentence).
    review_reasons: list[str] = Field(default_factory=list)
    # False when output validation failed: nothing the model wrote may reach Lever.
    rationale_publishable: bool = True
    validation: ValidationReport = Field(default_factory=ValidationReport)
    usage: TokenUsage | None = None
    file_hmac: str | None = None
    redacted_text_hmac: str | None = None
    error: str | None = None
    # ERROR only: the failure says nothing about the applicant and may pass on a re-run
    # (Lever 5xx/429, timeout or connection failure; a retryable model error). Write-back
    # then leaves an existing AI ranking in Lever unchanged.
    transient: bool = False

    @property
    def is_ranked(self) -> bool:
        return self.score is not None and self.status in (Status.SCORED, Status.KNOCKOUT_FLAGGED,
                                                          Status.NEEDS_REVIEW)


# ----------------------------------------------------------------------------------------
# Write-back plan (dry-run prints it; live mode sends it)
# ----------------------------------------------------------------------------------------


class PlannedRequest(BaseModel):
    method: Literal["GET", "POST", "PUT", "DELETE"]
    path: str  # relative to the Lever base URL, e.g. /opportunities/{id}/addTags
    params: dict[str, str] = Field(default_factory=dict)
    json_body: dict | None = None
    purpose: str = ""


class WritebackPlan(BaseModel):
    opportunity_id: str
    evaluation_id: str
    tags_to_add: list[str] = Field(default_factory=list)
    tags_to_remove: list[str] = Field(default_factory=list)
    note_text: str | None = None
    existing_note_id: str | None = None  # update in place when we wrote one before
    requests: list[PlannedRequest] = Field(default_factory=list)
    # Why a non-SKIPPED plan has no requests (becomes the result's dry_run_reason).
    reason: str | None = None


class WritebackResult(BaseModel):
    opportunity_id: str
    evaluation_id: str
    executed: bool
    dry_run_reason: str | None = None  # why it was not sent
    note_id: str | None = None
    # One per request sent: {"method", "path", "status": "ok"|"error", "http_status": int|None}
    responses: list[dict] = Field(default_factory=list)
    error: str | None = None


# ----------------------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------------------


class RankingRow(BaseModel):
    rank: int | None  # None for unranked (error/skipped) rows, which are still listed
    evaluation: Evaluation
    writeback: WritebackResult | None = None


class RankingReport(BaseModel):
    posting_id: str
    posting_title: str = ""
    criteria_role_id: str | None = None
    criteria_version: int | None = None
    mode: RunMode
    generated_at: datetime = Field(default_factory=utcnow)
    policy_regimes: list[str] = Field(default_factory=list)
    policy_notes: list[str] = Field(default_factory=list)
    # Run-level remarks for the reader (e.g. Lever listed no applicants); never model text.
    notes: list[str] = Field(default_factory=list)
    rows: list[RankingRow] = Field(default_factory=list)

    @property
    def watermark(self) -> str | None:
        if self.mode == RunMode.EVALUATION:
            return "EVALUATION ONLY - not for hiring decisions"
        return "Advisory ranking - not a decision. Review every application."
