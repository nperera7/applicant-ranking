"""Orchestration: posting -> applicants -> resumes -> LLM -> validated scores -> ranking
-> (optional) write-back. The only module that knows the whole flow.

``rank_posting`` flow:
1. Fetch the posting; find its criteria (none -> RankingError telling the user to run
   ``criteria draft``). Lint errors -> RankingError (all modes). Criteria whose approval
   cannot be relied on (``RoleCriteria.approval_issue``: unapproved, or ``approved_at``
   in the future) -> RankingError unless mode is EVALUATION.
2. Policy: ``policy.evaluate(posting, mode, today)`` with ``today`` = ``policy_today``
   (the earliest calendar date among the regimes' local time zones, so a date-based gate
   never opens before the day has begun where the rule applies); not allowed
   (DRY_RUN/LIVE) -> PolicyBlocked(decision).
3. List opportunities (active only unless include_archived; ``confidentiality="all"`` for
   a confidential posting, so its applicants are listed too), apply ``limit``. When Lever
   lists nobody the report carries ``NO_APPLICANTS_NOTE`` in ``notes``.
4. Per opportunity (bounded concurrency): skip -> SKIPPED row if the store has a deletion
   tombstone, the Lever record could not be read (``Opportunity.record_valid`` False:
   ``SKIP_UNREADABLE``, no name stored) or ``not opportunity.storage_allowed``. Pick the
   newest resume whose file.status == "processed" (by file.uploadedAt, then createdAt);
   download (422 -> fall back to parsedData text with ``resume_from_parsed_data``);
   prepare (names from opportunity.name; emails/phones from the record as extra
   identifiers; JD text from the posting, plus its requirement/responsibility lists on
   their own for the ``jd_copy`` check). needs_vision -> evaluator.transcribe(page_images)
   -> prepare_text(transcript) and merge flags (keep ``scanned_resume`` and any
   hidden-text flags).
   On the webhook path (``evaluate_opportunity`` with ``defer_pending``, the default) a
   resume Lever is still parsing (``resume_pending``: no file yet, or the newest file's
   status is "processing" or null) is a SKIPPED row with ``SKIP_RESUME_PENDING`` instead:
   nothing is scored, recorded or written, and the service retries later. The newest
   file is never bypassed for an older processed one there; batch runs keep scoring the
   newest processed file.
5. Batch integrity signals across the posting: ``signals.find_duplicates`` on redacted
   texts -> ``duplicate_resume`` (review) on each member; the same email/phone (from the
   Lever records) on opportunities with different names -> ``contact_reuse`` (review).
6. Per prepared resume (``llm_concurrency``): cache key = sha256 of
   "opportunity|content|criteria_hash|prompt_version|model_id|redaction_version|
   lexicon_version". content is the redacted_text_hmac (the exact text sent) or, for a
   scanned resume, the file_hmac (so the cache is checked before the paid vision call;
   the preparer folds its settings into redaction_version for this);
   lexicon_version is ``lexicon_fingerprint``. Unless ``force``, a cache hit returns the
   stored evaluation (or a derived one, see below; write-back still runs for it).
   Otherwise assess -> validate_and_score -> if corrective_note: assess once more with
   it and validate with corrective_retries=1, carrying the first attempt's
   review-severity model signals (an injection report, a possible_prompt_injection red
   flag, mostly ungrounded evidence) as fixed-text prior flags, so a clean second answer
   cannot erase what the model saw the first time (``carried_review_signals``). LLM
   errors -> ERROR evaluation (error
   text, ``model_blocked``/``invalid_model_output`` flags as appropriate). Record every
   evaluation in the store, EVALUATION runs included (mode=evaluation), for
   auditability.
7. Write-back per evaluation, once every evaluation of the run has finished: plan with
   current tags + stored note id; apply (sent only in LIVE with write_enabled and an
   allowed policy decision).
8. Rows: ranked evaluations (score not None) sorted by score desc, ties by
   opportunity createdAt asc then id; rank = 1..n. Unranked rows follow with rank None.
   Nobody is dropped from the report.

Opportunities processed independently: one failure becomes an ERROR row, never aborts
the batch. The exception is a systemic error (``is_systemic_error``: Lever rejecting the
API key, the model rejecting the credentials, project or model id): it cancels the rest
of steps 4-6 and propagates before step 7, so nothing is written to Lever. Every
applicant would fail the same way, and "Not scored" written over their tags would hide
the outage. An ERROR caused by an outage (Lever 5xx/429, timeouts, retryable model
errors) is marked ``transient``; write-back then leaves existing AI ranking tags alone.

Implementation notes (choices the flow above leaves open):
- Concurrency: Lever reads, downloads, resume preparation and write-backs share one small
  limit (``io_concurrency``); model calls (assessments and vision transcriptions) share
  ``settings.llm_concurrency``. Semaphores are per event loop, so one Ranker can serve
  several ``asyncio.run`` calls. Output validation (pure CPU: quote grounding, lexicon
  scan) runs in a worker thread so the webhook service's event loop keeps answering.
- Cache: a hit counts only within the same kind of run: EVALUATION runs reuse EVALUATION
  results, DRY_RUN/LIVE runs reuse DRY_RUN/LIVE results, so an evaluation-only result
  never stands in for a real run in the audit log or a Lever note. (The store returns the
  newest result for a key; when that one is of the other kind, the applicant is simply
  evaluated again.) Settings that only affect code-side scoring (tier thresholds, the
  score-disagreement threshold) are not in the key, so changing them costs no model
  call. A hit is recorded again as a derived evaluation (the cached assessment under a
  new evaluation id, ``generation_config`` naming ``derived_from`` and the reason; no
  model call) when:
  - this run's batch signals raise a review flag it lacks (for example an applicant
    first scored alone through the webhook path, whose duplicate shows up in a later
    posting run): the flags are added and status and review reasons recomputed;
  - its tier or ``score_disagreement`` flag differs under the current thresholds: both
    are recomputed from the stored score, so one report never mixes two tier scales;
  - a LIVE run reuses a DRY_RUN result: the derived row has mode=live and this run's
    policy decision, so the audit log records the live use and the write-back links to
    it. (A DRY_RUN run reuses a LIVE result as it is.)
- A failed corrective retry (LLMError) keeps the first, already fail-safe outcome
  (protected-term flag, rationale withheld) instead of turning it into an ERROR.
- No processed resume file -> ERROR with ``no_resume``; when the newest file's status is
  "unsupported" or "error" (Lever could not process it) also ``unreadable_file``. A file
  over ``max_resume_bytes`` -> ERROR with ``unreadable_file``. A 422 without usable
  parsedData -> ERROR with ``unreadable_file`` and ``no_resume``.
- SKIPPED evaluations (deleted, unreadable record, anonymized / no storage consent, no
  criteria, resume pending) are never recorded and carry no candidate name when Lever
  forbids storing the candidate or the record could not be read; the reason is in
  ``review_reasons``. Nothing is written back for them.
- Just before an evaluation is recorded the deletion tombstone is checked again, so a
  candidateDeleted webhook that lands mid-run can't be undone by the run.
- Deletions: a candidateDeleted webhook names its opportunity in the unsigned ``data``,
  so ``handle_candidate_deleted(source="webhook")`` confirms it with Lever first (GET
  /opportunities/{id}: 404, ``isAnonymized`` or a ``deletedAt`` confirm; a live record
  means the event is ignored and logged, ``PurgeResult.confirmed`` False).
  ``reconcile_deletions`` reads GET /opportunities/deleted
  (``LeverClient.list_deleted_opportunities``: deleted candidates, both bounds
  inclusive), never /applications/deleted, whose rows are deleted applications of
  possibly live candidates.
- Error texts on evaluations come only from LeverError/LLMError messages (which carry no
  resume text); any other exception is reported by type, and logged by type and code
  location only.
- ``evaluate_local`` never uses the cache (fairness runs repeat the same text on
  purpose) and records with no cache key.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import traceback
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from applicant_ranking.config import Settings
from applicant_ranking.criteria import CriteriaRepository
from applicant_ranking.criteria_lint import lint_criteria
from applicant_ranking.domain import (
    Evaluation,
    Flag,
    FlagCode,
    PreparedResume,
    RankingReport,
    RankingRow,
    RoleCriteria,
    RunMode,
    Status,
    TokenUsage,
    WritebackResult,
    utcnow,
)
from applicant_ranking.lever.client import (
    LeverAuthError,
    LeverClient,
    LeverConnectionError,
    LeverError,
    LeverFileTooLarge,
    LeverNotFound,
    LeverUnprocessable,
)
from applicant_ranking.lever.models import Application, Opportunity, Posting, Resume
from applicant_ranking.llm.base import (
    AssessmentRequest,
    Evaluator,
    LLMBlockedError,
    LLMCallResult,
    LLMConfigError,
    LLMError,
    LLMOutputError,
)
from applicant_ranking.policy import (
    CA_FEHA,
    CO_ADMT,
    IL_HRA,
    NYC_LL144,
    ON_ESA,
    PolicyDecision,
    PolicyEngine,
)
from applicant_ranking.resumes.prepare import ResumePreparer, parsed_data_to_text
from applicant_ranking.resumes.redact import REDACTION_VERSION
from applicant_ranking.resumes.signals import find_duplicates
from applicant_ranking.scoring.compute import tier_for
from applicant_ranking.scoring.lexicon import ProtectedTermLexicon
from applicant_ranking.scoring.validate import ScoringOutcome, review_reason, validate_and_score
from applicant_ranking.store import PurgeResult, Store
from applicant_ranking.writeback import LeverWriter, WritebackPlanner, shorten

logger = logging.getLogger(__name__)

LOCAL_OPPORTUNITY_ID = "local"
DEFAULT_IO_CONCURRENCY = 4
MAX_ERROR_CHARS = 300
_MAX_LISTED_IDS = 5
_MIN_PHONE_DIGITS = 7
# LLMConfigError codes that mean the deployment is wrong, not one request: credentials,
# permission, model id. (None: wiring's placeholder for a missing model configuration.)
_SYSTEMIC_LLM_CODES = frozenset({None, 401, 403, 404})
# Lever file.status values (processing | processed | unsupported | error | null): a file
# Lever is still parsing, and one it gave up on.
_PENDING_FILE_STATUSES = frozenset({"processing", ""})
_UNPROCESSABLE_FILE_STATUSES = frozenset({"unsupported", "error"})
# The local calendar each jurisdiction's date rules run on (see ``policy_today``).
_REGIME_ZONES: dict[str, str] = {
    IL_HRA: "America/Chicago",
    NYC_LL144: "America/New_York",
    CO_ADMT: "America/Denver",
    CA_FEHA: "America/Los_Angeles",
    ON_ESA: "America/Toronto",
}

SKIP_DELETED = "Not evaluated: the candidate was deleted in Lever."
SKIP_STORAGE = ("Not evaluated: Lever's data-protection settings don't allow storing this "
                "candidate (anonymized, or no consent to store).")
SKIP_NO_POSTING = "Not evaluated: the opportunity has no application to a posting."
SKIP_NO_CRITERIA = "Not evaluated: the posting has no criteria file."
SKIP_RESUME_PENDING = "Not evaluated yet: Lever is still processing the resume."
SKIP_UNREADABLE = ("Not evaluated: the Lever record could not be read (unexpected field "
                   "shapes); review it in Lever.")
NO_APPLICANTS_NOTE = ("Lever returned no applicants for this posting; check the archived "
                      "filter and, for a confidential posting, that the API key has "
                      "confidential access.")
DELETED_OPPORTUNITIES_PATH = "/opportunities/deleted"
_CONFIDENTIAL = "confidential"


class RankingError(RuntimeError):
    """The run can't start (no criteria, lint errors, unapproved criteria, no Lever key)."""


class PolicyBlocked(RankingError):
    def __init__(self, decision: PolicyDecision) -> None:
        super().__init__("blocked by jurisdiction policy: " + "; ".join(decision.blockers))
        self.decision = decision


def cache_key(*, opportunity_id: str, content_hmac: str, criteria_hash: str,
              prompt_version: str, model_id: str, redaction_version: str,
              lexicon_version: str) -> str:
    raw = "|".join((opportunity_id, content_hmac, criteria_hash, prompt_version, model_id,
                    redaction_version, lexicon_version))
    return hashlib.sha256(raw.encode()).hexdigest()


def lexicon_fingerprint(lexicon: ProtectedTermLexicon) -> str:
    """``v{version}:{sha256}`` over the lexicon's categories and allow phrases. Output
    validation (and its corrective retry) depends on them, so a changed list must not
    reuse results validated under the old one, even when its version was not bumped."""
    payload = {"version": lexicon.version, "categories": lexicon.categories,
               "allow_phrases": lexicon.allow_phrases}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"v{lexicon.version}:{hashlib.sha256(canonical.encode()).hexdigest()}"


def is_systemic_error(exc: BaseException) -> bool:
    """An error about this deployment rather than one applicant, which aborts the run:
    any LeverAuthError (bad key, missing permission), and an LLMConfigError for rejected
    credentials, permission or model id (401/403/404) or without a code (the model is not
    configured). A 400 stays per-applicant: one resume (a bad page image) can cause it."""
    if isinstance(exc, LeverAuthError):
        return True
    return isinstance(exc, LLMConfigError) and exc.code in _SYSTEMIC_LLM_CODES


def _is_transient(exc: BaseException) -> bool:
    """The failure says nothing about the applicant and may pass on a re-run: Lever
    5xx/429 (after the client's retries), timeouts and connection failures, and model
    errors marked retryable."""
    if isinstance(exc, LLMError):
        return exc.retryable
    if isinstance(exc, LeverConnectionError):  # LeverTimeout included
        return True
    status = exc.status if isinstance(exc, LeverError) else None
    return status is not None and (status == 429 or status >= 500)


async def _gather_or_cancel[T](aws: Iterable[Awaitable[T]]) -> list[T]:
    """``asyncio.gather``, except that the first exception cancels the other tasks and
    waits for them before it propagates (unchanged, so the CLI can map its type)."""
    tasks = [asyncio.ensure_future(aw) for aw in aws]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


# ------------------------------------------------------------------------------- helpers


def job_description_text(posting: Posting) -> str:
    """The posting's text (description, lists, closing) for the ``jd_copy`` check. HTML
    list items are left in; ``signals.jd_overlap`` strips tags."""
    content = posting.content
    parts = [content.description or content.descriptionHtml]
    for item in content.lists:
        parts.extend((item.text, item.content))
    parts.append(content.closing or content.closingHtml)
    return "\n".join(part for part in parts if part and part.strip())


def job_description_lists(posting: Posting) -> str:
    """Only the posting's lists (requirements, responsibilities, ...): the preparer
    measures ``jd_copy`` against these when they are long enough, so "about us" and
    benefits boilerplate neither dilutes nor inflates the overlap."""
    parts: list[str] = []
    for item in posting.content.lists:
        parts.extend((item.text, item.content))
    return "\n".join(part for part in parts if part and part.strip())


def _confidentiality_for(posting: Posting) -> str | None:
    """``"all"`` for a confidential posting (Lever lists its opportunities only when
    asked), else None (Lever's default: non-confidential)."""
    confidential = (posting.confidentiality or "").strip().lower() == _CONFIDENTIAL
    return "all" if confidential else None


def _file_status(resume: Resume) -> str:
    return (resume.file.status or "").strip().lower() if resume.file is not None else ""


def _newest_key(resume: Resume) -> tuple[int, int]:
    uploaded = resume.file.uploadedAt if resume.file is not None else None
    return (uploaded or 0, resume.createdAt or 0)


def newest_resume(resumes: Sequence[Resume]) -> Resume | None:
    """The newest resume that has a file (by file.uploadedAt, then createdAt), whatever
    its status; None when no resume has a file."""
    with_file = [r for r in resumes if r.file is not None]
    return max(with_file, key=_newest_key) if with_file else None


def select_resume(resumes: Sequence[Resume]) -> Resume | None:
    """The newest resume whose file Lever has processed (by file.uploadedAt, then
    createdAt); None when there is none."""
    processed = [r for r in resumes if _file_status(r) == "processed"]
    return max(processed, key=_newest_key) if processed else None


def resume_pending(resumes: Sequence[Resume]) -> bool:
    """True while Lever may still deliver a processed resume: no resume has a file yet,
    or the newest file's status is "processing" or null. "unsupported" and "error" are
    final, so they are not pending. An older processed file does not make a pending newer
    one ready: scoring it would rank the applicant on a resume they replaced."""
    newest = newest_resume(resumes)
    return newest is None or _file_status(newest) in _PENDING_FILE_STATUSES


def policy_today(now: datetime, regimes: Iterable[str]) -> date:
    """The calendar date the jurisdiction policy is evaluated on: the earliest local date
    among the ``regimes``' time zones (``_REGIME_ZONES``) at the instant ``now``, so a
    rule that opens on a date (the NYC notice wait, ``approved_at``) does not open while
    it is still the day before where that rule applies. ``now`` is UTC when naive. Without
    time zone data (or regimes), the UTC date."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    dates: list[date] = []
    for regime in regimes:
        zone = _REGIME_ZONES.get(regime)
        if zone is None:
            continue
        try:
            dates.append(now.astimezone(ZoneInfo(zone)).date())
        except ZoneInfoNotFoundError:
            logger.warning("time zone %s is not available (install tzdata); policy dates fall "
                           "back to UTC", zone)
    return min(dates) if dates else now.astimezone(UTC).date()


def carried_review_signals(flags: Iterable[Flag]) -> list[Flag]:
    """Review-severity signals of a first model attempt that a corrective retry must not
    drop: the model's own injection report (``model_injection_report`` or a review-level
    ``model_red_flag``), and evidence that was mostly ungrounded (review-level
    ``ungrounded_evidence``). Returned with fixed details (never model text), to be passed
    as prior flags to the retry's validation."""
    carried: list[Flag] = []
    if any(f.code is FlagCode.MODEL_INJECTION_REPORT
           or (f.code is FlagCode.MODEL_RED_FLAG and f.severity == "review") for f in flags):
        carried.append(Flag(code=FlagCode.MODEL_INJECTION_REPORT, source="model",
                            severity="review",
                            detail="the model reported an attempt to manipulate the "
                                   "evaluation (first attempt)"))
    if any(f.code is FlagCode.UNGROUNDED_EVIDENCE and f.severity == "review" for f in flags):
        carried.append(Flag(code=FlagCode.UNGROUNDED_EVIDENCE, severity="review",
                            detail="most of the evidence quoted in the first attempt was not "
                                   "found in the resume"))
    return carried


def contact_identifiers(opportunity: Opportunity) -> list[str]:
    """Emails and phone numbers on the Lever record (redacted from the resume as extra
    identifiers, and compared across applications for ``contact_reuse``)."""
    values = [email.strip() for email in opportunity.emails
              if isinstance(email, str) and email.strip()]
    for phone in opportunity.phones:
        value = phone.get("value") if isinstance(phone, dict) else None
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return list(dict.fromkeys(values))


def _application_id(opportunity: Opportunity, posting_id: str | None) -> str | None:
    fallback: str | None = None
    for app in opportunity.applications:
        if isinstance(app, Application):
            if posting_id is not None and app.posting == posting_id:
                return app.id
            fallback = fallback or app.id
        elif isinstance(app, str) and app:
            fallback = fallback or app
    return fallback


def _norm_name(name: str | None) -> str:
    return " ".join((name or "").casefold().split())


def _contact_keys(opportunity: Opportunity) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for email in opportunity.emails:
        if isinstance(email, str) and "@" in email:
            keys.append(("email address", email.strip().casefold()))
    for phone in opportunity.phones:
        value = phone.get("value") if isinstance(phone, dict) else None
        digits = re.sub(r"\D", "", value) if isinstance(value, str) else ""
        if len(digits) >= _MIN_PHONE_DIGITS:
            keys.append(("phone number", digits[-10:]))
    return list(dict.fromkeys(keys))


def _listed(ids: Sequence[str]) -> str:
    shown = ", ".join(ids[:_MAX_LISTED_IDS])
    return shown + (f" (+{len(ids) - _MAX_LISTED_IDS} more)" if len(ids) > _MAX_LISTED_IDS
                    else "")


def batch_integrity_flags(redacted_texts: Mapping[str, str],
                          opportunities: Iterable[Opportunity]) -> dict[str, list[Flag]]:
    """Step 5: ``duplicate_resume`` for exact/near-duplicate redacted texts and
    ``contact_reuse`` for an email or phone number shared by applications under different
    names. Both are review flags. Details name the other opportunity ids, never the
    contact value. Records without a name are not compared (no "different name" to see)."""
    flags: dict[str, list[Flag]] = {}
    for opp_id, others in find_duplicates(redacted_texts).items():
        flags.setdefault(opp_id, []).append(Flag(
            code=FlagCode.DUPLICATE_RESUME, severity="review",
            detail=f"same or near-identical resume text as {len(others)} other "
                   f"application(s) for this posting: {_listed(others)}",
        ))

    by_contact: dict[tuple[str, str], dict[str, str]] = {}
    for opp in opportunities:
        name = _norm_name(opp.name)
        if not name:
            continue
        for key in _contact_keys(opp):
            by_contact.setdefault(key, {})[opp.id] = name
    shared: dict[str, dict[str, list[str]]] = {}
    for (kind, _value), members in by_contact.items():
        if len(set(members.values())) < 2:
            continue
        for opp_id, name in members.items():
            others = [other for other, other_name in members.items() if other_name != name]
            bucket = shared.setdefault(opp_id, {}).setdefault(kind, [])
            bucket.extend(o for o in others if o not in bucket)
    for opp_id, kinds in shared.items():
        for kind, others in kinds.items():
            flags.setdefault(opp_id, []).append(Flag(
                code=FlagCode.CONTACT_REUSE, severity="review",
                detail=f"{kind} shared with {len(others)} application(s) under a different "
                       f"name: {_listed(others)}",
            ))
    return flags


def _dedupe_flags(flags: Iterable[Flag]) -> list[Flag]:
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[Flag] = []
    for flag in flags:
        key = (flag.code.value, flag.source, flag.severity, flag.detail)
        if key not in seen:
            seen.add(key)
            unique.append(flag)
    return unique


def _review_reasons(flags: Iterable[Flag]) -> list[str]:
    reasons: list[str] = []
    for flag in flags:
        if flag.severity == "review":
            reason = review_reason(flag)
            if reason not in reasons:
                reasons.append(reason)
    return reasons


def _llm_error_flags(exc: LLMError) -> list[Flag]:
    if isinstance(exc, LLMBlockedError):
        return [Flag(code=FlagCode.MODEL_BLOCKED, severity="review",
                     detail="the model declined or was blocked from answering")]
    if isinstance(exc, LLMOutputError):
        return [Flag(code=FlagCode.INVALID_MODEL_OUTPUT, severity="review",
                     detail="the model's answer could not be used")]
    return []


def _error_text(exc: BaseException, *, prefix: str = "") -> str:
    """Messages of our own Lever/LLM errors carry no resume text; anything else is named
    by type only."""
    if isinstance(exc, LeverError | LLMError | RankingError) and str(exc):
        text = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, LLMError) and exc.retryable:
            text += " (retryable)"
    else:
        text = f"{type(exc).__name__} (details in the service logs)"
    return shorten(prefix + text, MAX_ERROR_CHARS)


def _where(exc: BaseException) -> str:
    frames = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
    if not frames:
        return "unknown location"
    last = frames[-1]
    return f"{last.filename}:{last.lineno} in {last.name}"


def _add_usage(a: TokenUsage, b: TokenUsage) -> TokenUsage:
    cost = (None if a.estimated_cost_usd is None or b.estimated_cost_usd is None
            else a.estimated_cost_usd + b.estimated_cost_usd)
    return TokenUsage(prompt_tokens=a.prompt_tokens + b.prompt_tokens,
                      output_tokens=a.output_tokens + b.output_tokens,
                      thinking_tokens=a.thinking_tokens + b.thinking_tokens,
                      total_tokens=a.total_tokens + b.total_tokens,
                      estimated_cost_usd=cost)


def _created_key(opportunity: Opportunity | None) -> tuple[bool, int]:
    created = opportunity.createdAt if opportunity is not None else None
    return (created is None, created or 0)


def _order_rows(items: Sequence[tuple[RankingRow, Opportunity | None]]) -> list[RankingRow]:
    """Step 8: ranked rows by score desc, then createdAt asc, then id; ranks 1..n; the
    unranked rows after them by createdAt, then id."""
    ranked = sorted(
        (item for item in items if item[0].evaluation.score is not None),
        key=lambda item: (-(item[0].evaluation.score or 0), _created_key(item[1]),
                          item[0].evaluation.opportunity_id),
    )
    unranked = sorted(
        (item for item in items if item[0].evaluation.score is None),
        key=lambda item: (_created_key(item[1]), item[0].evaluation.opportunity_id),
    )
    rows: list[RankingRow] = []
    for rank, (row, _) in enumerate(ranked, start=1):
        rows.append(row.model_copy(update={"rank": rank}))
    rows.extend(row.model_copy(update={"rank": None}) for row, _ in unranked)
    return rows


# ------------------------------------------------------------------------- run state


@dataclass(frozen=True)
class _Context:
    mode: RunMode
    criteria: RoleCriteria
    criteria_hash: str
    posting_id: str | None
    job_description: str = ""
    job_description_lists: str = ""
    policy_allowed: bool = False
    # Jurisdiction decision recorded in the audit log; None for local evaluations (no gate).
    policy_regimes: list[str] | None = None
    policy_gate: bool | None = None
    lookup_cache: bool = True  # False: force, or local evaluation
    store_cache_key: bool = True
    record: bool = True
    # Webhook path: a resume Lever is still processing is a SKIPPED row to retry later.
    defer_pending: bool = False


@dataclass
class _Work:
    """One applicant moving through the run."""

    opportunity_id: str
    opportunity: Opportunity | None = None
    application_id: str | None = None
    candidate_name: str | None = None
    profile_url: str | None = None
    names: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    prepared: PreparedResume | None = None
    cached: Evaluation | None = None
    cached_key: str | None = None
    # Scanned resume: its cache key (file HMAC), computed before the vision call.
    vision_key: str | None = None
    batch_flags: list[Flag] = field(default_factory=list)
    # Set early when the applicant is skipped or failed before the model step.
    evaluation: Evaluation | None = None


# ---------------------------------------------------------------------------------- ranker


class Ranker:
    def __init__(
        self,
        *,
        settings: Settings,
        lever: LeverClient | None,
        evaluator: Evaluator,
        store: Store,
        criteria: CriteriaRepository,
        policy: PolicyEngine,
        preparer: ResumePreparer,
        lexicon: ProtectedTermLexicon,
        planner: WritebackPlanner,
        writer: LeverWriter,
        clock: Callable[[], datetime] = utcnow,
        io_concurrency: int = DEFAULT_IO_CONCURRENCY,
    ) -> None:
        """``io_concurrency`` bounds concurrent Lever reads/downloads, resume preparation
        and write-backs; model calls are bounded by ``settings.llm_concurrency``."""
        if io_concurrency < 1:
            raise ValueError("io_concurrency must be at least 1")
        self._settings = settings
        self._lever = lever
        self._evaluator = evaluator
        self._store = store
        self._criteria = criteria
        self._policy = policy
        self._preparer = preparer
        self._lexicon = lexicon
        self._lexicon_version = lexicon_fingerprint(lexicon)
        self._planner = planner
        self._writer = writer
        self.clock = clock
        self._io_concurrency = io_concurrency
        self._semaphores_for: tuple[asyncio.AbstractEventLoop, asyncio.Semaphore,
                                    asyncio.Semaphore] | None = None

    # ------------------------------------------------------------------ public API

    async def rank_posting(self, posting_id: str, *, mode: RunMode = RunMode.DRY_RUN,
                           limit: int | None = None, include_archived: bool = False,
                           force: bool = False) -> RankingReport:
        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1")
        mode = RunMode(mode)
        lever = self._require_lever()
        posting = await lever.get_posting(posting_id)
        criteria = self._criteria.for_posting(posting_id)
        if criteria is None:
            raise RankingError(self._no_criteria_message(posting_id))
        self.check_criteria(criteria, mode=mode)
        decision = self._policy_decision(posting, mode)
        ctx = self._context(criteria, mode=mode, posting_id=posting_id, posting=posting,
                            decision=decision, force=force)

        opportunities = await self._list_opportunities(
            lever, posting_id, include_archived=include_archived, limit=limit,
            confidentiality=_confidentiality_for(posting))
        works = [self._new_work(opp, posting_id) for opp in opportunities]
        notes = [NO_APPLICANTS_NOTE] if not works else []
        logger.info("ranking posting %s (%s): %d application(s)", posting_id, mode.value,
                    len(works))
        await _gather_or_cancel(self._collect(lever, work, ctx) for work in works)
        self._apply_batch_signals(works)
        # Every evaluation finishes before any write-back is planned, so a systemic error
        # (raised by _collect or _score) ends the run with nothing written to Lever.
        evaluations = await _gather_or_cancel(self._score(work, ctx) for work in works)
        rows = await asyncio.gather(*(self._write_row(work, evaluation, ctx)
                                      for work, evaluation in zip(works, evaluations,
                                                                  strict=True)))

        ordered = _order_rows([(row, work.opportunity)
                               for row, work in zip(rows, works, strict=True)])
        return RankingReport(
            posting_id=posting_id,
            posting_title=posting.text,
            criteria_role_id=criteria.role_id,
            criteria_version=criteria.version,
            mode=mode,
            generated_at=self.clock(),
            policy_regimes=list(decision.regimes),
            policy_notes=list(decision.notes),
            notes=notes,
            rows=ordered,
        )

    async def evaluate_opportunity(self, opportunity_id: str, *,
                                   mode: RunMode = RunMode.DRY_RUN,
                                   force: bool = False,
                                   defer_pending: bool = True) -> RankingRow:
        """Single applicant (webhook path). Same gates as rank_posting for the
        opportunity's posting. A posting without criteria -> SKIPPED row (not an error:
        most postings won't be configured). Deleted/anonymized -> SKIPPED.

        With ``defer_pending`` (the default) a resume Lever is still processing
        (``resume_pending``) is a SKIPPED row with ``review_reasons ==
        [SKIP_RESUME_PENDING]``: nothing is recorded or written, and the caller retries
        later. ``defer_pending=False`` (a last retry, or a batch-like caller) scores the
        newest processed file or reports ERROR ``no_resume`` as ``rank_posting`` does.

        A tombstoned opportunity is skipped without calling Lever. A scored row gets rank
        1. Batch signals (duplicates, contact reuse) need the whole posting, so they are
        raised by ``rank_posting`` only. Lever errors while fetching the opportunity or
        its posting propagate."""
        mode = RunMode(mode)
        lever = self._require_lever()
        if self._store.is_deleted(opportunity_id):
            return RankingRow(rank=None, evaluation=self._skipped(
                _Work(opportunity_id=opportunity_id), mode, None, SKIP_DELETED))
        opportunity = await lever.get_opportunity(opportunity_id)
        posting_id = opportunity.posting_id
        work = self._new_work(opportunity, posting_id)
        if not opportunity.record_valid:
            return RankingRow(rank=None, evaluation=self._skipped(work, mode, posting_id,
                                                                  SKIP_UNREADABLE,
                                                                  anonymous=True))
        if not opportunity.storage_allowed:
            return RankingRow(rank=None, evaluation=self._skipped(work, mode, posting_id,
                                                                  SKIP_STORAGE))
        if not posting_id:
            return RankingRow(rank=None, evaluation=self._skipped(work, mode, None,
                                                                  SKIP_NO_POSTING))
        criteria = self._criteria.for_posting(posting_id)
        if criteria is None:
            logger.info("opportunity %s: posting %s has no criteria; skipped", opportunity_id,
                        posting_id)
            return RankingRow(rank=None, evaluation=self._skipped(work, mode, posting_id,
                                                                  SKIP_NO_CRITERIA))
        posting = await lever.get_posting(posting_id)
        self.check_criteria(criteria, mode=mode)
        decision = self._policy_decision(posting, mode)
        ctx = self._context(criteria, mode=mode, posting_id=posting_id, posting=posting,
                            decision=decision, force=force, defer_pending=defer_pending)
        await self._collect(lever, work, ctx)
        row = await self._finish(work, ctx)
        if row.evaluation.score is not None:
            row = row.model_copy(update={"rank": 1})
        return row

    async def evaluate_local(self, *, criteria: RoleCriteria, data: bytes | None = None,
                             text: str | None = None, file_name: str | None = None,
                             candidate_names: Sequence[str] = (),
                             opportunity_id: str = LOCAL_OPPORTUNITY_ID) -> Evaluation:
        """EVALUATION mode on a local file or text (criteria tuning, fairness checks).
        No Lever, no write-back; lint errors still raise RankingError; recorded in the
        store only when ``opportunity_id`` != "local".

        Exactly one of ``data`` and ``text`` must be given (ValueError otherwise). The
        cache is never used. Model and preparation failures give an ERROR evaluation,
        except systemic ones (``is_systemic_error``), which raise."""
        if (data is None) == (text is None):
            raise ValueError("pass exactly one of data or text")
        self.check_criteria(criteria, mode=RunMode.EVALUATION)
        ctx = _Context(mode=RunMode.EVALUATION, criteria=criteria,
                       criteria_hash=criteria.content_hash(), posting_id=None,
                       lookup_cache=False, store_cache_key=False,
                       record=opportunity_id != LOCAL_OPPORTUNITY_ID)
        names = [name for name in candidate_names if name and name.strip()]
        work = _Work(opportunity_id=opportunity_id, candidate_name=names[0] if names else None,
                     names=names)
        try:
            if data is not None:
                work.prepared = await self._preparer.prepare_file(
                    data, opportunity_id=opportunity_id, file_name=file_name,
                    candidate_names=names, source="local_file")
            else:
                work.prepared = self._preparer.prepare_text(
                    text or "", opportunity_id=opportunity_id, candidate_names=names,
                    source="local_file", file_name=file_name)
            if work.prepared.needs_vision:
                await self._transcribe(work, ctx)
            return await self._evaluate(work, ctx)
        except Exception as exc:
            if is_systemic_error(exc):
                raise
            return self._failed(work, ctx, exc)

    async def handle_candidate_deleted(self, opportunity_id: str, *,
                                       source: str = "webhook",
                                       posting_id: str | None = None) -> PurgeResult:
        """Tombstone (never re-score) + purge PII unless on legal hold.

        The tombstone is written even when a hold stops the purge. ``posting_id`` (when
        known) is checked for a posting-level hold too.

        ``source="webhook"``: the id comes from the webhook's unsigned ``data``, so the
        deletion is confirmed with Lever first (``_deletion_confirmed``). An opportunity
        Lever still serves as live is not touched: the event is logged and an empty
        PurgeResult (nothing deleted, not on hold) returned. Other Lever errors propagate,
        so the job is logged as failed and ``reconcile_deletions`` catches up. Ids from
        ``reconcile_deletions`` (or a CLI operator) already come from Lever and are acted
        on directly."""
        if source == "webhook" and not await self._deletion_confirmed(opportunity_id):
            logger.warning("candidateDeleted for opportunity %s not confirmed by Lever (the "
                           "record still exists); ignored", opportunity_id)
            return PurgeResult(opportunity_id=opportunity_id, on_hold=False, confirmed=False)
        self._store.record_deletion(opportunity_id, source=source, deleted_at=self.clock())
        result = self._store.purge_candidate(opportunity_id, posting_id=posting_id)
        logger.info("candidate deletion (%s) for opportunity %s: on_hold=%s, %d PII row(s) "
                    "deleted", source, opportunity_id, result.on_hold, result.pii_rows_deleted)
        return result

    async def _deletion_confirmed(self, opportunity_id: str) -> bool:
        """GET /opportunities/{id}: deleted when Lever answers 404, or serves a record
        that is anonymized or carries a ``deletedAt``."""
        lever = self._require_lever()
        try:
            opportunity = await lever.get_opportunity(opportunity_id)
        except LeverNotFound:
            return True
        extra = opportunity.model_extra or {}
        return bool(opportunity.isAnonymized) or extra.get("deletedAt") is not None

    async def reconcile_deletions(self, *, days: int = 30) -> list[PurgeResult]:
        """Catch missed candidateDeleted webhooks: tombstone and purge every candidate
        Lever lists as deleted (GET /opportunities/deleted, ``deleted_at_start`` and
        ``deleted_at_end`` both inclusive, no window cap) over the last ``days``. The
        record's ``id`` is the opportunity id. Posting-level legal holds are honoured
        through the postings the candidate has stored evaluations for.

        /applications/deleted is deliberately not consulted: its rows are deleted
        applications, whose candidates may well still exist."""
        if days < 1:
            raise ValueError("days must be at least 1")
        lever = self._require_lever()
        end = self.clock()
        start = end - timedelta(days=days)
        found: list[str] = []
        stream = lever.list_deleted_opportunities(start_ms=int(start.timestamp() * 1000),
                                                  end_ms=int(end.timestamp() * 1000))
        async with aclosing(stream) as deleted:
            async for record in deleted:
                if record.id.strip() and record.id not in found:
                    found.append(record.id)
        results = [await self.handle_candidate_deleted(opp_id, source="reconcile")
                   for opp_id in found]
        logger.info("deletion reconciliation over %d day(s): %d opportunity(ies)", days,
                    len(results))
        return results

    def check_criteria(self, criteria: RoleCriteria, *, mode: RunMode) -> None:
        """The criteria gates: lint errors block every mode; criteria whose approval
        cannot be relied on (unapproved, or ``approved_at`` in the future at the clock's
        time) run only in EVALUATION mode. Raises RankingError."""
        errors = [issue for issue in lint_criteria(criteria, self._lexicon)
                  if issue.level == "error"]
        if errors:
            listed = "; ".join(f"{issue.location}: {issue.message}" for issue in errors[:5])
            more = f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""
            raise RankingError(
                f"criteria {criteria.role_id} v{criteria.version} has {len(errors)} lint "
                f"error(s); fix them (applicant-ranking criteria lint): {listed}{more}"
            )
        if RunMode(mode) is not RunMode.EVALUATION:
            issue = criteria.approval_issue(self.clock())
            if issue:
                raise RankingError(
                    f"criteria {criteria.role_id} v{criteria.version}: {issue}; TA/HR must "
                    "approve them before dry-run or live runs; use --mode evaluation to try "
                    "them out"
                )

    # ------------------------------------------------------------------ gates & setup

    def _require_lever(self) -> LeverClient:
        if self._lever is None:
            raise RankingError("no Lever client configured: set AR_LEVER_API_KEY")
        return self._lever

    def _no_criteria_message(self, posting_id: str) -> str:
        message = (f"no criteria for posting {posting_id}: run `applicant-ranking criteria "
                   f"draft --posting {posting_id}`, curate the file and have it approved")
        if self._criteria.errors:
            message += (f" ({len(self._criteria.errors)} criteria file problem(s) found; "
                        "run `applicant-ranking criteria lint`)")
        return message

    def _policy_decision(self, posting: Posting, mode: RunMode) -> PolicyDecision:
        today = policy_today(self.clock(), self._policy.detect_regimes(posting))
        decision = self._policy.evaluate(posting, mode=mode, today=today)
        if not decision.allowed:
            raise PolicyBlocked(decision)
        return decision

    def _context(self, criteria: RoleCriteria, *, mode: RunMode, posting_id: str,
                 posting: Posting, decision: PolicyDecision, force: bool,
                 defer_pending: bool = False) -> _Context:
        return _Context(mode=mode, criteria=criteria, criteria_hash=criteria.content_hash(),
                        posting_id=posting_id, job_description=job_description_text(posting),
                        job_description_lists=job_description_lists(posting),
                        policy_allowed=decision.allowed,
                        policy_regimes=list(decision.regimes), policy_gate=decision.allowed,
                        lookup_cache=not force, defer_pending=defer_pending)

    def _semaphores(self) -> tuple[asyncio.Semaphore, asyncio.Semaphore]:
        loop = asyncio.get_running_loop()
        if self._semaphores_for is None or self._semaphores_for[0] is not loop:
            self._semaphores_for = (loop, asyncio.Semaphore(self._io_concurrency),
                                    asyncio.Semaphore(self._settings.llm_concurrency))
        return self._semaphores_for[1], self._semaphores_for[2]

    @staticmethod
    async def _list_opportunities(lever: LeverClient, posting_id: str, *,
                                  include_archived: bool, limit: int | None,
                                  confidentiality: str | None = None) -> list[Opportunity]:
        found: dict[str, Opportunity] = {}
        stream = lever.list_opportunities(posting_id=posting_id,
                                          archived=None if include_archived else False,
                                          confidentiality=confidentiality)
        async with aclosing(stream) as opportunities:
            async for opportunity in opportunities:
                found.setdefault(opportunity.id, opportunity)
                if limit is not None and len(found) >= limit:
                    break
        return list(found.values())

    @staticmethod
    def _new_work(opportunity: Opportunity, posting_id: str | None) -> _Work:
        allowed = opportunity.storage_allowed
        return _Work(
            opportunity_id=opportunity.id,
            opportunity=opportunity,
            application_id=_application_id(opportunity, posting_id),
            candidate_name=opportunity.name if allowed else None,
            profile_url=opportunity.profile_url if allowed else None,
            names=[opportunity.name] if opportunity.name and opportunity.name.strip() else [],
            identifiers=contact_identifiers(opportunity),
        )

    # --------------------------------------------------------- step 4: resume gathering

    async def _collect(self, lever: LeverClient, work: _Work, ctx: _Context) -> None:
        """Skip checks, resume selection, download and preparation; then the vision step
        (outside the I/O limit). Failures become the work's ERROR evaluation; systemic
        ones (``is_systemic_error``) propagate and end the run."""
        io, _ = self._semaphores()
        try:
            if self._store.is_deleted(work.opportunity_id):
                work.evaluation = self._skipped(work, ctx.mode, ctx.posting_id, SKIP_DELETED,
                                                anonymous=True)
                return
            if work.opportunity is not None and not work.opportunity.record_valid:
                work.evaluation = self._skipped(work, ctx.mode, ctx.posting_id,
                                                SKIP_UNREADABLE, anonymous=True)
                return
            if work.opportunity is not None and not work.opportunity.storage_allowed:
                work.evaluation = self._skipped(work, ctx.mode, ctx.posting_id, SKIP_STORAGE,
                                                anonymous=True)
                return
            async with io:
                resumes = await lever.list_resumes(work.opportunity_id)
                if ctx.defer_pending and resume_pending(resumes):
                    logger.info("opportunity %s: resume still processing in Lever; deferred",
                                work.opportunity_id)
                    work.evaluation = self._skipped(work, ctx.mode, ctx.posting_id,
                                                    SKIP_RESUME_PENDING)
                    return
                work.prepared = await self._prepare_from_lever(lever, work, ctx, resumes)
            if work.prepared.needs_vision:
                await self._transcribe(work, ctx)
        except Exception as exc:
            if is_systemic_error(exc):
                raise
            work.evaluation = self._failed(work, ctx, exc)

    async def _prepare_from_lever(self, lever: LeverClient, work: _Work, ctx: _Context,
                                  resumes: Sequence[Resume]) -> PreparedResume:
        opp_id = work.opportunity_id
        resume = select_resume(resumes)
        if resume is None:
            newest = newest_resume(resumes)
            if newest is not None and _file_status(newest) in _UNPROCESSABLE_FILE_STATUSES:
                return self._no_resume(
                    opp_id, newest.id, "Lever could not process the resume file (status "
                    f"{_file_status(newest)!r})", unreadable=True)
            return self._no_resume(opp_id, None, "no processed resume file in Lever")
        file_name = resume.file.name if resume.file else None
        try:
            downloaded = await lever.download_resume(
                opp_id, resume.id, max_bytes=self._settings.max_resume_bytes)
        except LeverUnprocessable:
            text = parsed_data_to_text(resume.parsedData) if resume.parsedData else ""
            if not text.strip():
                return self._no_resume(
                    opp_id, resume.id, "Lever could not process the resume file and has no "
                    "parsed resume data", unreadable=True)
            logger.info("opportunity %s: resume download returned 422; using parsedData",
                        opp_id)
            return self._preparer.prepare_text(
                text, opportunity_id=opp_id, resume_id=resume.id, candidate_names=work.names,
                extra_identifiers=work.identifiers, job_description=ctx.job_description,
                job_description_lists=ctx.job_description_lists,
                source="parsed_data", file_name=file_name)
        except LeverFileTooLarge:
            limit_mb = self._settings.max_resume_bytes / (1024 * 1024)
            return self._no_resume(opp_id, resume.id, f"the resume file is larger than the "
                                   f"{limit_mb:g} MB limit", unreadable=True)
        return await self._preparer.prepare_file(
            downloaded.content, opportunity_id=opp_id, resume_id=resume.id,
            file_name=file_name, candidate_names=work.names,
            extra_identifiers=work.identifiers, job_description=ctx.job_description,
            job_description_lists=ctx.job_description_lists)

    def _no_resume(self, opportunity_id: str, resume_id: str | None, reason: str, *,
                   unreadable: bool = False) -> PreparedResume:
        flags = []
        if unreadable:
            flags.append(Flag(code=FlagCode.UNREADABLE_FILE, severity="review", detail=reason))
        flags.append(Flag(code=FlagCode.NO_RESUME, severity="review", detail=reason))
        return PreparedResume(opportunity_id=opportunity_id, resume_id=resume_id,
                              source="none",
                              redaction_version=self._preparer.redaction_version,
                              flags=flags, error=reason)

    async def _transcribe(self, work: _Work, ctx: _Context) -> None:
        """Vision step. The cache (keyed on the file HMAC, see ``vision_key``) is checked
        first so a scanned resume is not transcribed again for nothing. The transcript is
        prepared under the file's kind and HMAC; the file's own flags (scanned_resume,
        hidden text, injections found in its text layer) are kept. A systemic LLMError
        propagates."""
        prepared = work.prepared
        assert prepared is not None
        if prepared.file_hmac:
            work.vision_key = self._cache_key(work.opportunity_id, prepared.file_hmac, ctx,
                                              prepared.redaction_version)
        if ctx.lookup_cache and work.vision_key:
            cached = self._find_cached(work.vision_key, ctx.mode)
            if cached is not None:
                work.cached, work.cached_key = cached, work.vision_key
                work.prepared = prepared.model_copy(update={"page_images": []})
                return
        _, llm = self._semaphores()
        try:
            async with llm:
                transcript = await self._evaluator.transcribe(list(prepared.page_images))
        except LLMError as exc:
            if is_systemic_error(exc):
                raise
            logger.warning("transcription failed for opportunity %s: %s", work.opportunity_id,
                           type(exc).__name__)
            work.prepared = prepared.model_copy(update={"page_images": []})
            evaluation = self._error_evaluation(
                work, ctx, error=_error_text(exc, prefix="transcription failed: "),
                flags=[*prepared.flags, *_llm_error_flags(exc)], with_model=True,
                transient=exc.retryable)
            work.evaluation = self._commit(evaluation, ctx, cache_key=None)
            return
        source = prepared.source if prepared.source in ("file", "local_file") else "file"
        from_text = self._preparer.prepare_text(
            transcript, opportunity_id=work.opportunity_id, resume_id=prepared.resume_id,
            candidate_names=work.names, extra_identifiers=work.identifiers,
            job_description=ctx.job_description,
            job_description_lists=ctx.job_description_lists, source=source,
            file_hmac=prepared.file_hmac, kind=prepared.kind, file_name=prepared.file_name)
        work.prepared = from_text.model_copy(
            update={"flags": _dedupe_flags([*prepared.flags, *from_text.flags])})

    # --------------------------------------------------------------- step 5: signals

    @staticmethod
    def _apply_batch_signals(works: Sequence[_Work]) -> None:
        texts = {w.opportunity_id: w.prepared.redacted_text for w in works
                 if w.evaluation is None and w.prepared is not None
                 and w.prepared.redacted_text}
        opportunities = [w.opportunity for w in works
                         if w.opportunity is not None and w.opportunity.storage_allowed
                         and not (w.evaluation is not None
                                  and w.evaluation.status is Status.SKIPPED)]
        flags = batch_integrity_flags(texts, opportunities)
        for work in works:
            work.batch_flags = flags.get(work.opportunity_id, [])

    # ------------------------------------------------------- steps 6-7: score and write

    async def _finish(self, work: _Work, ctx: _Context) -> RankingRow:
        """One applicant: score, then write back (a systemic error propagates first)."""
        return await self._write_row(work, await self._score(work, ctx), ctx)

    async def _score(self, work: _Work, ctx: _Context) -> Evaluation:
        """Step 6. Failures become a recorded ERROR evaluation; systemic ones propagate."""
        try:
            return await self._evaluate(work, ctx)
        except Exception as exc:
            if is_systemic_error(exc):
                raise
            return self._failed(work, ctx, exc)

    async def _write_row(self, work: _Work, evaluation: Evaluation,
                         ctx: _Context) -> RankingRow:
        """Step 7: plan and apply the write-back (none for SKIPPED or local work)."""
        writeback: WritebackResult | None = None
        if work.opportunity is not None and evaluation.status is not Status.SKIPPED:
            io, _ = self._semaphores()
            async with io:
                writeback = await self._write_back(evaluation, work.opportunity, ctx)
        return RankingRow(rank=None, evaluation=evaluation, writeback=writeback)

    async def _evaluate(self, work: _Work, ctx: _Context) -> Evaluation:
        if work.evaluation is not None:
            return work.evaluation
        if work.cached is not None:
            return self._from_cache(work.cached, work, ctx, work.cached_key)
        prepared = work.prepared
        assert prepared is not None
        prior = _dedupe_flags([*prepared.flags, *work.batch_flags])
        if prepared.error or not prepared.redacted_text:
            evaluation = self._error_evaluation(
                work, ctx, error=prepared.error or "no usable resume text", flags=prior)
            return self._commit(evaluation, ctx, cache_key=None)

        key = work.vision_key or self._cache_key(
            work.opportunity_id, prepared.redacted_text_hmac or "", ctx,
            prepared.redaction_version)
        if ctx.lookup_cache:
            cached = self._find_cached(key, ctx.mode)
            if cached is not None:
                return self._from_cache(cached, work, ctx, key)

        evaluation, generation_config = await self._assess(work, ctx, prepared, prior)
        return self._commit(evaluation, ctx, cache_key=key if ctx.store_cache_key else None,
                            generation_config=generation_config,
                            redacted_text=prepared.redacted_text)

    async def _assess(self, work: _Work, ctx: _Context, prepared: PreparedResume,
                      prior: list[Flag]) -> tuple[Evaluation, dict[str, Any] | None]:
        evaluation_id = uuid.uuid4().hex
        request = AssessmentRequest(criteria=ctx.criteria, resume_text=prepared.redacted_text,
                                    evaluation_id=evaluation_id)
        _, llm = self._semaphores()
        async with llm:
            try:
                result = await self._evaluator.assess(request)
            except LLMError as exc:
                if is_systemic_error(exc):
                    raise
                logger.warning("assessment failed for opportunity %s: %s",
                               work.opportunity_id, type(exc).__name__)
                return self._error_evaluation(
                    work, ctx, error=_error_text(exc), flags=[*prior, *_llm_error_flags(exc)],
                    evaluation_id=evaluation_id, with_model=True,
                    transient=exc.retryable), None
            outcome = await asyncio.to_thread(self._validate, ctx, result, prepared, prior,
                                              retries=0)
            usage = result.usage
            if outcome.corrective_note:
                retry = request.model_copy(update={"corrective_note": outcome.corrective_note})
                try:
                    second = await self._evaluator.assess(retry)
                except LLMError as exc:
                    if is_systemic_error(exc):
                        raise
                    logger.warning("corrective retry failed for opportunity %s (%s); keeping "
                                   "the first assessment with its rationale withheld",
                                   work.opportunity_id, type(exc).__name__)
                else:
                    carried = carried_review_signals(outcome.flags)
                    outcome = await asyncio.to_thread(self._validate, ctx, second, prepared,
                                                      [*prior, *carried], retries=1)
                    usage = _add_usage(usage, second.usage)
                    result = second
        evaluation = Evaluation(
            evaluation_id=evaluation_id,
            created_at=self.clock(),
            mode=ctx.mode,
            posting_id=ctx.posting_id,
            opportunity_id=work.opportunity_id,
            application_id=work.application_id,
            resume_id=prepared.resume_id,
            candidate_name=work.candidate_name,
            profile_url=work.profile_url,
            criteria_role_id=ctx.criteria.role_id,
            criteria_version=ctx.criteria.version,
            criteria_hash=ctx.criteria_hash,
            prompt_version=result.prompt_version,
            redaction_version=prepared.redaction_version or None,
            model_id=result.model_id,
            model_version=result.model_version,
            status=outcome.status,
            score=outcome.score,
            tier=outcome.tier,
            holistic_score=outcome.holistic_score,
            confidence=outcome.confidence,
            criteria_results=outcome.criteria_results,
            knockout_results=outcome.knockout_results,
            strengths=outcome.strengths,
            gaps=outcome.gaps,
            summary=outcome.summary,
            flags=outcome.flags,
            review_reasons=outcome.review_reasons,
            rationale_publishable=outcome.rationale_publishable,
            validation=outcome.validation,
            usage=usage,
            file_hmac=prepared.file_hmac,
            redacted_text_hmac=prepared.redacted_text_hmac,
        )
        return evaluation, dict(result.generation_config)

    def _validate(self, ctx: _Context, result: LLMCallResult, prepared: PreparedResume,
                  prior: list[Flag], *, retries: int) -> ScoringOutcome:
        """Pure CPU work (``_assess`` runs it in a worker thread)."""
        return validate_and_score(
            criteria=ctx.criteria, assessment=result.assessment,
            resume_text=prepared.redacted_text, lexicon=self._lexicon,
            thresholds=self._settings.tiers,
            disagreement_threshold=self._settings.disagreement_threshold,
            prior_flags=prior, corrective_retries=retries)

    def _cache_key(self, opportunity_id: str, content_hmac: str, ctx: _Context,
                   redaction_version: str) -> str:
        return cache_key(opportunity_id=opportunity_id, content_hmac=content_hmac,
                         criteria_hash=ctx.criteria_hash,
                         prompt_version=self._evaluator.prompt_version,
                         model_id=self._evaluator.model_id,
                         redaction_version=redaction_version or REDACTION_VERSION,
                         lexicon_version=self._lexicon_version)

    def _find_cached(self, key: str, mode: RunMode) -> Evaluation | None:
        """Newest reusable result of the same kind: an EVALUATION-mode result (watermarked,
        possibly from unapproved criteria) is never reused for a DRY_RUN/LIVE run, nor the
        reverse."""
        evaluation_mode = mode is RunMode.EVALUATION
        cached = self._store.find_cached(key, evaluation_mode=evaluation_mode)
        if cached is None or (cached.mode is RunMode.EVALUATION) != evaluation_mode:
            return None
        return cached

    def _from_cache(self, cached: Evaluation, work: _Work, ctx: _Context,
                    key: str | None) -> Evaluation:
        """The cached evaluation unchanged, or a derived evaluation recorded under a new
        id (no model call) when this run's batch signals raise a flag it lacks, its tier
        or score-disagreement flag differs under the current thresholds, or a LIVE run
        reuses a non-LIVE result. ``generation_config`` names the reasons and
        ``derived_from``. The derived row stores the prepared text when there is one;
        a vision hit (cache checked before transcription) has none, so its text stays
        with the original row."""
        reasons: list[str] = []
        update: dict[str, Any] = {}
        flags = list(cached.flags)
        present = {flag.code for flag in flags}
        missing = [flag for flag in work.batch_flags if flag.code not in present]
        if missing:
            reasons.append("batch integrity signals")
            flags = _dedupe_flags([*flags, *missing])
            status = cached.status
            if any(flag.severity == "review" for flag in missing):
                status = Status.worst(status, Status.NEEDS_REVIEW)
            review = list(cached.review_reasons)
            review.extend(r for r in _review_reasons(missing) if r not in review)
            update.update(status=status, review_reasons=review)
        tier = (tier_for(cached.score, self._settings.tiers) if cached.score is not None
                else None)
        if tier != cached.tier:
            reasons.append("tier thresholds changed")
            update["tier"] = tier
        rescored = self._disagreement_flags(cached.score, cached.holistic_score, flags)
        if rescored != flags:
            reasons.append("score disagreement threshold changed")
            flags = rescored
        if ctx.mode is RunMode.LIVE and cached.mode is not RunMode.LIVE:
            reasons.append("reused for a live run")
        if not reasons:
            logger.debug("opportunity %s: cache hit %s", work.opportunity_id,
                         cached.evaluation_id)
            return cached
        derived = cached.model_copy(update={
            **update,
            "evaluation_id": uuid.uuid4().hex,
            "created_at": self.clock(),
            "mode": ctx.mode,
            "flags": flags,
        })
        reason = "; ".join(reasons)
        logger.info("opportunity %s: cache hit %s recorded as %s (%s)", work.opportunity_id,
                    cached.evaluation_id, derived.evaluation_id, reason)
        prepared = work.prepared
        return self._commit(derived, ctx, cache_key=key,
                            generation_config={"derived_from": cached.evaluation_id,
                                               "reason": reason},
                            redacted_text=(prepared.redacted_text or None) if prepared
                            else None)

    def _disagreement_flags(self, score: int | None, holistic: int | None,
                            flags: list[Flag]) -> list[Flag]:
        """``flags`` with the info-level ``score_disagreement`` flag present exactly when
        validation would set it under the current threshold (same detail wording)."""
        has = any(flag.code is FlagCode.SCORE_DISAGREEMENT for flag in flags)
        due = (score is not None and holistic is not None
               and abs(score - holistic) >= self._settings.disagreement_threshold)
        if has == due:
            return flags
        if not due:
            return [flag for flag in flags if flag.code is not FlagCode.SCORE_DISAGREEMENT]
        return [*flags, Flag(code=FlagCode.SCORE_DISAGREEMENT, severity="info",
                             detail=f"computed score {score} vs model's holistic {holistic}")]

    async def _write_back(self, evaluation: Evaluation, opportunity: Opportunity,
                          ctx: _Context) -> WritebackResult:
        try:
            plan = self._planner.plan(
                evaluation, current_tags=list(opportunity.tags),
                existing_note_id=self._store.get_note_id(opportunity.id))
            return await self._writer.apply(plan, mode=ctx.mode,
                                            policy_allowed=ctx.policy_allowed)
        except Exception as exc:
            logger.warning("write-back for opportunity %s failed: %s at %s", opportunity.id,
                           type(exc).__name__, _where(exc))
            return WritebackResult(opportunity_id=opportunity.id,
                                   evaluation_id=evaluation.evaluation_id, executed=False,
                                   error=_error_text(exc, prefix="write-back failed: "))

    # ------------------------------------------------------------- evaluation records

    def _base_fields(self, work: _Work, ctx: _Context) -> dict[str, Any]:
        return {
            "created_at": self.clock(),
            "mode": ctx.mode,
            "posting_id": ctx.posting_id,
            "opportunity_id": work.opportunity_id,
            "application_id": work.application_id,
            "candidate_name": work.candidate_name,
            "profile_url": work.profile_url,
            "criteria_role_id": ctx.criteria.role_id,
            "criteria_version": ctx.criteria.version,
            "criteria_hash": ctx.criteria_hash,
        }

    def _skipped(self, work: _Work, mode: RunMode, posting_id: str | None, reason: str, *,
                 anonymous: bool = False) -> Evaluation:
        return Evaluation(
            evaluation_id=uuid.uuid4().hex,
            created_at=self.clock(),
            mode=mode,
            posting_id=posting_id,
            opportunity_id=work.opportunity_id,
            application_id=work.application_id,
            candidate_name=None if anonymous else work.candidate_name,
            profile_url=None if anonymous else work.profile_url,
            status=Status.SKIPPED,
            review_reasons=[reason],
        )

    def _error_evaluation(self, work: _Work, ctx: _Context, *, error: str,
                          flags: Iterable[Flag], evaluation_id: str | None = None,
                          with_model: bool = False, transient: bool = False) -> Evaluation:
        prepared = work.prepared
        unique = _dedupe_flags(flags)
        return Evaluation(
            evaluation_id=evaluation_id or uuid.uuid4().hex,
            **self._base_fields(work, ctx),
            resume_id=prepared.resume_id if prepared else None,
            prompt_version=self._evaluator.prompt_version if with_model else None,
            redaction_version=(prepared.redaction_version or None) if prepared else None,
            model_id=self._evaluator.model_id if with_model else None,
            status=Status.ERROR,
            flags=unique,
            review_reasons=_review_reasons(unique),
            file_hmac=prepared.file_hmac if prepared else None,
            redacted_text_hmac=prepared.redacted_text_hmac if prepared else None,
            error=shorten(error, MAX_ERROR_CHARS),
            transient=transient,
        )

    def _failed(self, work: _Work, ctx: _Context, exc: Exception) -> Evaluation:
        """Per-opportunity isolation: an unexpected failure becomes a recorded ERROR."""
        logger.error("evaluation of opportunity %s failed: %s at %s", work.opportunity_id,
                     type(exc).__name__, _where(exc))
        flags = [*(work.prepared.flags if work.prepared else []), *work.batch_flags]
        evaluation = self._error_evaluation(work, ctx, error=_error_text(exc), flags=flags,
                                            transient=_is_transient(exc))
        try:
            return self._commit(evaluation, ctx, cache_key=None)
        except Exception as store_exc:
            logger.error("could not record the failed evaluation of opportunity %s: %s",
                         work.opportunity_id, type(store_exc).__name__)
            return evaluation

    def _commit(self, evaluation: Evaluation, ctx: _Context, *, cache_key: str | None,
                generation_config: dict[str, Any] | None = None,
                redacted_text: str | None = None) -> Evaluation:
        """Record (when the run records) unless the candidate was deleted meanwhile, in
        which case nothing is stored and a SKIPPED evaluation is returned."""
        if not ctx.record:
            return evaluation
        if self._store.is_deleted(evaluation.opportunity_id):
            logger.info("opportunity %s was deleted during the run; result discarded",
                        evaluation.opportunity_id)
            return Evaluation(evaluation_id=evaluation.evaluation_id,
                              created_at=evaluation.created_at, mode=evaluation.mode,
                              posting_id=evaluation.posting_id,
                              opportunity_id=evaluation.opportunity_id,
                              application_id=evaluation.application_id,
                              status=Status.SKIPPED, review_reasons=[SKIP_DELETED])
        self._store.record_evaluation(evaluation, cache_key=cache_key,
                                      generation_config=generation_config,
                                      redacted_text=redacted_text,
                                      policy_regimes=ctx.policy_regimes,
                                      policy_allowed=ctx.policy_gate)
        return evaluation
