"""Plan and (only in live mode) send the Lever write-back for one evaluation.

Tags (Lever can filter by tag, not sort; tags are the only write-back recruiters can
filter on). All managed tags start with ``f"{tag_prefix}: "`` and exactly one of the
primary tags is present after a write:
- ``"{p}: Tier A"`` .. ``"{p}: Tier D"`` for SCORED / KNOCKOUT_FLAGGED evaluations with a
  publishable rationale,
- ``"{p}: Needs review"`` for NEEDS_REVIEW (the tier tag is still added when the rationale
  is publishable; withheld when it is not),
- ``"{p}: Not scored"`` for ERROR, except a transient one (below).
Plus ``"{p}: Verify knockout"`` when any knockout is "unmet", and
``"{p}: Score 80-89"``-style band tags when ``write_score_band_tag``.
Stale managed tags (in the namespace but not desired) are removed; tags outside the
namespace are never touched. SKIPPED evaluations produce an empty plan (no requests).
A transient ERROR (``Evaluation.transient``: a Lever or model outage, not the applicant)
on an opportunity that already carries a primary managed tag (a tier, "Needs review" or
"Not scored") produces an empty plan too, with ``REASON_TRANSIENT``: its tags and note
are left unchanged. Without such a tag it gets "Not scored" as usual.

Note (plain text, <= note_max_chars, truncated with "…" and a pointer to the full
report when longer): first line "AI ranking (advisory, not a decision) - review the full
application."; then tier/score/status, per-requirement lines "- [rating] description
(evidence: "…")", knockout checks, strengths, gaps, flags (system flag codes in plain
words; hidden-text snippets are NOT quoted), summary, and a footer with evaluation id,
criteria role/version, model and prompt version. If ``rationale_publishable`` is False
the note says only that the automated assessment was withheld for human review (no
model text, no score). ERROR: a one-line reason. Update the note we wrote before
(``existing_note_id`` -> PUT) instead of adding a new one.

Implementation choices beyond the above:
- The primary tag is always ``desired_tags(...)[0]``. A SCORED/KNOCKOUT_FLAGGED
  evaluation whose rationale is not publishable gets "Needs review" (its tier is
  withheld). "Verify knockout" and band tags are model-derived, so they are withheld
  with the tier.
- Lengths are counted in UTF-16 code units (what Lever's JavaScript stack counts), which
  is never less than the code-point count; text is cut on code-point boundaries only.
- Flag ``detail`` strings are never written, except the detail of an info-severity
  ``model_red_flag`` (validated, length-limited model text). Raw ``error`` strings and
  ``review_reasons`` are never written either: they are not validated output.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime
from urllib.parse import quote

from applicant_ranking.config import ConfigError, Settings
from applicant_ranking.domain import (
    Evaluation,
    Flag,
    FlagCode,
    PlannedRequest,
    RunMode,
    Status,
    Tier,
    WritebackPlan,
    WritebackResult,
    utcnow,
)
from applicant_ranking.lever.client import LeverClient, LeverError, LeverNotFound
from applicant_ranking.scoring.compute import score_band, tier_for
from applicant_ranking.store import Store

logger = logging.getLogger(__name__)

ADVISORY_LINE = "AI ranking (advisory, not a decision) - review the full application."
EVALUATION_WATERMARK = "EVALUATION ONLY - not for hiring decisions"
WITHHELD_TEXT = (
    "The automated assessment was withheld for human review: its output did not pass "
    "validation, so no score or rationale is shown here."
)
TRUNCATION_POINTER = "(Note truncated - the full assessment is in the ranking report.)"
ELLIPSIS = "…"

# dry_run_reason values (the report shows NOTHING_TO_WRITE as "-").
NOTHING_TO_WRITE = "nothing to write"
REASON_EVALUATION = "evaluation mode never writes to Lever"
REASON_DRY_RUN = "dry run: write-back planned, not sent"
REASON_WRITE_DISABLED = "write-back disabled (AR_WRITE_ENABLED is not true)"
REASON_POLICY = "blocked by the jurisdiction policy for this posting"
REASON_NO_CLIENT = "no Lever client configured"
REASON_TRANSIENT = "transient error: existing AI ranking tags and note left unchanged"

PRIMARY_NEEDS_REVIEW = "Needs review"
PRIMARY_NOT_SCORED = "Not scored"
VERIFY_KNOCKOUT = "Verify knockout"

FLAG_WORDS: dict[FlagCode, str] = {
    FlagCode.HIDDEN_TEXT: "Hidden text found in the resume file",
    FlagCode.INJECTION_PHRASE: "Instruction-like text found in the resume",
    FlagCode.UNICODE_TAG_PAYLOAD: "Invisible encoded characters found in the resume",
    FlagCode.INVISIBLE_CHARACTERS: "Invisible characters removed from the resume",
    FlagCode.MIXED_SCRIPT: "Words mixing alphabets (possible look-alike characters)",
    FlagCode.HIDDEN_TEXT_CHECK_UNAVAILABLE: "Hidden-text check not available for this file",
    FlagCode.SCANNED_RESUME: "Scanned resume (text transcribed from page images)",
    FlagCode.TRUNCATED: "Resume text was truncated before evaluation",
    FlagCode.UNSUPPORTED_FILE: "Resume file type not supported",
    FlagCode.UNREADABLE_FILE: "Resume file could not be read",
    FlagCode.NO_RESUME: "No resume found",
    FlagCode.RESUME_FROM_PARSED_DATA: "Evaluated from Lever's parsed resume data, not the file",
    FlagCode.DUPLICATE_RESUME: "Same resume text as another application",
    FlagCode.CONTACT_REUSE: "Contact details shared with another applicant",
    FlagCode.JD_COPY: "Resume closely copies the job description",
    FlagCode.UNGROUNDED_EVIDENCE: "Some model evidence was not found in the resume",
    FlagCode.PROTECTED_TERM_IN_OUTPUT: "Model output mentioned a protected characteristic",
    FlagCode.INVALID_MODEL_OUTPUT: "Model output was invalid",
    FlagCode.MODEL_BLOCKED: "Model declined to evaluate",
    FlagCode.SCORE_DISAGREEMENT: "Computed score and the model's overall estimate disagree",
    FlagCode.LOW_CONFIDENCE: "Model reported low confidence",
    FlagCode.INSUFFICIENT_INFORMATION: "Too little information for the must-have requirements",
    FlagCode.MODEL_INJECTION_REPORT: "Model reported an attempt to manipulate the evaluation",
    FlagCode.MODEL_RED_FLAG: "Model-reported concern",
}

# First match wins for the one-line ERROR reason. A file that exists but could not be used
# (unsupported, unreadable, rejected by Lever) comes before the plain "no resume": such
# evaluations carry both flags, and "no resume was found" would misdescribe them.
_ERROR_REASONS: tuple[tuple[FlagCode, str], ...] = (
    (FlagCode.UNSUPPORTED_FILE, "the resume file type is not supported"),
    (FlagCode.UNREADABLE_FILE, "the resume file could not be read"),
    (FlagCode.NO_RESUME, "no resume was found"),
    (FlagCode.MODEL_BLOCKED, "the model declined to evaluate this application"),
    (FlagCode.INVALID_MODEL_OUTPUT, "the model returned invalid output"),
)
_DEFAULT_ERROR_REASON = "the application could not be evaluated automatically"

_STATUS_WORDS: dict[Status, str] = {
    Status.SCORED: "scored",
    Status.KNOCKOUT_FLAGGED: "knockout flagged - a human must verify",
    Status.NEEDS_REVIEW: "needs human review",
    Status.ERROR: "not scored",
    Status.SKIPPED: "not evaluated",
}

_WHITESPACE = re.compile(r"\s+")
_LONE_SURROGATE = re.compile(r"[\ud800-\udfff]")
_MAX_QUOTES = 2
_QUOTE_CHARS = 80
_ITEM_CHARS = 300
_SUMMARY_CHARS = 800
_RED_FLAG_DETAIL_CHARS = 200


# ---------------------------------------------------------------------------- text helpers


def utf16_len(text: str) -> int:
    """Length in UTF-16 code units (astral characters count 2)."""
    return sum(2 if ord(ch) > 0xFFFF else 1 for ch in text)


def _clean_line(text: str) -> str:
    """One line of plain text: lone surrogates replaced, control and format characters
    (bidi overrides, zero-width, Unicode tag characters) dropped, whitespace collapsed."""
    text = _LONE_SURROGATE.sub("\ufffd", text)
    text = "".join(
        " " if ch.isspace() else ch
        for ch in text
        if ch.isspace() or unicodedata.category(ch) not in ("Cc", "Cf", "Cs", "Co", "Cn")
    )
    return _WHITESPACE.sub(" ", text).strip()


def _cut(text: str, max_units: int) -> str:
    """Longest prefix of ``text`` within ``max_units`` UTF-16 units, cut on a code-point
    boundary and backed off so it never ends on a joiner or before a combining mark."""
    if max_units <= 0:
        return ""
    units = 0
    end = 0
    for i, ch in enumerate(text):
        size = 2 if ord(ch) > 0xFFFF else 1
        if units + size > max_units:
            break
        units += size
        end = i + 1
    while 0 < end < len(text) and (
        unicodedata.combining(text[end]) or text[end] in "\u200d\ufe0e\ufe0f"
    ):
        end -= 1
    while end > 0 and text[end - 1] == "\u200d":
        end -= 1
    return text[:end]


def shorten(text: str, max_units: int) -> str:
    """One clean line, at most ``max_units`` UTF-16 units, ending with "…" if cut
    (preferably at a word boundary)."""
    line = _clean_line(text)
    if utf16_len(line) <= max_units:
        return line
    head = _cut(line, max_units - 1)
    space = head.rfind(" ")
    if space >= len(head) * 2 // 3:
        head = head[:space]
    return head.rstrip() + ELLIPSIS


def _truncate_block(text: str, max_units: int) -> str:
    """Cut a multi-line block to ``max_units`` including the trailing "…", preferring the
    end of a line when that keeps at least half of the room."""
    if utf16_len(text) <= max_units:
        return text
    head = _cut(text, max_units - 2)
    newline = head.rfind("\n")
    if newline >= len(head) // 2:
        return head[:newline].rstrip() + "\n" + ELLIPSIS
    return _cut(text, max_units - 1).rstrip() + ELLIPSIS


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _segment(value: str) -> str:
    return quote(value, safe="")


# ------------------------------------------------------------------------------- planner


class WritebackPlanner:
    def __init__(self, settings: Settings) -> None:
        prefix = settings.tag_prefix.strip()
        if not prefix:
            raise ConfigError("AR_TAG_PREFIX must not be empty (it namespaces managed tags)")
        self._settings = settings
        self._prefix = prefix

    def namespace(self) -> str:
        """``f"{tag_prefix}: "``."""
        return f"{self._prefix}: "

    def _tag(self, label: str) -> str:
        return f"{self.namespace()}{label}"

    def primary_tags(self) -> list[str]:
        """Every managed tag that can be ``desired_tags(...)[0]``."""
        return [*(self._tag(f"Tier {tier.value}") for tier in Tier),
                self._tag(PRIMARY_NEEDS_REVIEW), self._tag(PRIMARY_NOT_SCORED)]

    def _tier(self, evaluation: Evaluation) -> Tier | None:
        if evaluation.tier is not None:
            return evaluation.tier
        if evaluation.score is not None:
            return tier_for(evaluation.score, self._settings.tiers)
        return None

    def desired_tags(self, evaluation: Evaluation) -> list[str]:
        """Managed tags the opportunity should carry; the first one is the primary tag.
        Empty for SKIPPED."""
        status = evaluation.status
        if status is Status.SKIPPED:
            return []
        if status is Status.ERROR:
            return [self._tag(PRIMARY_NOT_SCORED)]

        tier = self._tier(evaluation) if evaluation.rationale_publishable else None
        tags: list[str] = []
        if status is Status.NEEDS_REVIEW or tier is None:
            tags.append(self._tag(PRIMARY_NEEDS_REVIEW))
        if tier is not None:
            tags.append(self._tag(f"Tier {tier.value}"))
            if self._settings.write_score_band_tag and evaluation.score is not None:
                low, high = score_band(evaluation.score)
                tags.append(self._tag(f"Score {low}-{high}"))
        if evaluation.rationale_publishable and (
            status is Status.KNOCKOUT_FLAGGED
            or any(k.result == "unmet" for k in evaluation.knockout_results)
        ):
            tags.append(self._tag(VERIFY_KNOCKOUT))
        return tags

    # --------------------------------------------------------------------------- note

    def _footer(self, evaluation: Evaluation) -> str:
        criteria = "-"
        if evaluation.criteria_role_id:
            criteria = evaluation.criteria_role_id
            if evaluation.criteria_version is not None:
                criteria += f" v{evaluation.criteria_version}"
        model = evaluation.model_id or "-"
        if evaluation.model_version and evaluation.model_version != evaluation.model_id:
            model += f" ({evaluation.model_version})"
        return shorten(
            f"Ref: evaluation {evaluation.evaluation_id} | criteria {criteria} | "
            f"model {model} | prompt {evaluation.prompt_version or '-'}",
            400,
        )

    def _status_line(self, evaluation: Evaluation) -> str:
        tier = self._tier(evaluation)
        parts = [
            f"Tier {tier.value}" if tier is not None else "Tier -",
            f"Score {evaluation.score}/100" if evaluation.score is not None else "Score -",
            f"Status: {_STATUS_WORDS[evaluation.status]}",
        ]
        if evaluation.confidence:
            parts.append(f"Confidence: {evaluation.confidence}")
        return " | ".join(parts)

    @staticmethod
    def _evidence(quotes: list[str]) -> str:
        shown = [f'"{shorten(q, _QUOTE_CHARS)}"' for q in quotes[:_MAX_QUOTES] if q.strip()]
        if not shown:
            return ""
        return f" (evidence: {'; '.join(shown)})"

    @staticmethod
    def _flag_lines(flags: list[Flag]) -> list[str]:
        lines: list[str] = []
        for flag in flags:
            words = FLAG_WORDS.get(flag.code, flag.code.value.replace("_", " ").capitalize())
            if flag.code is FlagCode.MODEL_RED_FLAG and flag.severity == "info" and flag.detail:
                words = f"{words}: {shorten(flag.detail, _RED_FLAG_DETAIL_CHARS)}"
            if flag.severity == "review":
                words += " (needs review)"
            lines.append(f"- {words}")
        return _dedupe(lines)

    def _body(self, evaluation: Evaluation) -> list[str]:
        lines: list[str] = []
        if evaluation.criteria_results:
            lines.append("Requirements:")
            for cr in evaluation.criteria_results:
                lines.append(f"- [{cr.rating}] {shorten(cr.description, _ITEM_CHARS)}"
                             f"{self._evidence(cr.evidence)}")
        if evaluation.knockout_results:
            lines.append("Knockout checks (never applied automatically):")
            for kc in evaluation.knockout_results:
                verify = " - a human must verify" if kc.result == "unmet" else ""
                lines.append(f"- [{kc.result}] {shorten(kc.description, _ITEM_CHARS)}{verify}"
                             f"{self._evidence(kc.evidence)}")
        for title, items in (("Strengths:", evaluation.strengths), ("Gaps:", evaluation.gaps)):
            cleaned = [shorten(item, _ITEM_CHARS) for item in items]
            cleaned = [item for item in cleaned if item]
            if cleaned:
                lines.append(title)
                lines.extend(f"- {item}" for item in cleaned)
        flag_lines = self._flag_lines(evaluation.flags)
        if flag_lines:
            lines.append("Flags:")
            lines.extend(flag_lines)
        summary = shorten(evaluation.summary, _SUMMARY_CHARS)
        if summary:
            lines.append(f"Summary: {summary}")
        return lines

    def render_note(self, evaluation: Evaluation) -> str:
        """Deterministic plain-text note; see the module docstring for the layout."""
        head = [ADVISORY_LINE]
        if evaluation.mode is RunMode.EVALUATION:
            head.append(EVALUATION_WATERMARK)

        body: list[str]
        if evaluation.status is Status.SKIPPED:
            body = ["Not evaluated: this application was skipped."]
        elif evaluation.status is Status.ERROR:
            body = [f"Not scored: {self._error_reason(evaluation)}. "
                    "Review the application manually."]
        elif not evaluation.rationale_publishable:
            body = [WITHHELD_TEXT]
        else:
            head.append(self._status_line(evaluation))
            body = self._body(evaluation)

        return self._assemble(head, body, self._footer(evaluation))

    @staticmethod
    def _error_reason(evaluation: Evaluation) -> str:
        codes = {flag.code for flag in evaluation.flags}
        for code, reason in _ERROR_REASONS:
            if code in codes:
                return reason
        return _DEFAULT_ERROR_REASON

    def _assemble(self, head: list[str], body: list[str], footer: str) -> str:
        limit = self._settings.note_max_chars
        head_text = "\n".join(head)
        body_text = "\n".join(body)
        full = "\n".join(part for part in (head_text, body_text) if part) + "\n\n" + footer
        if utf16_len(full) <= limit:
            return full
        tail = f"\n{TRUNCATION_POINTER}\n\n{footer}"
        room = limit - utf16_len(head_text) - 1 - utf16_len(tail)
        if room >= 40:
            note = f"{head_text}\n{_truncate_block(body_text, room)}{tail}"
        else:  # pathological: very long head/footer
            note = _truncate_block(full, limit)
        return note if utf16_len(note) <= limit else _truncate_block(note, limit)

    # --------------------------------------------------------------------------- plan

    def _params(self) -> dict[str, str]:
        perform_as = self._settings.lever_perform_as
        return {"perform_as": perform_as} if perform_as else {}

    def plan(self, evaluation: Evaluation, *, current_tags: list[str],
             existing_note_id: str | None) -> WritebackPlan:
        """Requests in order: removeTags (if any), addTags (if any), note POST or PUT.
        Paths are relative to the base URL; perform_as goes in params when configured.
        Empty for SKIPPED, and for a transient ERROR when ``current_tags`` has a primary
        managed tag (``reason`` = REASON_TRANSIENT)."""
        opp = evaluation.opportunity_id
        if evaluation.status is Status.SKIPPED:
            return WritebackPlan(opportunity_id=opp, evaluation_id=evaluation.evaluation_id)

        current = _dedupe(current_tags)
        if evaluation.status is Status.ERROR and evaluation.transient:
            primary = set(self.primary_tags())
            if any(tag in primary for tag in current):
                return WritebackPlan(opportunity_id=opp,
                                     evaluation_id=evaluation.evaluation_id,
                                     reason=REASON_TRANSIENT)

        namespace = self.namespace()
        desired = self.desired_tags(evaluation)
        to_remove = [t for t in current if t.startswith(namespace) and t not in desired]
        to_add = [t for t in desired if t not in current]
        note_text = self.render_note(evaluation)

        base = f"/opportunities/{_segment(opp)}"
        params = self._params()
        requests: list[PlannedRequest] = []
        if to_remove:
            requests.append(PlannedRequest(
                method="POST", path=f"{base}/removeTags", params=dict(params),
                json_body={"tags": list(to_remove)}, purpose="remove stale AI ranking tags"))
        if to_add:
            requests.append(PlannedRequest(
                method="POST", path=f"{base}/addTags", params=dict(params),
                json_body={"tags": list(to_add)}, purpose="add AI ranking tags"))
        if existing_note_id:
            requests.append(PlannedRequest(
                method="PUT", path=f"{base}/notes/{_segment(existing_note_id)}",
                params=dict(params),
                json_body={"values": [{"value": note_text}],
                           "secret": self._settings.note_secret},
                purpose="update the AI ranking note"))
        else:
            requests.append(PlannedRequest(
                method="POST", path=f"{base}/notes", params=dict(params),
                json_body={"value": note_text, "secret": self._settings.note_secret,
                           "notifyFollowers": self._settings.note_notify_followers},
                purpose="add the AI ranking note"))

        return WritebackPlan(
            opportunity_id=opp,
            evaluation_id=evaluation.evaluation_id,
            tags_to_add=to_add,
            tags_to_remove=to_remove,
            note_text=note_text,
            existing_note_id=existing_note_id or None,
            requests=requests,
        )


# -------------------------------------------------------------------------------- writer


class _SendFailed(Exception):
    pass


class LeverWriter:
    def __init__(self, client: LeverClient | None, store: Store, settings: Settings, *,
                 clock: Callable[[], datetime] = utcnow) -> None:
        self._client = client
        self._store = store
        self._settings = settings
        self._clock = clock
        self._namespace = f"{settings.tag_prefix.strip()}: "

    def _blocked_reason(self, plan: WritebackPlan, mode: RunMode, policy_allowed: bool,
                        ) -> str | None:
        if not plan.requests:
            return plan.reason or NOTHING_TO_WRITE
        if mode is RunMode.EVALUATION:
            return REASON_EVALUATION
        if mode is not RunMode.LIVE:
            return REASON_DRY_RUN
        if not self._settings.write_enabled:
            return REASON_WRITE_DISABLED
        if not policy_allowed:
            return REASON_POLICY
        return None

    async def apply(self, plan: WritebackPlan, *, mode: RunMode,
                    policy_allowed: bool) -> WritebackResult:
        """Send only when mode is LIVE, ``settings.write_enabled``, ``policy_allowed`` and a
        client is present; otherwise return ``executed=False`` with ``dry_run_reason``.
        EVALUATION mode can never send. Sends via the client's typed methods (tags, then
        note); a PUT that fails with LeverNotFound falls back to creating a new note.
        Stores the note id and the result. A failure is captured in ``error`` (not
        raised) so one candidate can't abort a batch."""
        reason = self._blocked_reason(plan, RunMode(mode), policy_allowed)
        client = self._client
        if reason is None and client is not None:
            result = await self._send(client, plan)
        else:
            result = WritebackResult(opportunity_id=plan.opportunity_id,
                                     evaluation_id=plan.evaluation_id, executed=False,
                                     dry_run_reason=reason or REASON_NO_CLIENT)
        self._record(result)
        logger.info("write-back opportunity=%s evaluation=%s executed=%s reason=%s error=%s",
                    result.opportunity_id, result.evaluation_id, result.executed,
                    result.dry_run_reason, bool(result.error))
        return result

    async def _send(self, client: LeverClient, plan: WritebackPlan) -> WritebackResult:
        result = WritebackResult(opportunity_id=plan.opportunity_id,
                                 evaluation_id=plan.evaluation_id, executed=True)
        unmanaged = [t for t in (*plan.tags_to_remove, *plan.tags_to_add)
                     if not t.startswith(self._namespace)]
        if unmanaged:
            result.executed = False
            result.error = (f"refused: {len(unmanaged)} tag(s) outside the "
                            f"'{self._namespace}' namespace")
            return result

        opp = plan.opportunity_id
        base = f"/opportunities/{_segment(opp)}"
        try:
            if plan.tags_to_remove:
                await self._call(result, "POST", f"{base}/removeTags",
                                 client.remove_tags(opp, list(plan.tags_to_remove)))
            if plan.tags_to_add:
                await self._call(result, "POST", f"{base}/addTags",
                                 client.add_tags(opp, list(plan.tags_to_add)))
            if plan.note_text is not None:
                result.note_id = await self._write_note(client, plan, result, base)
        except _SendFailed:
            pass
        return result

    async def _write_note(self, client: LeverClient, plan: WritebackPlan,
                          result: WritebackResult, base: str) -> str:
        opp = plan.opportunity_id
        value = plan.note_text or ""
        secret = self._settings.note_secret
        note_id: str | None = None
        if plan.existing_note_id:
            put_path = f"{base}/notes/{_segment(plan.existing_note_id)}"
            try:
                note_id = await self._call(
                    result, "PUT", put_path,
                    client.update_note(opp, plan.existing_note_id, value=value, secret=secret),
                    not_found_ok=True)
            except LeverNotFound:
                logger.info("note for opportunity %s no longer exists; creating a new one", opp)
        if note_id is None:
            note_id = await self._call(
                result, "POST", f"{base}/notes",
                client.create_note(opp, value=value, secret=secret,
                                   notify_followers=self._settings.note_notify_followers))
        if not note_id:
            result.error = "Lever returned no note id"
            raise _SendFailed
        try:
            self._store.set_note_id(opp, str(note_id), now=self._clock())
        except Exception as exc:  # the note exists in Lever; keep going, report it
            logger.exception("could not store the note id for opportunity %s", opp)
            result.error = f"note written but its id could not be stored: {type(exc).__name__}"
        return str(note_id)

    @staticmethod
    async def _call[T](result: WritebackResult, method: str, path: str,
                       awaitable: Awaitable[T], *, not_found_ok: bool = False) -> T:
        """Await one Lever call and log its outcome in ``result.responses``. Failures set
        ``result.error`` and raise _SendFailed (LeverNotFound is re-raised as-is when
        ``not_found_ok``, for the PUT -> POST fallback)."""
        try:
            value = await awaitable
        except Exception as exc:
            not_found = isinstance(exc, LeverNotFound)
            status = exc.status if isinstance(exc, LeverError) else None
            result.responses.append({"method": method, "path": path, "status": "error",
                                     "http_status": status or (404 if not_found else None)})
            if not_found and not_found_ok:
                raise
            result.error = _describe(method, path, exc)
            logger.warning("write-back %s %s failed: %s", method, path, type(exc).__name__)
            raise _SendFailed from exc
        result.responses.append({"method": method, "path": path, "status": "ok"})
        return value

    def _record(self, result: WritebackResult) -> None:
        try:
            self._store.record_writeback(result, now=self._clock())
        except Exception as exc:
            logger.exception("could not record the write-back result for opportunity %s",
                             result.opportunity_id)
            note = f"result not recorded: {type(exc).__name__}"
            result.error = f"{result.error}; {note}" if result.error else note


def _describe(method: str, path: str, exc: BaseException) -> str:
    status = getattr(exc, "status", None)
    detail = shorten(str(exc), 200) if isinstance(exc, LeverError) else ""
    parts = [f"{method} {path} failed: {type(exc).__name__}"]
    if status:
        parts.append(f"HTTP {status}")
    if detail:
        parts.append(detail)
    return " - ".join(parts)
