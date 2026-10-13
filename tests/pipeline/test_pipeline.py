"""Ranker end to end: fake Lever over MockTransport, real store/preparer/lexicon/validation,
scripted model (see conftest.py)."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import logging
import re
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from applicant_ranking.config import TierThresholds
from applicant_ranking.domain import (
    Evaluation,
    Flag,
    FlagCode,
    RankingReport,
    RankingRow,
    RoleCriteria,
    RunMode,
    Status,
    Tier,
)
from applicant_ranking.lever.client import LeverAuthError, LeverError
from applicant_ranking.lever.models import Opportunity, Posting, Resume
from applicant_ranking.lever.webhooks import compute_signature
from applicant_ranking.llm.base import (
    AssessmentRequest,
    LLMBlockedError,
    LLMCallResult,
    LLMConfigError,
    LLMOutputError,
    LLMUnavailableError,
)
from applicant_ranking.llm.schema import RedFlag
from applicant_ranking.pipeline import (
    NO_APPLICANTS_NOTE,
    SKIP_DELETED,
    SKIP_NO_CRITERIA,
    SKIP_NO_POSTING,
    SKIP_RESUME_PENDING,
    SKIP_STORAGE,
    SKIP_UNREADABLE,
    PolicyBlocked,
    RankingError,
    batch_integrity_flags,
    cache_key,
    carried_review_signals,
    contact_identifiers,
    job_description_lists,
    job_description_text,
    lexicon_fingerprint,
    newest_resume,
    policy_today,
    resume_pending,
    select_resume,
)
from applicant_ranking.policy import (
    CA_FEHA,
    CO_ADMT,
    IL_HRA,
    NYC_LL144,
    IllinoisPolicy,
    NycPolicy,
    PolicyConfig,
    PolicyEngine,
)
from applicant_ranking.report import audit_export_csv, writeback_label
from applicant_ranking.scoring.compute import tier_for
from applicant_ranking.scoring.lexicon import ProtectedTermLexicon
from applicant_ranking.service.app import create_app
from applicant_ranking.wiring import Components
from applicant_ranking.writeback import (
    ADVISORY_LINE,
    EVALUATION_WATERMARK,
    REASON_DRY_RUN,
    REASON_EVALUATION,
    REASON_TRANSIENT,
    REASON_WRITE_DISABLED,
)

HEX32 = re.compile(r"^[0-9a-f]{32}$")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
NAMES_AND_CONTACTS = ("Alice", "Novak", "alice.novak@example.com", "555-0100", "Marsh",
                      "Kerr", "Patel", "Osei", "frank.osei@example.com")


@pytest.fixture
def kit(pipeline_kit: SimpleNamespace) -> SimpleNamespace:
    return pipeline_kit


def by_opp(report: RankingReport) -> dict[str, RankingRow]:
    return {row.evaluation.opportunity_id: row for row in report.rows}


def codes(evaluation: Evaluation) -> set[FlagCode]:
    return {flag.code for flag in evaluation.flags}


def unapproved(criteria: RoleCriteria) -> RoleCriteria:
    return criteria.model_copy(update={"approved_by": None, "approved_at": None})


# ------------------------------------------------------------------------------- helpers


def test_cache_key_is_sha256_of_the_documented_parts() -> None:
    key = cache_key(opportunity_id="opp-1", content_hmac="h", criteria_hash="c",
                    prompt_version="p", model_id="m", redaction_version="r",
                    lexicon_version="l")
    assert key == hashlib.sha256(b"opp-1|h|c|p|m|r|l").hexdigest()
    assert key != cache_key(opportunity_id="opp-1", content_hmac="h", criteria_hash="c",
                            prompt_version="p", model_id="m2", redaction_version="r",
                            lexicon_version="l")


def _resume(resume_id: str, *, status: str | None, uploaded: int | None,
            created: int | None = None, file: bool = True) -> Resume:
    data: dict[str, Any] = {"id": resume_id, "createdAt": created}
    if file:
        data["file"] = {"name": "cv.pdf", "status": status, "uploadedAt": uploaded}
    return Resume.model_validate(data)


def test_select_resume_prefers_the_newest_processed_file() -> None:
    resumes = [
        _resume("old", status="processed", uploaded=10, created=5),
        _resume("tie-older", status="processed", uploaded=20, created=1),
        _resume("tie-newer", status="Processed", uploaded=20, created=2),
        _resume("processing", status="processing", uploaded=30),
        _resume("no-status", status=None, uploaded=40),
        _resume("error", status="error", uploaded=50),
        _resume("no-file", status=None, uploaded=None, file=False),
    ]
    chosen = select_resume(resumes)
    assert chosen is not None and chosen.id == "tie-newer"
    assert select_resume(resumes[3:]) is None
    assert select_resume([]) is None


def test_resume_pending_follows_the_newest_file_only() -> None:
    processed = _resume("done", status="processed", uploaded=20)
    processing = _resume("parsing", status="processing", uploaded=30)
    no_status = _resume("no-status", status=None, uploaded=40)
    failed = _resume("failed", status="error", uploaded=50)
    unsupported = _resume("unsupported", status="Unsupported", uploaded=60)
    replaced = _resume("replaced", status="processed", uploaded=70)
    no_file = _resume("no-file", status=None, uploaded=None, created=99, file=False)

    assert resume_pending([]) and resume_pending([no_file])  # nothing uploaded yet
    assert resume_pending([processing]) and resume_pending([no_status])
    assert resume_pending([processed, processing])  # a newer upload is still parsing
    assert resume_pending([processed, no_status, no_file])
    assert not resume_pending([processing, replaced])  # the processed one is newer
    assert not resume_pending([processed, failed]) and not resume_pending([unsupported])
    assert newest_resume([processed, processing, no_file]) is processing
    assert newest_resume([no_file]) is None and newest_resume([]) is None
    assert newest_resume([
        _resume("tie-a", status="processing", uploaded=10, created=1),
        _resume("tie-b", status="processed", uploaded=10, created=2),
    ]).id == "tie-b"  # type: ignore[union-attr]


def test_job_description_text_and_contact_identifiers() -> None:
    posting = Posting.model_validate({
        "id": "p", "content": {"description": "Intro.", "lists": [
            {"text": "Requirements", "content": "<li>Python</li>"}], "closing": "Bye."}})
    jd = job_description_text(posting)
    assert jd.split("\n") == ["Intro.", "Requirements", "<li>Python</li>", "Bye."]
    assert job_description_lists(posting) == "Requirements\n<li>Python</li>"
    assert job_description_lists(Posting.model_validate({"id": "p"})) == ""
    opp = Opportunity.model_validate({
        "id": "o", "emails": ["a@example.com", " ", "a@example.com"],
        "phones": [{"value": "(312) 555-0100"}, {"type": "home"}, {"value": ""}]})
    assert contact_identifiers(opp) == ["a@example.com", "(312) 555-0100"]


def test_batch_integrity_flags() -> None:
    text = "Built BigQuery pipelines in Python for retail clients."
    opps = [
        Opportunity(id="a", name="Ann Lee", emails=["Ann@Example.com"],
                    phones=[{"value": "+1 (312) 555-0199"}]),
        Opportunity(id="b", name="Bo Chan", emails=["ann@example.com"]),
        Opportunity(id="c", name=" ann  LEE ", emails=["ann@example.com"]),  # same person
        Opportunity(id="d", name="Di Ray", phones=[{"value": "312.555.0199"}]),
        Opportunity(id="e", name=None, emails=["ann@example.com"]),  # no name: not compared
        Opportunity(id="f", name="Fay Wu", phones=[{"value": "12"}]),  # too short to match
    ]
    flags = batch_integrity_flags({"a": text, "b": text, "c": "", "d": "Other text."}, opps)

    def summary(opp_id: str) -> list[tuple[FlagCode, str]]:
        return [(f.code, f.detail) for f in flags.get(opp_id, [])]

    assert all(f.severity == "review" for fs in flags.values() for f in fs)
    assert summary("a") == [
        (FlagCode.DUPLICATE_RESUME, "same or near-identical resume text as 1 other "
                                    "application(s) for this posting: b"),
        (FlagCode.CONTACT_REUSE, "email address shared with 1 application(s) under a "
                                 "different name: b"),
        (FlagCode.CONTACT_REUSE, "phone number shared with 1 application(s) under a "
                                 "different name: d"),
    ]
    assert summary("b")[1] == (FlagCode.CONTACT_REUSE, "email address shared with 2 "
                                                       "application(s) under a different "
                                                       "name: a, c")
    assert [c for c, _ in summary("c")] == [FlagCode.CONTACT_REUSE]  # empty text: no dup
    assert [c for c, _ in summary("d")] == [FlagCode.CONTACT_REUSE]
    assert "e" not in flags and "f" not in flags
    details = " ".join(f.detail for fs in flags.values() for f in fs)
    assert "example.com" not in details and "555" not in details


# ---------------------------------------------------------------------- rank_posting


async def test_dry_run_ranks_flags_and_sends_nothing(make_harness: Callable[..., Any],
                                                     kit: SimpleNamespace,
                                                     sample_criteria: RoleCriteria) -> None:
    h = make_harness()
    report = await h.ranker.rank_posting(kit.POSTING_ID)

    assert report.mode is RunMode.DRY_RUN
    assert report.posting_title == "Senior Data Engineer"
    assert (report.criteria_role_id, report.criteria_version) == ("senior-data-engineer", 2)
    assert report.generated_at == kit.NOW
    assert report.policy_regimes == ["IL_HRA"]
    assert any(note.startswith("IL HRA") for note in report.policy_notes)
    assert report.watermark == "Advisory ranking - not a decision. Review every application."
    assert [(r.rank, r.evaluation.opportunity_id) for r in report.rows] == [
        (1, "opp-alice"), (2, "opp-erin"), (3, "opp-bob"), (4, "opp-dana"), (5, "opp-frank"),
        (None, "opp-hal"),
    ]  # the archived applicant is not listed; the anonymized one is listed, unranked
    rows = by_opp(report)
    assert {opp: rows[opp].evaluation.score for opp in kit.EXPECTED_SCORES} == kit.EXPECTED_SCORES
    assert [rows[o].evaluation.tier for o in ("opp-alice", "opp-erin", "opp-bob", "opp-frank")
            ] == [Tier.A, Tier.B, Tier.C, Tier.D]

    alice = rows["opp-alice"].evaluation
    assert alice.status is Status.SCORED and alice.rationale_publishable
    assert alice.validation.corrective_retries == 1  # protected term on the first answer
    assert alice.summary != kit.PROTECTED_SUMMARY
    assert FlagCode.PROTECTED_TERM_IN_OUTPUT not in codes(alice)
    assert alice.knockout_results[0].result == "met" and alice.knockout_results[0].evidence
    assert HEX32.match(alice.evaluation_id)
    assert (alice.candidate_name, alice.profile_url) == (
        "Alice Novak", "https://hire.lever.co/candidates/opp-alice")
    assert (alice.posting_id, alice.application_id, alice.resume_id) == (
        kit.POSTING_ID, "app-opp-alice", "res-alice-new")
    assert alice.criteria_hash == sample_criteria.content_hash()
    assert (alice.model_id, alice.prompt_version, alice.redaction_version) == (
        "fake-model", "p1-fake", h.preparer.redaction_version)
    assert alice.redaction_version.endswith("+eduyears=1+max=30000")
    assert alice.file_hmac and alice.redacted_text_hmac and alice.created_at == kit.NOW

    for opp in ("opp-bob", "opp-dana"):
        evaluation = rows[opp].evaluation
        assert evaluation.status is Status.NEEDS_REVIEW
        assert {FlagCode.DUPLICATE_RESUME, FlagCode.CONTACT_REUSE} <= codes(evaluation)
        assert len(evaluation.review_reasons) == 2
    erin = rows["opp-erin"].evaluation
    assert erin.status is Status.SCORED and FlagCode.RESUME_FROM_PARSED_DATA in codes(erin)
    assert erin.file_hmac is None and erin.redacted_text_hmac
    frank = rows["opp-frank"].evaluation
    assert frank.status is Status.SCORED and FlagCode.SCANNED_RESUME in codes(frank)
    assert frank.file_hmac  # the transcript is filed under the scanned file's HMAC
    assert len(h.evaluator.transcribe_calls) == 1
    assert all(img.startswith(PNG_SIGNATURE) for img in h.evaluator.transcribe_calls[0])

    hal = rows["opp-hal"]
    assert hal.evaluation.status is Status.SKIPPED and hal.writeback is None
    assert hal.evaluation.review_reasons == [SKIP_STORAGE]
    assert hal.evaluation.candidate_name is None and hal.evaluation.profile_url is None

    # The model saw redacted text only, and was called once per applicant plus one retry.
    assert len(h.evaluator.requests) == 6
    for request in h.evaluator.requests:
        assert not any(secret in request.resume_text for secret in NAMES_AND_CONTACTS)
    retry = [r for r in h.evaluator.requests if r.corrective_note]
    assert len(retry) == 1 and kit.ALICE_MARKER in retry[0].resume_text
    assert kit.ALICE_MARKER not in (retry[0].corrective_note or "")

    # Lever: nothing written; anonymized resumes never fetched; newest processed file used.
    assert h.api.writes == []
    gets = h.api.paths("GET")
    assert not any("opp-hal/" in path or "opp-gina" in path for path in gets)
    assert "/opportunities/opp-alice/resumes/res-alice-new/download" in gets
    assert not any("res-alice-old" in p or "res-alice-processing" in p for p in gets)
    listing = next(r for r in h.api.requests if r.path == "/opportunities")
    assert listing.params == {"posting_id": kit.POSTING_ID, "expand": "applications",
                              "archived": "false", "limit": "100"}

    # Store: every evaluated applicant is audited (skipped ones are not); plans recorded.
    audit = h.store.audit_rows(posting_id=kit.POSTING_ID)
    assert sorted(row["opportunity_id"] for row in audit) == sorted(kit.EXPECTED_SCORES)
    assert {row["mode"] for row in audit} == {"dry_run"}
    assert h.store.get_evaluation(alice.evaluation_id) == alice
    for row in report.rows[:5]:
        assert row.writeback is not None
        assert (row.writeback.executed, row.writeback.dry_run_reason, row.writeback.error) == (
            False, REASON_DRY_RUN, None)
    assert h.store.get_note_id("opp-alice") is None


async def test_ties_break_on_created_at_then_id(make_harness: Callable[..., Any],
                                                kit: SimpleNamespace) -> None:
    api = kit.populate_posting(kit.FakeLever())
    api.opportunities["opp-dana"]["createdAt"] = api.opportunities["opp-bob"]["createdAt"] - 1
    h = make_harness(api=api)
    rows = (await h.ranker.rank_posting(kit.POSTING_ID)).rows
    assert [(r.rank, r.evaluation.opportunity_id) for r in rows[2:4]] == [
        (3, "opp-dana"), (4, "opp-bob")]

    api2 = kit.populate_posting(kit.FakeLever())
    api2.opportunities["opp-dana"]["createdAt"] = api2.opportunities["opp-bob"]["createdAt"]
    rows2 = (await make_harness(api=api2).ranker.rank_posting(kit.POSTING_ID)).rows
    assert [r.evaluation.opportunity_id for r in rows2[2:4]] == ["opp-bob", "opp-dana"]


async def test_include_archived_and_limit(make_harness: Callable[..., Any],
                                          kit: SimpleNamespace) -> None:
    h = make_harness()
    report = await h.ranker.rank_posting(kit.POSTING_ID, include_archived=True)
    assert "opp-gina" in by_opp(report)
    listing = next(r for r in h.api.requests if r.path == "/opportunities")
    assert "archived" not in listing.params

    limited = await h.ranker.rank_posting(kit.POSTING_ID, limit=2)
    assert {r.evaluation.opportunity_id for r in limited.rows} == {"opp-alice", "opp-bob"}
    with pytest.raises(ValueError, match="limit"):
        await h.ranker.rank_posting(kit.POSTING_ID, limit=0)


async def test_live_sends_tags_and_notes_then_updates_the_note(
        make_harness: Callable[..., Any], kit: SimpleNamespace) -> None:
    h = make_harness(write_enabled=True)
    first = await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE)
    rows = by_opp(first)
    for opp in kit.EXPECTED_SCORES:
        wb = rows[opp].writeback
        assert wb is not None and wb.executed and wb.error is None and wb.note_id
        assert h.store.get_note_id(opp) == wb.note_id
    assert rows["opp-hal"].writeback is None
    assert not any("opp-hal" in w.path or "opp-gina" in w.path for w in h.api.writes)

    alice_writes = [w for w in h.api.writes if w.path.startswith("/opportunities/opp-alice/")]
    assert [(w.method, w.path) for w in alice_writes] == [
        ("POST", "/opportunities/opp-alice/removeTags"),
        ("POST", "/opportunities/opp-alice/addTags"),
        ("POST", "/opportunities/opp-alice/notes"),
    ]
    assert alice_writes[0].json == {"tags": ["AI Rank: Tier C"]}
    assert alice_writes[1].json == {"tags": ["AI Rank: Tier A"]}
    note = alice_writes[2].json
    assert note["value"].startswith(ADVISORY_LINE)
    assert (note["secret"], note["notifyFollowers"]) == (False, False)
    assert "Alice" not in note["value"] and "score" not in note
    assert h.api.opportunities["opp-alice"]["tags"] == ["Referral", "AI Rank: Tier A"]
    assert "AI Rank: Needs review" in h.api.opportunities["opp-bob"]["tags"]

    calls, transcriptions = len(h.evaluator.requests), len(h.evaluator.transcribe_calls)
    writes_before = len(h.api.writes)
    second = await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE)
    assert len(h.evaluator.requests) == calls  # cache hits
    assert len(h.evaluator.transcribe_calls) == transcriptions  # checked before vision
    for opp in kit.EXPECTED_SCORES:
        assert by_opp(second)[opp].evaluation == rows[opp].evaluation
    new_writes = h.api.writes[writes_before:]
    assert {w.method for w in new_writes} == {"PUT"}  # tags already right; notes updated
    alice_note_id = rows["opp-alice"].writeback.note_id  # type: ignore[union-attr]
    put = next(w for w in new_writes if "opp-alice" in w.path)
    assert put.path == f"/opportunities/opp-alice/notes/{alice_note_id}"
    assert put.json == {"values": [{"value": note["value"]}], "secret": False}
    assert len(h.api.notes["opp-alice"]) == 1
    assert len(h.store.audit_rows()) == len(kit.EXPECTED_SCORES)  # nothing re-recorded


async def test_live_without_write_enabled_only_plans(make_harness: Callable[..., Any],
                                                     kit: SimpleNamespace) -> None:
    h = make_harness()
    report = await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE)
    assert h.api.writes == []
    assert {r.writeback.dry_run_reason for r in report.rows if r.writeback} == {
        REASON_WRITE_DISABLED}


async def test_writeback_failure_is_isolated(make_harness: Callable[..., Any],
                                             kit: SimpleNamespace) -> None:
    h = make_harness(write_enabled=True)
    h.api.failures[("POST", "/opportunities/opp-erin/addTags")] = 500
    rows = by_opp(await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE))
    erin = rows["opp-erin"].writeback
    assert erin is not None and erin.error and "HTTP 500" in erin.error
    assert rows["opp-erin"].evaluation.status is Status.SCORED
    assert not any(w.path == "/opportunities/opp-erin/notes" for w in h.api.writes)
    alice = rows["opp-alice"].writeback
    assert alice is not None and alice.executed and alice.error is None


# --------------------------------------------------------------------------- gates


async def test_evaluation_mode_allows_unapproved_criteria_and_never_writes(
        make_harness: Callable[..., Any], kit: SimpleNamespace,
        sample_criteria: RoleCriteria) -> None:
    h = make_harness(criteria=unapproved(sample_criteria), write_enabled=True)
    report = await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.EVALUATION)
    assert report.watermark == EVALUATION_WATERMARK
    assert {r.evaluation.mode for r in report.rows} == {RunMode.EVALUATION}
    assert {r.writeback.dry_run_reason for r in report.rows if r.writeback} == {
        REASON_EVALUATION}
    assert h.api.writes == []
    assert {row["mode"] for row in h.store.audit_rows()} == {"evaluation"}

    for mode in (RunMode.DRY_RUN, RunMode.LIVE):
        requests_before = len(h.api.requests)
        with pytest.raises(RankingError, match="not approved"):
            await h.ranker.rank_posting(kit.POSTING_ID, mode=mode)
        assert [r.path for r in h.api.requests[requests_before:]] == [
            f"/postings/{kit.POSTING_ID}"]


async def test_future_approval_date_blocks_dry_run_and_live(
        make_harness: Callable[..., Any], kit: SimpleNamespace,
        sample_criteria: RoleCriteria) -> None:
    future = sample_criteria.model_copy(update={"approved_at": kit.NOW + timedelta(days=1)})
    h = make_harness(criteria=future, write_enabled=True)

    for mode in (RunMode.DRY_RUN, RunMode.LIVE):
        with pytest.raises(RankingError, match="is in the future"):
            await h.ranker.rank_posting(kit.POSTING_ID, mode=mode)
        with pytest.raises(RankingError, match="is in the future"):
            await h.ranker.evaluate_opportunity("opp-alice", mode=mode)
    assert h.evaluator.requests == [] and h.api.writes == []

    report = await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.EVALUATION)
    assert report.rows and report.mode is RunMode.EVALUATION

    h.clock.now = kit.NOW + timedelta(days=2)  # the approval time has passed
    assert (await h.ranker.rank_posting(kit.POSTING_ID)).rows


async def test_unreadable_lever_record_is_a_visible_skipped_row(
        make_harness: Callable[..., Any], kit: SimpleNamespace) -> None:
    api = kit.populate_posting(kit.FakeLever())
    lines = ["Nula Null", "Data Engineer, Nulls Inc (2020 - present)",
             "Maintained BigQuery tables and wrote Python loaders."]
    broken = api.add_opportunity("opp-broken", name="Broken Record", created=50,
                                 emails=["broken.record@example.com"])
    broken["tags"] = [1, 2]  # not strings: the record fails validation
    api.add_resume("opp-broken", "res-broken", kit.text_pdf(lines), uploaded=60)
    nulls = api.add_opportunity("opp-nulls", name="Nula Null", created=51)
    nulls["tags"] = None  # a null list means "empty": scored like anyone else
    api.add_resume("opp-nulls", "res-nulls", kit.text_pdf(lines), uploaded=60)
    unknown = api.add_opportunity("opp-anon-null", name="Ann Unknown", created=52)
    unknown["isAnonymized"] = None  # unknown: storage fails closed
    api.add_resume("opp-anon-null", "res-anon", kit.text_pdf(kit.BOB_LINES), uploaded=60)
    h = make_harness(api=api, write_enabled=True)

    rows = by_opp(await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE))

    broken_row = rows["opp-broken"]
    assert broken_row.rank is None and broken_row.evaluation.status is Status.SKIPPED
    assert broken_row.evaluation.review_reasons == [SKIP_UNREADABLE]
    assert broken_row.evaluation.candidate_name is None
    assert broken_row.evaluation.profile_url is None and broken_row.writeback is None
    assert rows["opp-nulls"].rank is not None and rows["opp-nulls"].evaluation.score
    anon = rows["opp-anon-null"]
    assert anon.evaluation.status is Status.SKIPPED
    assert anon.evaluation.review_reasons == [SKIP_STORAGE]
    assert anon.evaluation.candidate_name is None and anon.writeback is None
    audited = {row["opportunity_id"] for row in h.store.audit_rows()}
    assert "opp-nulls" in audited and not {"opp-broken", "opp-anon-null"} & audited
    assert not any("opp-broken" in w.path or "opp-anon-null" in w.path for w in h.api.writes)
    assert not any("opp-broken/" in p or "opp-anon-null/" in p for p in h.api.paths())
    dumped = repr([r.evaluation.model_dump(mode="json") for r in rows.values()])
    assert "Broken Record" not in dumped and "Ann Unknown" not in dumped
    assert "AI Rank" in " ".join(api.opportunities["opp-nulls"]["tags"] or [])


async def test_no_applicants_note_and_confidential_posting_listing(
        make_harness: Callable[..., Any], kit: SimpleNamespace) -> None:
    empty = kit.populate_posting(kit.FakeLever())
    empty.opportunities.clear()
    h = make_harness(api=empty)
    report = await h.ranker.rank_posting(kit.POSTING_ID)
    assert report.rows == [] and report.notes == [NO_APPLICANTS_NOTE]
    listing = next(r for r in h.api.requests if r.path == "/opportunities")
    assert "confidentiality" not in listing.params  # Lever's default for an ordinary posting

    secret = kit.populate_posting(kit.FakeLever())
    secret.postings[kit.POSTING_ID]["confidentiality"] = "confidential"
    h2 = make_harness(api=secret)
    report = await h2.ranker.rank_posting(kit.POSTING_ID)
    assert len(report.rows) == 6 and report.notes == []
    listing = next(r for r in h2.api.requests if r.path == "/opportunities")
    assert listing.params["confidentiality"] == "all"


async def test_policy_blocks_dry_run_and_live_but_not_evaluation(
        make_harness: Callable[..., Any], kit: SimpleNamespace) -> None:
    h = make_harness(policy=kit.approved_policy(illinois=False), write_enabled=True)
    for mode in (RunMode.DRY_RUN, RunMode.LIVE):
        with pytest.raises(PolicyBlocked) as info:
            await h.ranker.rank_posting(kit.POSTING_ID, mode=mode)
        assert isinstance(info.value, RankingError)
        assert any(b.startswith("IL HRA") for b in info.value.decision.blockers)
        assert "IL HRA" in str(info.value)
    assert h.evaluator.requests == [] and "/opportunities" not in h.api.paths()

    report = await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.EVALUATION)
    assert any(n.startswith("EVALUATION only - DRY_RUN/LIVE would be blocked: IL HRA")
               for n in report.policy_notes)
    assert h.api.writes == []

    missing = make_harness(policy=PolicyEngine(None, source="config/policy.yaml"))
    with pytest.raises(PolicyBlocked, match="No jurisdiction policy file"):
        await missing.ranker.rank_posting(kit.POSTING_ID)


@pytest.mark.parametrize("mode", list(RunMode))
async def test_lint_errors_block_every_mode(make_harness: Callable[..., Any],
                                            kit: SimpleNamespace,
                                            sample_criteria: RoleCriteria,
                                            mode: RunMode) -> None:
    bad = sample_criteria.model_copy(update={
        "summary": "Looking for a young digital native to join the team."})
    h = make_harness(criteria=bad)
    with pytest.raises(RankingError, match="lint error"):
        await h.ranker.rank_posting(kit.POSTING_ID, mode=mode)
    with pytest.raises(RankingError, match="lint error"):
        await h.ranker.evaluate_local(criteria=bad, text="Python and BigQuery engineer.")
    assert h.evaluator.requests == []


async def test_missing_criteria_and_missing_lever(make_harness: Callable[..., Any],
                                                  kit: SimpleNamespace) -> None:
    h = make_harness(criteria=None)
    with pytest.raises(RankingError, match="criteria draft --posting posting-1"):
        await h.ranker.rank_posting(kit.POSTING_ID)

    offline = make_harness(with_lever=False)
    with pytest.raises(RankingError, match="AR_LEVER_API_KEY"):
        await offline.ranker.rank_posting(kit.POSTING_ID)
    with pytest.raises(RankingError, match="AR_LEVER_API_KEY"):
        await offline.ranker.evaluate_opportunity("opp-alice")
    with pytest.raises(RankingError, match="AR_LEVER_API_KEY"):
        await offline.ranker.reconcile_deletions()


# ----------------------------------------------------------------- per-applicant edges


async def test_resume_edge_cases_become_error_rows(make_harness: Callable[..., Any],
                                                   kit: SimpleNamespace) -> None:
    api = kit.populate_posting(kit.FakeLever())
    api.add_opportunity("opp-ivy", name="Ivy None", created=1)
    api.add_opportunity("opp-jo", name="Jo Pending", created=2)
    api.add_resume("opp-jo", "res-jo", None, uploaded=5, status="processing")
    api.add_opportunity("opp-kim", name="Kim Broken", created=3)
    api.add_resume("opp-kim", "res-kim", 422, uploaded=5)
    api.add_opportunity("opp-lou", name="Lou Failed", created=4)
    api.add_resume("opp-lou", "res-lou-old", None, uploaded=5, status="processing")
    api.add_resume("opp-lou", "res-lou", None, uploaded=6, status="error")
    api.add_opportunity("opp-max", name="Max Image", created=5)
    api.add_resume("opp-max", "res-max", None, uploaded=5, status="unsupported", name="cv.jpg")
    h = make_harness(api=api)
    rows = by_opp(await h.ranker.rank_posting(kit.POSTING_ID))

    # A batch run does not wait for Lever: a file still processing is "no resume" here.
    for opp in ("opp-ivy", "opp-jo"):
        evaluation = rows[opp].evaluation
        assert evaluation.status is Status.ERROR and rows[opp].rank is None
        assert codes(evaluation) == {FlagCode.NO_RESUME}
        assert evaluation.error == "no processed resume file in Lever"
        assert not evaluation.transient
    kim = rows["opp-kim"].evaluation
    assert kim.status is Status.ERROR
    assert codes(kim) == {FlagCode.UNREADABLE_FILE, FlagCode.NO_RESUME}
    # A file Lever gave up on is reported as unreadable, with the file named, not as
    # "no resume was found".
    for opp, status in (("opp-lou", "error"), ("opp-max", "unsupported")):
        evaluation = rows[opp].evaluation
        assert evaluation.status is Status.ERROR
        assert codes(evaluation) == {FlagCode.UNREADABLE_FILE, FlagCode.NO_RESUME}
        assert evaluation.error == f"Lever could not process the resume file (status {status!r})"
        assert evaluation.resume_id == f"res-{opp.removeprefix('opp-')}"
        note = h.planner.render_note(evaluation)
        assert "Not scored: the resume file could not be read" in note
    assert not any(f"/{opp}/resumes/" in p for opp in ("opp-jo", "opp-lou", "opp-max")
                   for p in h.api.paths())
    assert len(h.evaluator.requests) == 6  # only the usual applicants reached the model
    audited = {row["opportunity_id"]: row["status"] for row in h.store.audit_rows()}
    assert audited["opp-ivy"] == audited["opp-kim"] == audited["opp-lou"] == "error"
    # unranked rows are listed after the ranked ones, oldest first
    assert [r.evaluation.opportunity_id for r in (await h.ranker.rank_posting(
        kit.POSTING_ID)).rows if r.rank is None] == [
        "opp-ivy", "opp-jo", "opp-kim", "opp-lou", "opp-max", "opp-hal"]


async def test_oversized_file_is_unreadable(make_harness: Callable[..., Any],
                                            kit: SimpleNamespace) -> None:
    h = make_harness(max_resume_bytes=1024)
    rows = by_opp(await h.ranker.rank_posting(kit.POSTING_ID))
    alice = rows["opp-alice"].evaluation
    assert alice.status is Status.ERROR and FlagCode.UNREADABLE_FILE in codes(alice)
    assert "larger than" in (alice.error or "")
    assert rows["opp-erin"].evaluation.status is Status.SCORED  # parsedData, no file


async def test_llm_errors_become_error_rows_without_aborting(
        make_harness: Callable[..., Any], kit: SimpleNamespace) -> None:
    responder = kit.ScriptedResponder(errors={
        kit.BOB_MARKER: LLMBlockedError("blocked: SAFETY"),
        kit.ERIN_MARKER: LLMOutputError("schema mismatch at requirements.0.rating"),
        kit.FRANK_MARKER: LLMUnavailableError("HTTP 429 quota exhausted"),
    })
    h = make_harness(responder=responder)
    report = await h.ranker.rank_posting(kit.POSTING_ID)
    rows = by_opp(report)
    assert [(r.rank, r.evaluation.opportunity_id) for r in report.rows] == [
        (1, "opp-alice"), (None, "opp-bob"), (None, "opp-dana"), (None, "opp-erin"),
        (None, "opp-frank"), (None, "opp-hal")]
    bob = rows["opp-bob"].evaluation
    assert bob.status is Status.ERROR and bob.score is None
    assert bob.error == "LLMBlockedError: blocked: SAFETY"
    assert {FlagCode.MODEL_BLOCKED, FlagCode.DUPLICATE_RESUME} <= codes(bob)
    assert (bob.model_id, bob.prompt_version) == ("fake-model", "p1-fake")
    erin = rows["opp-erin"].evaluation
    assert FlagCode.INVALID_MODEL_OUTPUT in codes(erin) and erin.status is Status.ERROR
    frank = rows["opp-frank"].evaluation
    assert frank.error == "LLMUnavailableError: HTTP 429 quota exhausted (retryable)"
    assert codes(frank) == {FlagCode.SCANNED_RESUME}
    assert rows["opp-bob"].writeback is not None  # "Not scored" is planned
    audited = {row["opportunity_id"]: row["status"] for row in h.store.audit_rows()}
    assert audited == {"opp-alice": "scored", "opp-bob": "error", "opp-dana": "error",
                       "opp-erin": "error", "opp-frank": "error"}

    # Errors are never served from the cache: the next run asks the model again.
    responder.errors.clear()
    calls = len(h.evaluator.requests)
    rerun = by_opp(await h.ranker.rank_posting(kit.POSTING_ID))
    assert len(h.evaluator.requests) == calls + 4  # bob, dana, erin, frank; alice cached
    assert {o: rerun[o].evaluation.score for o in kit.EXPECTED_SCORES} == kit.EXPECTED_SCORES


async def test_unexpected_failures_are_isolated_and_logged_safely(
        make_harness: Callable[..., Any], kit: SimpleNamespace,
        caplog: pytest.LogCaptureFixture) -> None:
    secret = "SECRET-RESUME-TEXT Gamma Retail"
    h = make_harness(responder=kit.ScriptedResponder(
        errors={kit.ERIN_MARKER: RuntimeError(secret)}))
    h.api.failures[("GET", "/opportunities/opp-bob/resumes")] = 500
    with caplog.at_level(logging.DEBUG):
        rows = by_opp(await h.ranker.rank_posting(kit.POSTING_ID))
    bob = rows["opp-bob"].evaluation
    assert bob.status is Status.ERROR
    assert (bob.error or "").startswith("LeverError: Lever GET /opportunities/opp-bob/resumes "
                                        "failed with HTTP 500")
    erin = rows["opp-erin"].evaluation
    assert erin.status is Status.ERROR
    assert erin.error == "RuntimeError (details in the service logs)"
    assert rows["opp-alice"].evaluation.status is Status.SCORED
    dana = rows["opp-dana"].evaluation  # bob's text never arrived: no duplicate any more
    assert FlagCode.DUPLICATE_RESUME not in codes(dana)
    assert FlagCode.CONTACT_REUSE in codes(dana)  # the Lever record still counts
    assert secret not in caplog.text and "Gamma Retail" not in caplog.text
    assert "RuntimeError" in caplog.text
    audited = {row["opportunity_id"]: row["status"] for row in h.store.audit_rows()}
    assert audited["opp-bob"] == audited["opp-erin"] == "error"


async def test_vision_failure_keeps_the_file_flags(make_harness: Callable[..., Any],
                                                   kit: SimpleNamespace) -> None:
    h = make_harness(transcriber=lambda images: LLMBlockedError("image blocked"))
    frank = by_opp(await h.ranker.rank_posting(kit.POSTING_ID))["opp-frank"].evaluation
    assert frank.status is Status.ERROR
    assert codes(frank) == {FlagCode.SCANNED_RESUME, FlagCode.MODEL_BLOCKED}
    assert frank.error == "transcription failed: LLMBlockedError: image blocked"
    assert frank.file_hmac
    assert not any(kit.FRANK_MARKER in r.resume_text for r in h.evaluator.requests)


async def test_corrective_retry_failure_keeps_the_withheld_first_outcome(
        make_harness: Callable[..., Any], kit: SimpleNamespace,
        sample_criteria: RoleCriteria) -> None:
    scripted = kit.ScriptedResponder(protected_always=kit.ALICE_MARKER)

    def responder(request: AssessmentRequest) -> Any:
        if request.corrective_note:
            return LLMUnavailableError("timeout on retry")
        return scripted(request)

    h = make_harness(responder=responder, with_lever=False)
    evaluation = await h.ranker.evaluate_local(criteria=sample_criteria,
                                               text="\n".join(kit.ALICE_LINES),
                                               candidate_names=["Alice Novak"])
    assert len(h.evaluator.requests) == 2
    assert evaluation.status is Status.NEEDS_REVIEW and evaluation.score == 85
    assert not evaluation.rationale_publishable
    assert FlagCode.PROTECTED_TERM_IN_OUTPUT in codes(evaluation)
    assert evaluation.validation.corrective_retries == 0


async def test_protected_term_after_the_retry_withholds_the_rationale(
        make_harness: Callable[..., Any], kit: SimpleNamespace,
        sample_criteria: RoleCriteria) -> None:
    h = make_harness(responder=kit.ScriptedResponder(protected_always=kit.ALICE_MARKER),
                     with_lever=False)
    evaluation = await h.ranker.evaluate_local(criteria=sample_criteria,
                                               text="\n".join(kit.ALICE_LINES))
    first, second = h.evaluator.requests
    assert first.corrective_note is None and second.corrective_note
    assert first.evaluation_id == second.evaluation_id == evaluation.evaluation_id
    assert evaluation.validation.corrective_retries == 1
    assert not evaluation.rationale_publishable
    assert evaluation.status is Status.NEEDS_REVIEW


# ------------------------------------------------------------------------------- cache


async def test_force_bypasses_the_cache(make_harness: Callable[..., Any],
                                        kit: SimpleNamespace) -> None:
    h = make_harness()
    first = by_opp(await h.ranker.rank_posting(kit.POSTING_ID))
    calls, transcriptions = len(h.evaluator.requests), len(h.evaluator.transcribe_calls)
    second = by_opp(await h.ranker.rank_posting(kit.POSTING_ID, force=True))
    assert len(h.evaluator.requests) == 2 * calls
    assert len(h.evaluator.transcribe_calls) == transcriptions + 1
    for opp in kit.EXPECTED_SCORES:
        assert second[opp].evaluation.evaluation_id != first[opp].evaluation.evaluation_id
        assert second[opp].evaluation.score == first[opp].evaluation.score
    assert len(h.store.audit_rows()) == 2 * len(kit.EXPECTED_SCORES)


async def test_cache_is_not_shared_between_evaluation_and_real_runs(
        make_harness: Callable[..., Any], kit: SimpleNamespace) -> None:
    h = make_harness()
    await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.EVALUATION)
    calls = len(h.evaluator.requests)
    dry = by_opp(await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.DRY_RUN))
    assert len(h.evaluator.requests) == 2 * calls
    assert {dry[o].evaluation.mode for o in kit.EXPECTED_SCORES} == {RunMode.DRY_RUN}
    live = by_opp(await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE))
    assert len(h.evaluator.requests) == 2 * calls  # DRY_RUN results serve LIVE runs
    for opp in kit.EXPECTED_SCORES:  # ... recorded again as LIVE, see the next test
        derived, cached = live[opp].evaluation, dry[opp].evaluation
        assert derived.mode is RunMode.LIVE and derived.evaluation_id != cached.evaluation_id
        assert derived.model_copy(update={"evaluation_id": cached.evaluation_id,
                                          "mode": cached.mode}) == cached


async def test_live_reuse_of_dry_run_results_is_audited_as_live(
        make_harness: Callable[..., Any], kit: SimpleNamespace) -> None:
    h = make_harness(write_enabled=True)
    dry = by_opp(await h.ranker.rank_posting(kit.POSTING_ID))
    h.clock.now = kit.NOW + timedelta(days=2)
    live = by_opp(await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE))

    audit = {row["evaluation_id"]: row for row in h.store.audit_rows()}
    for opp in kit.EXPECTED_SCORES:
        evaluation = live[opp].evaluation
        assert evaluation.mode is RunMode.LIVE and evaluation.created_at == h.clock.now
        row = audit[evaluation.evaluation_id]
        assert (row["mode"], row["policy_allowed"], row["policy_regimes"]) == (
            "live", True, ["IL_HRA"])
        assert datetime.fromisoformat(row["created_at"]) == h.clock.now
        assert row["generation_config"] == {
            "derived_from": dry[opp].evaluation.evaluation_id, "reason": "reused for a live run"}
        writeback = live[opp].writeback
        assert writeback is not None and writeback.executed
        assert writeback.evaluation_id == evaluation.evaluation_id
        recorded = h.store.writebacks_for_opportunity(opp)
        assert [w.evaluation_id for w in recorded] == [dry[opp].evaluation.evaluation_id,
                                                      evaluation.evaluation_id]
        assert [w.executed for w in recorded] == [False, True]

    # The audit export ranks the live group on its own, completely.
    export = list(csv.DictReader(io.StringIO(audit_export_csv(h.store.audit_rows()))))
    live_rows = [r for r in export if r["mode"] == "live"]
    assert len(live_rows) == len(kit.EXPECTED_SCORES)
    assert sorted(int(r["rank_in_posting"]) for r in live_rows) == [1, 2, 3, 3, 5]  # bob=dana
    assert sorted(int(r["rank_in_posting"]) for r in export if r["mode"] == "dry_run") == [
        1, 2, 3, 3, 5]

    # Retention: the dry-run rows expire first; the live rows and their write-backs stay.
    removed = h.store.purge_expired(now=h.clock.now + timedelta(days=1), retention_days=2)
    assert removed == len(kit.EXPECTED_SCORES)
    for opp in kit.EXPECTED_SCORES:
        assert h.store.get_evaluation(dry[opp].evaluation.evaluation_id) is None
        assert h.store.get_evaluation(live[opp].evaluation.evaluation_id) is not None
        assert [w.evaluation_id for w in h.store.writebacks_for_opportunity(opp)] == [
            live[opp].evaluation.evaluation_id]


async def test_cache_hit_gains_new_integrity_flags(make_harness: Callable[..., Any],
                                                   kit: SimpleNamespace) -> None:
    h = make_harness()
    singles = {opp: (await h.ranker.evaluate_opportunity(opp)).evaluation
               for opp in ("opp-bob", "opp-dana")}
    for evaluation in singles.values():
        assert evaluation.status is Status.SCORED
        assert FlagCode.DUPLICATE_RESUME not in codes(evaluation)
    bob_calls = sum(kit.BOB_MARKER in r.resume_text for r in h.evaluator.requests)

    rows = by_opp(await h.ranker.rank_posting(kit.POSTING_ID))
    assert sum(kit.BOB_MARKER in r.resume_text for r in h.evaluator.requests) == bob_calls
    for opp, single in singles.items():
        derived = rows[opp].evaluation
        assert derived.evaluation_id != single.evaluation_id
        assert derived.status is Status.NEEDS_REVIEW and derived.score == single.score
        assert {FlagCode.DUPLICATE_RESUME, FlagCode.CONTACT_REUSE} <= codes(derived)
        assert derived.criteria_results == single.criteria_results
        audit = next(r for r in h.store.audit_rows()
                     if r["evaluation_id"] == derived.evaluation_id)
        assert audit["generation_config"]["derived_from"] == single.evaluation_id

    again = by_opp(await h.ranker.rank_posting(kit.POSTING_ID))
    assert again["opp-bob"].evaluation == rows["opp-bob"].evaluation  # now unchanged


# --------------------------------------------------------------- evaluate_opportunity


def without_processing_resume(api: Any) -> Any:
    """``populate_posting`` with Alice's newest, still-processing upload removed."""
    api.resumes["opp-alice"] = [r for r in api.resumes["opp-alice"]
                                if r["id"] != "res-alice-processing"]
    return api


async def test_evaluate_opportunity_scores_one_applicant(make_harness: Callable[..., Any],
                                                         kit: SimpleNamespace) -> None:
    h = make_harness(api=without_processing_resume(kit.populate_posting(kit.FakeLever())))
    row = await h.ranker.evaluate_opportunity("opp-alice")
    assert row.rank == 1 and row.evaluation.score == 85
    assert row.evaluation.status is Status.SCORED and row.evaluation.posting_id == kit.POSTING_ID
    assert row.writeback is not None and row.writeback.dry_run_reason == REASON_DRY_RUN
    fetch = next(r for r in h.api.requests if r.path == "/opportunities/opp-alice")
    assert fetch.params == {"expand": "applications"}
    assert h.store.latest_for_opportunity("opp-alice") == row.evaluation

    error_row = await make_harness(
        api=h.api, responder=kit.ScriptedResponder(errors={kit.ALICE_MARKER: LLMOutputError(
            "bad")})).ranker.evaluate_opportunity(
        "opp-alice", force=True)  # same state file and resume text: a cache hit otherwise
    assert error_row.rank is None and error_row.evaluation.status is Status.ERROR


async def test_evaluate_opportunity_defers_a_resume_lever_is_still_processing(
        make_harness: Callable[..., Any], kit: SimpleNamespace) -> None:
    api = kit.populate_posting(kit.FakeLever())
    api.add_opportunity("opp-ivy", name="Ivy None", created=1)
    api.add_opportunity("opp-jo", name="Jo Pending", created=2)
    api.add_resume("opp-jo", "res-jo", None, uploaded=5, status="processing")
    h = make_harness(api=api, write_enabled=True)

    # No resume yet, a resume still processing, and Alice's newer upload still processing
    # while an older processed one exists: all deferred, none scored on a stale file.
    for opp in ("opp-ivy", "opp-jo", "opp-alice"):
        row = await h.ranker.evaluate_opportunity(opp, mode=RunMode.LIVE)
        assert row.rank is None and row.evaluation.status is Status.SKIPPED
        assert row.evaluation.review_reasons == [SKIP_RESUME_PENDING]
        assert row.evaluation.candidate_name is not None  # storage allowed: name kept
        assert row.writeback is None
    assert h.api.writes == [] and h.evaluator.requests == []
    assert not any("/download" in p for p in h.api.paths())
    assert h.store.audit_rows() == [] and h.store.latest_for_opportunity("opp-jo") is None
    assert h.api.opportunities["opp-alice"]["tags"] == ["Referral", "AI Rank: Tier C"]

    # Lever finishes parsing: the next attempt scores and writes.
    api.resumes["opp-jo"][0]["file"]["status"] = "processed"
    api.files[("opp-jo", "res-jo")] = kit.text_pdf(kit.BOB_LINES)
    row = await h.ranker.evaluate_opportunity("opp-jo", mode=RunMode.LIVE)
    assert row.rank == 1 and row.evaluation.status is Status.SCORED
    assert row.writeback is not None and row.writeback.executed
    assert "AI Rank: Tier C" in api.opportunities["opp-jo"]["tags"]

    # The last retry gives up waiting: the usual ERROR / "Not scored" path applies.
    row = await h.ranker.evaluate_opportunity("opp-ivy", mode=RunMode.LIVE,
                                              defer_pending=False)
    assert row.evaluation.status is Status.ERROR and codes(row.evaluation) == {
        FlagCode.NO_RESUME}
    assert api.opportunities["opp-ivy"]["tags"] == ["AI Rank: Not scored"]
    # ... and the batch rule applies to Alice: her newest processed file is scored.
    row = await h.ranker.evaluate_opportunity("opp-alice", mode=RunMode.LIVE,
                                              defer_pending=False)
    assert row.evaluation.score == 85 and row.evaluation.resume_id == "res-alice-new"


async def test_evaluate_opportunity_skips(make_harness: Callable[..., Any],
                                          kit: SimpleNamespace) -> None:
    api = kit.populate_posting(kit.FakeLever())
    api.add_opportunity("opp-other", name="Olga Other", created=1,
                        posting_id=kit.OTHER_POSTING_ID)
    api.add_opportunity("opp-lead", name="Lee Lead", created=2)["applications"] = []
    h = make_harness(api=api)

    other = (await h.ranker.evaluate_opportunity("opp-other")).evaluation
    assert other.status is Status.SKIPPED and other.review_reasons == [SKIP_NO_CRITERIA]
    assert other.candidate_name == "Olga Other" and other.posting_id == kit.OTHER_POSTING_ID
    assert f"/postings/{kit.OTHER_POSTING_ID}" not in h.api.paths()

    lead = (await h.ranker.evaluate_opportunity("opp-lead")).evaluation
    assert lead.status is Status.SKIPPED and lead.review_reasons == [SKIP_NO_POSTING]

    hal = await h.ranker.evaluate_opportunity("opp-hal")
    assert hal.evaluation.review_reasons == [SKIP_STORAGE]
    assert hal.evaluation.candidate_name is None and hal.writeback is None

    h.store.record_deletion("opp-bob", source="webhook", deleted_at=kit.NOW)
    requests_before = len(h.api.requests)
    bob = (await h.ranker.evaluate_opportunity("opp-bob")).evaluation
    assert bob.status is Status.SKIPPED and bob.review_reasons == [SKIP_DELETED]
    assert len(h.api.requests) == requests_before  # no Lever call for a deleted candidate

    assert h.evaluator.requests == [] and h.store.audit_rows() == []


async def test_evaluate_opportunity_gates(make_harness: Callable[..., Any],
                                          kit: SimpleNamespace,
                                          sample_criteria: RoleCriteria) -> None:
    h = make_harness(criteria=unapproved(sample_criteria))
    with pytest.raises(RankingError, match="not approved"):
        await h.ranker.evaluate_opportunity("opp-alice")
    row = await h.ranker.evaluate_opportunity("opp-alice", mode=RunMode.EVALUATION,
                                              defer_pending=False)
    assert row.evaluation.mode is RunMode.EVALUATION and row.evaluation.score == 85

    blocked = make_harness(policy=kit.approved_policy(illinois=False))
    with pytest.raises(PolicyBlocked):
        await blocked.ranker.evaluate_opportunity("opp-alice", mode=RunMode.LIVE)
    assert blocked.evaluator.requests == []


# --------------------------------------------------------------------------- deletions


async def test_deletion_tombstone_skips_in_a_batch(make_harness: Callable[..., Any],
                                                   kit: SimpleNamespace) -> None:
    h = make_harness()
    h.store.record_deletion("opp-bob", source="webhook", deleted_at=kit.NOW)
    rows = by_opp(await h.ranker.rank_posting(kit.POSTING_ID))
    bob = rows["opp-bob"]
    assert bob.evaluation.status is Status.SKIPPED and bob.rank is None
    assert bob.evaluation.review_reasons == [SKIP_DELETED]
    assert bob.evaluation.candidate_name is None and bob.writeback is None
    assert not any("opp-bob/" in p for p in h.api.paths())
    dana = rows["opp-dana"].evaluation  # her twin is gone: nothing to compare with
    assert dana.status is Status.SCORED
    assert not codes(dana) & {FlagCode.DUPLICATE_RESUME, FlagCode.CONTACT_REUSE}
    assert "opp-bob" not in {row["opportunity_id"] for row in h.store.audit_rows()}


async def test_deletion_during_the_run_is_not_recorded(make_harness: Callable[..., Any],
                                                       kit: SimpleNamespace) -> None:
    holder: dict[str, Any] = {}

    def delete_erin(request: AssessmentRequest) -> None:
        if kit.ERIN_MARKER in request.resume_text:
            holder["store"].record_deletion("opp-erin", source="webhook", deleted_at=kit.NOW)

    h = make_harness(responder=kit.ScriptedResponder(on_call=delete_erin))
    holder["store"] = h.store
    erin = by_opp(await h.ranker.rank_posting(kit.POSTING_ID))["opp-erin"]
    assert erin.evaluation.status is Status.SKIPPED and erin.writeback is None
    assert erin.evaluation.review_reasons == [SKIP_DELETED]
    assert erin.evaluation.candidate_name is None
    assert h.store.latest_for_opportunity("opp-erin") is None
    assert "opp-erin" not in {row["opportunity_id"] for row in h.store.audit_rows()}


async def test_handle_candidate_deleted_purges_pii_and_keeps_audit(
        make_harness: Callable[..., Any], kit: SimpleNamespace) -> None:
    h = make_harness()
    rows = by_opp(await h.ranker.rank_posting(kit.POSTING_ID))
    alice_id = rows["opp-alice"].evaluation.evaluation_id
    h.api.failures[("GET", "/opportunities/opp-alice")] = 404  # Lever deleted her

    result = await h.ranker.handle_candidate_deleted("opp-alice")
    assert (result.opportunity_id, result.on_hold, result.pii_rows_deleted,
            result.audit_rows_kept) == ("opp-alice", False, 1, 1)
    assert h.store.is_deleted("opp-alice")
    assert h.store.get_evaluation(alice_id) is None
    assert alice_id in {row["evaluation_id"] for row in h.store.audit_rows()}
    assert h.api.paths().count("/opportunities/opp-alice") == 1  # the confirmation

    calls = len(h.evaluator.requests)
    again = by_opp(await h.ranker.rank_posting(kit.POSTING_ID))
    assert again["opp-alice"].evaluation.status is Status.SKIPPED
    assert len(h.evaluator.requests) == calls  # never re-scored


async def test_handle_candidate_deleted_respects_legal_holds(
        make_harness: Callable[..., Any], kit: SimpleNamespace) -> None:
    h = make_harness()
    rows = by_opp(await h.ranker.rank_posting(kit.POSTING_ID))
    h.store.set_legal_hold(scope="posting", scope_id=kit.POSTING_ID, reason="litigation",
                           set_by="legal", now=kit.NOW)
    del h.api.opportunities["opp-bob"]
    result = await h.ranker.handle_candidate_deleted("opp-bob", source="webhook")
    assert result.on_hold and result.pii_rows_deleted == 0
    assert h.store.get_evaluation(rows["opp-bob"].evaluation.evaluation_id) is not None
    assert h.store.is_deleted("opp-bob")  # tombstoned anyway: never scored again

    h.store.set_legal_hold(scope="posting", scope_id="posting-9", reason="audit",
                           set_by="legal", now=kit.NOW)
    unseen = await h.ranker.handle_candidate_deleted("opp-new", posting_id="posting-9")
    assert unseen.on_hold


async def test_webhook_deletion_is_confirmed_with_lever_first(
        make_harness: Callable[..., Any], kit: SimpleNamespace,
        caplog: pytest.LogCaptureFixture) -> None:
    h = make_harness()
    rows = by_opp(await h.ranker.rank_posting(kit.POSTING_ID))
    ids = {opp: rows[opp].evaluation.evaluation_id for opp in kit.EXPECTED_SCORES}
    gets_before = len(h.api.paths())

    # The opportunity named in the (unsigned) webhook data still exists: ignored.
    with caplog.at_level(logging.WARNING):
        result = await h.ranker.handle_candidate_deleted("opp-alice", source="webhook")
    assert (result.on_hold, result.pii_rows_deleted, result.audit_rows_kept) == (False, 0, 0)
    assert result.confirmed is False
    assert not h.store.is_deleted("opp-alice")
    assert h.store.get_evaluation(ids["opp-alice"]) is not None
    confirmations = [r for r in h.api.requests[gets_before:]]
    assert [(r.method, r.path, r.params) for r in confirmations] == [
        ("GET", "/opportunities/opp-alice", {"expand": "applications"})]
    assert "not confirmed by Lever" in caplog.text and "ignored" in caplog.text

    # Lever says deleted (404), anonymized, or carries a deletedAt: purged.
    del h.api.opportunities["opp-bob"]
    h.api.opportunities["opp-dana"]["isAnonymized"] = True
    h.api.opportunities["opp-erin"]["deletedAt"] = int(kit.NOW.timestamp() * 1000)
    for opp in ("opp-bob", "opp-dana", "opp-erin"):
        result = await h.ranker.handle_candidate_deleted(opp, source="webhook")
        assert result.confirmed and result.pii_rows_deleted == 1 and h.store.is_deleted(opp)
        assert h.store.get_evaluation(ids[opp]) is None

    # Reconciliation ids come from Lever already: no confirmation call.
    gets_before = len(h.api.paths())
    await h.ranker.handle_candidate_deleted("opp-frank", source="reconcile")
    assert h.store.is_deleted("opp-frank") and len(h.api.paths()) == gets_before

    # Lever unavailable: the job fails (and is retried by reconciliation), nothing changes.
    h.api.failures[("GET", "/opportunities/opp-alice")] = 503
    with pytest.raises(LeverError):
        await h.ranker.handle_candidate_deleted("opp-alice", source="webhook")
    assert not h.store.is_deleted("opp-alice")


async def test_forged_candidate_deleted_webhook_does_not_purge_a_live_candidate(
        make_harness: Callable[..., Any], kit: SimpleNamespace) -> None:
    """Through the real service: Lever's signature covers token + triggeredAt only, so a
    validly signed candidateDeleted envelope with a swapped ``data.opportunityId`` must
    not tombstone or purge the candidate it names while Lever still serves them."""
    signing = "whsec-candidate-deleted-test"
    h = make_harness(lever_webhook_tokens={"candidateDeleted": signing})
    rows = by_opp(await h.ranker.rank_posting(kit.POSTING_ID))
    alice_id = rows["opp-alice"].evaluation.evaluation_id
    components = Components(settings=h.settings, store=h.store, criteria=h.criteria,
                            policy=h.policy, lexicon=h.lexicon, extractor=h.extractor,
                            preparer=h.preparer, planner=h.planner, writer=h.writer,
                            ranker=h.ranker, lever=h.lever, evaluator=h.evaluator)
    app = create_app(components=components, clock=h.clock)

    def signed(token: str, data: dict[str, Any]) -> dict[str, Any]:
        triggered_at = int(kit.NOW.timestamp() * 1000)
        return {"event": "candidateDeleted", "triggeredAt": triggered_at, "token": token,
                "signature": compute_signature(token, triggered_at, signing), "data": data}

    with TestClient(app) as client:
        forged = client.post("/webhooks/lever",
                             json=signed("tok-1", {"opportunityId": "opp-alice"}))
        assert (forged.status_code, forged.json()) == (202, {"status": "queued"})
        client.portal.call(app.state.queue.join)
        assert not h.store.is_deleted("opp-alice")
        assert h.store.get_evaluation(alice_id) is not None
        assert "/opportunities/opp-alice" in h.api.paths()
        assert h.api.writes == []

        real = client.post("/webhooks/lever",
                           json=signed("tok-2", {"opportunityId": "opp-really-deleted"}))
        assert real.status_code == 202
        client.portal.call(app.state.queue.join)
        assert h.store.is_deleted("opp-really-deleted")  # Lever answered 404: confirmed
        assert app.state.queue.failed == 0


async def test_reconcile_deletions_reads_deleted_opportunities(
        make_harness: Callable[..., Any], kit: SimpleNamespace) -> None:
    h = make_harness()
    rows = by_opp(await h.ranker.rank_posting(kit.POSTING_ID))
    now_ms = int(kit.NOW.timestamp() * 1000)
    day = int(timedelta(days=1).total_seconds() * 1000)
    h.api.deleted_opportunities = [
        {"id": "opp-alice", "deletedAt": now_ms - 2 * day, "deletedBy": "user-1"},
        {"id": "opp-alice", "deletedAt": now_ms - 3 * day},
        {"id": "opp-zed", "deletedAt": now_ms - 45 * day},  # exactly on the inclusive start
        {"deletedAt": now_ms - day},  # no id: skipped
        {"id": "opp-old", "deletedAt": now_ms - 50 * day},
    ]
    # A deleted *application* of a live candidate must never tombstone the candidate.
    h.api.deleted = [{"id": "a1", "opportunityId": "opp-bob", "postingId": kit.POSTING_ID,
                      "deletedAt": now_ms - day}]
    gets_before = len(h.api.paths())

    results = await h.ranker.reconcile_deletions(days=45)
    assert [r.opportunity_id for r in results] == ["opp-alice", "opp-zed"]
    assert h.store.is_deleted("opp-alice") and h.store.is_deleted("opp-zed")
    assert not h.store.is_deleted("opp-old") and not h.store.is_deleted("opp-bob")
    assert h.store.get_evaluation(rows["opp-bob"].evaluation.evaluation_id) is not None
    assert h.store.get_evaluation(rows["opp-alice"].evaluation.evaluation_id) is None
    listed = h.api.requests[gets_before:]
    assert [(r.method, r.path) for r in listed] == [("GET", "/opportunities/deleted")]
    assert listed[0].params == {"deleted_at_start": str(now_ms - 45 * day),
                                "deleted_at_end": str(now_ms), "limit": "100"}
    with pytest.raises(ValueError, match="days"):
        await h.ranker.reconcile_deletions(days=0)


# ----------------------------------------------------------------------- evaluate_local


async def test_evaluate_local_on_a_generated_pdf(make_harness: Callable[..., Any],
                                                 kit: SimpleNamespace,
                                                 sample_criteria: RoleCriteria) -> None:
    h = make_harness(with_lever=False)
    evaluation = await h.ranker.evaluate_local(
        criteria=unapproved(sample_criteria), data=kit.text_pdf(kit.ALICE_LINES),
        file_name="alice.pdf", candidate_names=["Alice Novak"])
    assert evaluation.mode is RunMode.EVALUATION and evaluation.posting_id is None
    assert evaluation.opportunity_id == "local" and evaluation.status is Status.SCORED
    assert evaluation.score == 85 and evaluation.candidate_name == "Alice Novak"
    assert evaluation.file_hmac
    assert not any("Novak" in r.resume_text for r in h.evaluator.requests)
    assert h.store.audit_rows() == []  # "local" is never recorded

    calls = len(h.evaluator.requests)
    again = await h.ranker.evaluate_local(criteria=sample_criteria,
                                          data=kit.text_pdf(kit.ALICE_LINES),
                                          candidate_names=["Alice Novak"])
    assert len(h.evaluator.requests) == 2 * calls  # never cached
    assert again.evaluation_id != evaluation.evaluation_id

    recorded = await h.ranker.evaluate_local(criteria=sample_criteria,
                                             text="\n".join(kit.BOB_LINES),
                                             candidate_names=["Bob Marsh"],
                                             opportunity_id="bench-1")
    assert h.store.get_evaluation(recorded.evaluation_id) == recorded
    assert [row["mode"] for row in h.store.audit_rows()] == ["evaluation"]
    assert h.store.audit_rows()[0]["generation_config"]["fake"] is True

    scan = await h.ranker.evaluate_local(criteria=sample_criteria,
                                         data=kit.scanned_pdf(kit.FRANK_SCAN_LINES),
                                         candidate_names=["Frank Osei"])
    assert scan.score == 38 and FlagCode.SCANNED_RESUME in codes(scan)

    empty = await h.ranker.evaluate_local(criteria=sample_criteria, text="  ")
    assert empty.status is Status.ERROR and FlagCode.NO_RESUME in codes(empty)

    for kwargs in ({}, {"data": b"x", "text": "y"}):
        with pytest.raises(ValueError, match="exactly one"):
            await h.ranker.evaluate_local(criteria=sample_criteria, **kwargs)


# ----------------------------------------------------------------- systemic errors


def tags_by_opp(api: Any) -> dict[str, list[str]]:
    return {opp_id: list(record["tags"]) for opp_id, record in api.opportunities.items()}


async def test_llm_config_error_aborts_a_live_rerun_before_any_write(
        make_harness: Callable[..., Any], kit: SimpleNamespace) -> None:
    h = make_harness(write_enabled=True)
    await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE)
    tags, writes, audited = tags_by_opp(h.api), len(h.api.writes), len(h.store.audit_rows())
    assert tags["opp-alice"] == ["Referral", "AI Rank: Tier A"]

    h.evaluator._responder = lambda request: LLMConfigError("403 PERMISSION_DENIED", code=403)
    with pytest.raises(LLMConfigError, match="PERMISSION_DENIED"):
        await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE, force=True)
    assert tags_by_opp(h.api) == tags and len(h.api.writes) == writes
    assert len(h.store.audit_rows()) == audited  # no ERROR rows for an outage of ours
    assert all(len(notes) == 1 for notes in h.api.notes.values())

    # The placeholder evaluator (model not configured) raises without a code: systemic too.
    h.evaluator._responder = lambda request: LLMConfigError("AR_GCP_PROJECT is not set")
    with pytest.raises(LLMConfigError, match="AR_GCP_PROJECT"):
        await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE, force=True)
    assert tags_by_opp(h.api) == tags and len(h.api.writes) == writes


async def test_lever_auth_error_mid_run_aborts_without_writes_or_audit_rows(
        make_harness: Callable[..., Any], kit: SimpleNamespace) -> None:
    h = make_harness(write_enabled=True)
    h.api.failures[("GET", "/opportunities/opp-dana/resumes")] = 403
    with pytest.raises(LeverAuthError):
        await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE)
    assert h.api.writes == [] and h.store.audit_rows() == []
    assert tags_by_opp(h.api)["opp-alice"] == ["Referral", "AI Rank: Tier C"]

    del h.api.failures[("GET", "/opportunities/opp-dana/resumes")]
    h.api.failures[("GET", "/opportunities/opp-alice/resumes/res-alice-new/download")] = 401
    with pytest.raises(LeverAuthError):
        await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE)
    assert h.api.writes == [] and h.store.audit_rows() == []


async def test_llm_config_error_400_stays_with_the_applicant(
        make_harness: Callable[..., Any], kit: SimpleNamespace) -> None:
    h = make_harness(responder=kit.ScriptedResponder(
        errors={kit.BOB_MARKER: LLMConfigError("400 INVALID_ARGUMENT: bad image", code=400)}))
    rows = by_opp(await h.ranker.rank_posting(kit.POSTING_ID))
    for opp in ("opp-bob", "opp-dana"):  # the same resume text under two names
        evaluation = rows[opp].evaluation
        assert evaluation.status is Status.ERROR and not evaluation.transient
        assert evaluation.error == "LLMConfigError: 400 INVALID_ARGUMENT: bad image"
    assert {o: rows[o].evaluation.score for o in ("opp-alice", "opp-erin", "opp-frank")} == {
        "opp-alice": 85, "opp-erin": 76, "opp-frank": 38}


async def test_systemic_vision_error_propagates(make_harness: Callable[..., Any],
                                                kit: SimpleNamespace) -> None:
    h = make_harness(write_enabled=True,
                     transcriber=lambda images: LLMConfigError("403 forbidden", code=403))
    with pytest.raises(LLMConfigError):
        await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE)
    assert h.api.writes == []


# ----------------------------------------------------------------- transient errors


async def test_transient_failure_in_a_live_rerun_leaves_tags_and_note_alone(
        make_harness: Callable[..., Any], kit: SimpleNamespace) -> None:
    h = make_harness(write_enabled=True)
    h.api.failures[("GET", "/opportunities/opp-erin/resumes")] = 503
    first = by_opp(await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE))
    erin = first["opp-erin"]
    assert erin.evaluation.status is Status.ERROR and erin.evaluation.transient
    assert h.api.opportunities["opp-erin"]["tags"] == ["AI Rank: Not scored"]  # nothing to keep
    assert erin.writeback is not None and erin.writeback.executed
    assert h.api.opportunities["opp-alice"]["tags"] == ["Referral", "AI Rank: Tier A"]
    alice_note = dict(h.api.notes["opp-alice"])
    writes = len(h.api.writes)

    del h.api.failures[("GET", "/opportunities/opp-erin/resumes")]
    h.api.failures[("GET", "/opportunities/opp-alice/resumes/res-alice-new/download")] = 503
    second = by_opp(await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE))
    alice = second["opp-alice"]
    assert alice.evaluation.status is Status.ERROR and alice.evaluation.transient
    assert "HTTP 503" in (alice.evaluation.error or "")
    assert alice.writeback is not None and not alice.writeback.executed
    assert alice.writeback.dry_run_reason == REASON_TRANSIENT and alice.writeback.error is None
    assert writeback_label(alice.writeback) == "unchanged"
    assert h.api.opportunities["opp-alice"]["tags"] == ["Referral", "AI Rank: Tier A"]
    assert h.api.notes["opp-alice"] == alice_note
    assert not any("opp-alice" in w.path for w in h.api.writes[writes:])
    assert second["opp-erin"].evaluation.status is Status.SCORED  # Lever is back for Erin
    statuses = [(r["opportunity_id"], r["status"]) for r in h.store.audit_rows()]
    assert statuses.count(("opp-alice", "error")) == 1  # the outage is still audited

    # A file that is gone (404) says something about the applicant: not transient.
    del h.api.failures[("GET", "/opportunities/opp-alice/resumes/res-alice-new/download")]
    del h.api.files[("opp-alice", "res-alice-new")]
    third = by_opp(await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE))
    assert third["opp-alice"].evaluation.status is Status.ERROR
    assert not third["opp-alice"].evaluation.transient
    assert h.api.opportunities["opp-alice"]["tags"] == ["Referral", "AI Rank: Not scored"]


async def test_retryable_model_error_keeps_existing_tags(make_harness: Callable[..., Any],
                                                         kit: SimpleNamespace) -> None:
    responder = kit.ScriptedResponder()
    h = make_harness(responder=responder, write_enabled=True)
    await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE)
    bob_tags = list(h.api.opportunities["opp-bob"]["tags"])
    assert bob_tags == ["AI Rank: Needs review", "AI Rank: Tier C"]

    responder.errors[kit.BOB_MARKER] = LLMUnavailableError("HTTP 429 quota exhausted")
    rows = by_opp(await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE, force=True))
    bob = rows["opp-bob"]
    assert bob.evaluation.status is Status.ERROR and bob.evaluation.transient
    assert bob.writeback is not None and bob.writeback.dry_run_reason == REASON_TRANSIENT
    assert h.api.opportunities["opp-bob"]["tags"] == bob_tags

    responder.errors[kit.BOB_MARKER] = LLMBlockedError("blocked: SAFETY")  # final answer
    rows = by_opp(await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE, force=True))
    assert not rows["opp-bob"].evaluation.transient
    assert h.api.opportunities["opp-bob"]["tags"] == ["AI Rank: Not scored"]


# ------------------------------------------------------------- cache inputs


def test_lexicon_fingerprint_changes_with_content_not_only_version(
        pipeline_lexicon: ProtectedTermLexicon) -> None:
    same = ProtectedTermLexicon(pipeline_lexicon.categories, pipeline_lexicon.allow_phrases,
                                pipeline_lexicon.version)
    assert lexicon_fingerprint(same) == lexicon_fingerprint(pipeline_lexicon)
    assert lexicon_fingerprint(same).startswith(f"v{pipeline_lexicon.version}:")

    categories = pipeline_lexicon.categories
    categories["age"] = [*categories["age"], "zzz-extra-term"]
    extra_term = ProtectedTermLexicon(categories, pipeline_lexicon.allow_phrases,
                                      pipeline_lexicon.version)
    assert lexicon_fingerprint(extra_term) != lexicon_fingerprint(pipeline_lexicon)
    allow = ProtectedTermLexicon(pipeline_lexicon.categories,
                                 [*pipeline_lexicon.allow_phrases, "zzz phrase"],
                                 pipeline_lexicon.version)
    assert lexicon_fingerprint(allow) != lexicon_fingerprint(pipeline_lexicon)
    bumped = ProtectedTermLexicon(pipeline_lexicon.categories, pipeline_lexicon.allow_phrases,
                                  pipeline_lexicon.version + 1)
    assert lexicon_fingerprint(bumped) != lexicon_fingerprint(pipeline_lexicon)


async def test_tier_threshold_change_rederives_cached_rows_without_model_calls(
        make_harness: Callable[..., Any], kit: SimpleNamespace) -> None:
    first = make_harness()
    before = by_opp(await first.ranker.rank_posting(kit.POSTING_ID))
    assert before["opp-bob"].evaluation.tier is Tier.C

    tiers = TierThresholds(a=50, b=40, c=30)
    second = make_harness(api=first.api, store=first.store, tiers=tiers)
    rows = by_opp(await second.ranker.rank_posting(kit.POSTING_ID))
    assert second.evaluator.requests == [] and second.evaluator.transcribe_calls == []
    audit = {row["evaluation_id"]: row for row in second.store.audit_rows()}
    for opp, score in kit.EXPECTED_SCORES.items():
        evaluation = rows[opp].evaluation
        assert evaluation.score == score and evaluation.tier == tier_for(score, tiers)
        if opp == "opp-alice":  # 85 is Tier A on both scales: a plain hit
            assert evaluation == before[opp].evaluation
            continue
        assert evaluation.evaluation_id != before[opp].evaluation.evaluation_id
        row = audit[evaluation.evaluation_id]
        assert row["generation_config"]["derived_from"] == before[opp].evaluation.evaluation_id
        assert "tier thresholds changed" in row["generation_config"]["reason"]
    bob = rows["opp-bob"].evaluation
    assert bob.tier is Tier.A and bob.status is Status.NEEDS_REVIEW
    assert second.planner.desired_tags(bob) == ["AI Rank: Needs review", "AI Rank: Tier A"]
    assert rows["opp-frank"].evaluation.tier is Tier.C  # 38: D before, C under c=30

    again = by_opp(await second.ranker.rank_posting(kit.POSTING_ID))
    assert second.evaluator.requests == []
    for opp in kit.EXPECTED_SCORES:  # now a plain hit on the derived rows
        assert again[opp].evaluation == rows[opp].evaluation


async def test_derived_rows_store_the_prepared_text_only_when_there_is_one(
        make_harness: Callable[..., Any], kit: SimpleNamespace, tmp_path: Any) -> None:
    from applicant_ranking.store import SqliteStore

    store = SqliteStore(tmp_path / "with-text.sqlite3", store_resume_text=True)
    first = make_harness(store=store)
    before = by_opp(await first.ranker.rank_posting(kit.POSTING_ID))
    frank_text = store.get_redacted_text(before["opp-frank"].evaluation.evaluation_id)
    assert frank_text and "Delta Inc" in frank_text  # the transcript, stored once

    second = make_harness(api=first.api, store=store, tiers=TierThresholds(a=50, b=40, c=30))
    rows = by_opp(await second.ranker.rank_posting(kit.POSTING_ID))
    assert second.evaluator.transcribe_calls == []  # vision hit: no page images prepared
    frank = rows["opp-frank"].evaluation
    assert frank.evaluation_id != before["opp-frank"].evaluation.evaluation_id
    assert store.get_redacted_text(frank.evaluation_id) is None  # not an empty text
    bob = rows["opp-bob"].evaluation
    assert store.get_redacted_text(bob.evaluation_id) == store.get_redacted_text(
        before["opp-bob"].evaluation.evaluation_id)
    store.close()


async def test_education_year_redaction_setting_is_part_of_the_cache_key(
        make_harness: Callable[..., Any], kit: SimpleNamespace) -> None:
    api = kit.populate_posting(kit.FakeLever())
    api.files[("opp-alice", "res-alice-new")] = kit.text_pdf(
        [*kit.ALICE_LINES, "EDUCATION", "B.S. Computer Science, State University, 1998"])
    first = make_harness(api=api, redact_education_years=False)
    await first.ranker.rank_posting(kit.POSTING_ID)
    alice_requests = [r for r in first.evaluator.requests if kit.ALICE_MARKER in r.resume_text]
    assert alice_requests and all("1998" in r.resume_text for r in alice_requests)
    assert len(first.evaluator.transcribe_calls) == 1

    second = make_harness(api=api, store=first.store, redact_education_years=True)
    rows = by_opp(await second.ranker.rank_posting(kit.POSTING_ID))
    alice_requests = [r for r in second.evaluator.requests if kit.ALICE_MARKER in r.resume_text]
    assert alice_requests and not any("1998" in r.resume_text for r in alice_requests)
    assert rows["opp-alice"].evaluation.redaction_version == second.preparer.redaction_version
    assert second.preparer.redaction_version.endswith("+eduyears=1+max=30000")
    assert first.preparer.redaction_version.endswith("+eduyears=0+max=30000")
    assert len(second.evaluator.transcribe_calls) == 1  # the scan is prepared again too
    assert len(second.evaluator.requests) == len(first.evaluator.requests)  # all re-evaluated


async def test_lexicon_content_change_is_a_cache_miss(make_harness: Callable[..., Any],
                                                      kit: SimpleNamespace,
                                                      pipeline_lexicon: ProtectedTermLexicon,
                                                      ) -> None:
    first = make_harness()
    await first.ranker.rank_posting(kit.POSTING_ID)
    categories = pipeline_lexicon.categories
    categories["age"] = [*categories["age"], "zzz-extra-term"]
    changed = ProtectedTermLexicon(categories, pipeline_lexicon.allow_phrases,
                                   pipeline_lexicon.version)  # same version, new content
    second = make_harness(api=first.api, store=first.store, lexicon=changed)
    rows = by_opp(await second.ranker.rank_posting(kit.POSTING_ID))
    assert len(second.evaluator.requests) == len(first.evaluator.requests)
    assert {o: rows[o].evaluation.score for o in kit.EXPECTED_SCORES} == kit.EXPECTED_SCORES

    unchanged = make_harness(api=first.api, store=first.store)
    await unchanged.ranker.rank_posting(kit.POSTING_ID)
    assert unchanged.evaluator.requests == []


# ------------------------------------------------------------- policy calendar


def test_policy_today_is_the_earliest_local_date_of_the_regimes() -> None:
    instant = datetime(2026, 10, 23, 1, 30, tzinfo=UTC)  # 21:30 the day before in New York
    assert policy_today(instant, [IL_HRA, NYC_LL144]) == date(2026, 10, 22)
    assert policy_today(instant, [NYC_LL144]) == date(2026, 10, 22)
    assert policy_today(instant, []) == date(2026, 10, 23)  # no regime: UTC
    assert policy_today(instant, ["UNKNOWN"]) == date(2026, 10, 23)
    assert policy_today(instant.replace(tzinfo=None), [NYC_LL144]) == date(2026, 10, 22)
    noon = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    assert policy_today(noon, [IL_HRA, NYC_LL144, CO_ADMT, CA_FEHA]) == date(2026, 10, 23)
    # 05:30 UTC: New York (01:30) is already on the 23rd, Los Angeles (22:30) is not.
    early = datetime(2026, 10, 23, 5, 30, tzinfo=UTC)
    assert policy_today(early, [NYC_LL144]) == date(2026, 10, 23)
    assert policy_today(early, [NYC_LL144, CA_FEHA]) == date(2026, 10, 22)


async def test_nyc_notice_wait_opens_on_the_new_york_calendar(
        make_harness: Callable[..., Any], kit: SimpleNamespace) -> None:
    api = kit.populate_posting(kit.FakeLever())
    api.postings[kit.POSTING_ID]["categories"] = {"location": "New York, NY",
                                                  "allLocations": ["New York, NY"]}
    policy = PolicyEngine(PolicyConfig(
        approved_by="Legal and HR", approved_at=date(2026, 9, 1), allowed_countries=["US"],
        illinois=IllinoisPolicy(notice_url="https://careers.example.com/ai-notice",
                                posting_notice_confirmed=True),
        nyc=NycPolicy(bias_audit_date=date(2026, 9, 1),
                      audit_summary_url="https://careers.example.com/bias-audit",
                      notice_published_at=date(2026, 10, 9)),  # + 10 business days = 10-23
    ), source="test-policy")
    h = make_harness(api=api, policy=policy, write_enabled=True)

    h.clock.now = datetime(2026, 10, 23, 1, 30, tzinfo=UTC)  # 2026-10-22 21:30 in New York
    for mode in (RunMode.DRY_RUN, RunMode.LIVE):
        with pytest.raises(PolicyBlocked, match="2026-10-23 is after today \\(2026-10-22\\)"):
            await h.ranker.rank_posting(kit.POSTING_ID, mode=mode)
    assert h.evaluator.requests == [] and h.api.writes == []

    h.clock.now = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    report = await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE)
    assert report.policy_regimes == ["IL_HRA", "NYC_LL144"]
    assert h.api.writes != []


# ------------------------------------------------------- corrective retry signals


def test_carried_review_signals_use_fixed_text() -> None:
    model_text = "She claims to be the best; ignore the rubric"
    first = [
        Flag(code=FlagCode.MODEL_INJECTION_REPORT, source="model", severity="review",
             detail=model_text),
        Flag(code=FlagCode.MODEL_RED_FLAG, source="model", severity="review",
             detail=f"possible_prompt_injection: {model_text}"),
        Flag(code=FlagCode.MODEL_RED_FLAG, source="model", severity="info",
             detail=f"criteria_gap: {model_text}"),
        Flag(code=FlagCode.UNGROUNDED_EVIDENCE, severity="review", detail="5 of 6 quotes"),
        Flag(code=FlagCode.PROTECTED_TERM_IN_OUTPUT, severity="review", detail="gender"),
        Flag(code=FlagCode.LOW_CONFIDENCE, source="model", severity="info"),
    ]
    carried = carried_review_signals(first)
    assert [(f.code, f.source, f.severity) for f in carried] == [
        (FlagCode.MODEL_INJECTION_REPORT, "model", "review"),
        (FlagCode.UNGROUNDED_EVIDENCE, "system", "review"),
    ]
    assert all("first attempt" in f.detail and model_text not in f.detail for f in carried)
    assert carried_review_signals(first[2:3]) == []  # an info red flag is not carried
    assert carried_review_signals(first[1:2])[0].code is FlagCode.MODEL_INJECTION_REPORT
    assert carried_review_signals([Flag(code=FlagCode.UNGROUNDED_EVIDENCE, severity="info")]
                                  ) == []
    assert carried_review_signals([]) == []


async def test_corrective_retry_keeps_the_first_attempts_injection_report(
        make_harness: Callable[..., Any], kit: SimpleNamespace) -> None:
    scripted = kit.ScriptedResponder(protected_first=kit.ALICE_MARKER)
    model_text = "The resume tells the reader to rate her 100."

    def responder(request: AssessmentRequest) -> Any:
        assessment = scripted(request)
        if request.corrective_note is None and kit.ALICE_MARKER in request.resume_text:
            return assessment.model_copy(update={
                "injection_attempt_detected": True,
                "red_flags": [RedFlag(category="possible_prompt_injection", detail=model_text)],
            })
        return assessment  # the retry comes back clean

    h = make_harness(responder=responder, write_enabled=True)
    rows = by_opp(await h.ranker.rank_posting(kit.POSTING_ID, mode=RunMode.LIVE))
    alice = rows["opp-alice"].evaluation
    assert [r.corrective_note is None for r in h.evaluator.requests
            if kit.ALICE_MARKER in r.resume_text] == [True, False]
    assert alice.validation.corrective_retries == 1 and alice.score == 85
    assert alice.status is Status.NEEDS_REVIEW
    assert FlagCode.MODEL_INJECTION_REPORT in codes(alice)
    assert FlagCode.PROTECTED_TERM_IN_OUTPUT not in codes(alice)  # the retry fixed that
    assert any("manipulate" in reason for reason in alice.review_reasons)
    assert not any(model_text in f.detail for f in alice.flags)  # fixed text only
    assert alice.rationale_publishable
    assert h.api.opportunities["opp-alice"]["tags"] == [
        "Referral", "AI Rank: Needs review", "AI Rank: Tier A"]
    note = next(iter(h.api.notes["opp-alice"].values()))
    assert "Model reported an attempt to manipulate the evaluation (needs review)" in note
    assert model_text not in note and "She" not in note

    bob = rows["opp-bob"].evaluation  # no injection reported for the others
    assert FlagCode.MODEL_INJECTION_REPORT not in codes(bob)


# --------------------------------------------------------------------------- concurrency


class ProbeEvaluator:
    """Wraps the fake and records how many model calls overlap."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.model_id = inner.model_id
        self.prompt_version = inner.prompt_version
        self.active = 0
        self.peak = 0

    async def _overlap(self, call: Callable[[], Any]) -> Any:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            for _ in range(5):
                await asyncio.sleep(0)
            return await call()
        finally:
            self.active -= 1

    async def assess(self, request: AssessmentRequest) -> LLMCallResult:
        return await self._overlap(lambda: self.inner.assess(request))

    async def transcribe(self, page_images: list[bytes]) -> str:
        return await self._overlap(lambda: self.inner.transcribe(page_images))

    async def aclose(self) -> None:
        await self.inner.aclose()


@pytest.mark.parametrize("limit", [1, 2])
async def test_model_calls_are_bounded(make_harness: Callable[..., Any],
                                       kit: SimpleNamespace, limit: int) -> None:
    from applicant_ranking.llm.fake import FakeEvaluator

    probe = ProbeEvaluator(FakeEvaluator(kit.ScriptedResponder(),
                                         transcriber=lambda images: kit.FRANK_TRANSCRIPT))
    h = make_harness(evaluator=probe, llm_concurrency=limit)
    report = await h.ranker.rank_posting(kit.POSTING_ID)
    assert probe.peak == limit
    assert len([r for r in report.rows if r.rank is not None]) == 5


@pytest.mark.parametrize("limit", [1, 3])
async def test_downloads_and_preparation_are_bounded(make_harness: Callable[..., Any],
                                                     kit: SimpleNamespace,
                                                     monkeypatch: pytest.MonkeyPatch,
                                                     limit: int) -> None:
    h = make_harness(io_concurrency=limit)
    real = h.preparer.prepare_file
    state = {"active": 0, "peak": 0}

    async def prepare_file(*args: Any, **kwargs: Any) -> Any:
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        try:
            for _ in range(5):
                await asyncio.sleep(0)
            return await real(*args, **kwargs)
        finally:
            state["active"] -= 1

    monkeypatch.setattr(h.preparer, "prepare_file", prepare_file)
    await h.ranker.rank_posting(kit.POSTING_ID)
    assert state["peak"] == limit


async def test_io_concurrency_must_be_positive(make_harness: Callable[..., Any]) -> None:
    with pytest.raises(ValueError, match="io_concurrency"):
        make_harness(io_concurrency=0)

