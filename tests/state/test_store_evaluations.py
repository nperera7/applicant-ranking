"""Recording and reading evaluations: append-only audit, PII store, cache, ordering."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from applicant_ranking.domain import Evaluation, RunMode, Status
from applicant_ranking.store import DuplicateEvaluationError, SqliteStore

MakeEvaluation = Callable[..., Evaluation]


def count(store: SqliteStore, table: str) -> int:
    return store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# ---------------------------------------------------------------------------- round trip


def test_audit_row_keeps_the_count_of_unknown_ids_but_not_their_text(
        store: SqliteStore, make_evaluation: MakeEvaluation) -> None:
    original = make_evaluation()
    validation = original.validation.model_copy(update={
        "unknown_ids_dropped": ["made-up", "another-made-up"], "unknown_ids_dropped_count": 7})
    evaluation = original.model_copy(update={"validation": validation})
    store.record_evaluation(evaluation, cache_key="key-1")

    [row] = store.audit_rows()
    assert "unknown_ids_dropped" not in row["validation"]
    assert row["validation"]["unknown_ids_dropped_count"] == 7
    raw = store._conn.execute("SELECT validation_json FROM audit_log").fetchone()[0]
    assert "made-up" not in raw
    loaded = store.get_evaluation(evaluation.evaluation_id)
    assert loaded is not None
    assert loaded.validation.unknown_ids_dropped == ["made-up", "another-made-up"]
    assert loaded.validation.unknown_ids_dropped_count == 7


def test_round_trip_keeps_every_nested_field(store: SqliteStore,
                                             make_evaluation: MakeEvaluation) -> None:
    original = make_evaluation()
    store.record_evaluation(original, cache_key="key-1", generation_config={"seed": 7})
    loaded = store.get_evaluation(original.evaluation_id)
    assert loaded is not None
    assert loaded == original
    assert loaded.model_dump(mode="json") == original.model_dump(mode="json")
    assert loaded.created_at == original.created_at
    assert loaded.criteria_results[0].evidence == original.criteria_results[0].evidence
    assert loaded.knockout_results[0].result == "met"
    assert loaded.flags[0].detail == original.flags[0].detail
    assert loaded.validation.downgraded_requirements == ["gcp-pipelines"]
    assert loaded.usage is not None and loaded.usage.estimated_cost_usd == 0.0123
    assert loaded.rationale_publishable is False


def test_round_trip_minimal_evaluation(store: SqliteStore) -> None:
    minimal = Evaluation(evaluation_id="e-min", mode=RunMode.EVALUATION, opportunity_id="local",
                         status=Status.ERROR, error="no resume")
    store.record_evaluation(minimal, cache_key=None)
    assert store.get_evaluation("e-min") == minimal
    row = store.audit_rows()[0]
    assert row["usage"] is None and row["generation_config"] is None
    assert row["posting_id"] is None and row["score"] is None and row["tier"] is None


def test_unknown_evaluation_is_none(store: SqliteStore) -> None:
    assert store.get_evaluation("nope") is None
    assert store.latest_for_opportunity("nope") is None
    assert store.evaluations_for_posting("nope") == []


# ---------------------------------------------------------------------------- append-only


def test_rerecording_the_same_id_raises(store: SqliteStore,
                                        make_evaluation: MakeEvaluation) -> None:
    first = make_evaluation(evaluation_id="e-1", score=70)
    store.record_evaluation(first, cache_key="k")
    with pytest.raises(DuplicateEvaluationError, match="e-1"):
        store.record_evaluation(make_evaluation(evaluation_id="e-1", score=99), cache_key="k")
    assert count(store, "audit_log") == 1
    assert count(store, "pii_evaluations") == 1
    loaded = store.get_evaluation("e-1")
    assert loaded is not None and loaded.score == 70


def test_rerecording_after_candidate_purge_still_raises(store: SqliteStore,
                                                        make_evaluation: MakeEvaluation
                                                        ) -> None:
    store.record_evaluation(make_evaluation(evaluation_id="e-1"), cache_key="k")
    store.purge_candidate("opp-1")
    with pytest.raises(DuplicateEvaluationError):
        store.record_evaluation(make_evaluation(evaluation_id="e-1"), cache_key="k")
    assert count(store, "pii_evaluations") == 0


def test_direct_update_of_audit_log_is_aborted(store: SqliteStore,
                                               make_evaluation: MakeEvaluation) -> None:
    store.record_evaluation(make_evaluation(evaluation_id="e-1", score=70), cache_key="k")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store._conn.execute("UPDATE audit_log SET score = 100 WHERE evaluation_id = 'e-1'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store._conn.execute(
            "INSERT INTO audit_log (evaluation_id, created_at, mode, opportunity_id, status, "
            "flag_codes_json, review_reasons_json, validation_json, rationale_publishable) "
            "VALUES ('e-1', 'x', 'live', 'o', 'scored', '[]', '[]', '{}', 1) "
            "ON CONFLICT (evaluation_id) DO UPDATE SET score = 100")
    assert store.audit_rows()[0]["score"] == 70


def test_update_is_aborted_through_another_connection(db_path: Path,
                                                      make_evaluation: MakeEvaluation
                                                      ) -> None:
    with SqliteStore(db_path) as store:
        store.record_evaluation(make_evaluation(evaluation_id="e-1"), cache_key="k")
    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE audit_log SET status = 'scored'")
    finally:
        conn.close()


def test_pii_row_requires_an_audit_row(store: SqliteStore) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute("INSERT INTO pii_evaluations (evaluation_id, opportunity_id, "
                            "created_at, evaluation_json) VALUES ('orphan', 'o', 'x', '{}')")


# ----------------------------------------------------------------------------- PII text


def test_redacted_text_not_stored_by_default(store: SqliteStore,
                                             make_evaluation: MakeEvaluation) -> None:
    store.record_evaluation(make_evaluation(evaluation_id="e-1"), cache_key="k",
                            redacted_text="[CANDIDATE] resume text")
    assert store.get_redacted_text("e-1") is None


def test_redacted_text_stored_when_enabled(tmp_path: Path,
                                           make_evaluation: MakeEvaluation) -> None:
    with SqliteStore(tmp_path / "s.sqlite3", store_resume_text=True) as store:
        store.record_evaluation(make_evaluation(evaluation_id="e-1"), cache_key="k",
                                redacted_text="[CANDIDATE] resume text")
        assert store.get_redacted_text("e-1") == "[CANDIDATE] resume text"
        store.purge_candidate("opp-1")
        assert store.get_redacted_text("e-1") is None


# -------------------------------------------------------------------------------- cache


def test_cache_hit_returns_the_most_recent_non_error(store: SqliteStore,
                                                     make_evaluation: MakeEvaluation,
                                                     base_time: datetime) -> None:
    older = make_evaluation(evaluation_id="old", created_at=base_time, score=60)
    newer = make_evaluation(evaluation_id="new", created_at=base_time + timedelta(hours=1),
                            score=65)
    store.record_evaluation(newer, cache_key="k")  # recorded first, but newer
    store.record_evaluation(older, cache_key="k")
    hit = store.find_cached("k")
    assert hit is not None and hit.evaluation_id == "new"


def test_cache_miss(store: SqliteStore, make_evaluation: MakeEvaluation) -> None:
    store.record_evaluation(make_evaluation(), cache_key="k")
    assert store.find_cached("other") is None
    assert store.find_cached("") is None


def test_errors_are_not_cached(store: SqliteStore, make_evaluation: MakeEvaluation,
                               base_time: datetime) -> None:
    store.record_evaluation(make_evaluation(evaluation_id="err", status=Status.ERROR,
                                            score=None, tier=None, error="LLM timeout"),
                            cache_key="k")
    assert store.find_cached("k") is None
    store.record_evaluation(make_evaluation(evaluation_id="ok", created_at=base_time),
                            cache_key="k")
    store.record_evaluation(make_evaluation(evaluation_id="err2", status=Status.ERROR,
                                            created_at=base_time + timedelta(days=1)),
                            cache_key="k")
    hit = store.find_cached("k")
    assert hit is not None and hit.evaluation_id == "ok"


@pytest.mark.parametrize("status", [Status.SCORED, Status.KNOCKOUT_FLAGGED,
                                    Status.NEEDS_REVIEW, Status.SKIPPED])
def test_non_error_statuses_are_cached(store: SqliteStore, make_evaluation: MakeEvaluation,
                                       status: Status) -> None:
    store.record_evaluation(make_evaluation(status=status), cache_key="k")
    assert store.find_cached("k") is not None


def test_purged_evaluations_are_not_cached(store: SqliteStore,
                                           make_evaluation: MakeEvaluation) -> None:
    store.record_evaluation(make_evaluation(opportunity_id="opp-1"), cache_key="k")
    store.purge_candidate("opp-1")
    assert store.find_cached("k") is None


def test_evaluation_without_cache_key_is_never_a_hit(store: SqliteStore,
                                                     make_evaluation: MakeEvaluation) -> None:
    store.record_evaluation(make_evaluation(), cache_key=None)
    assert store.find_cached("None") is None


# ------------------------------------------------------------------------------ queries


def test_latest_for_opportunity_orders_by_created_at(store: SqliteStore,
                                                     make_evaluation: MakeEvaluation,
                                                     base_time: datetime) -> None:
    store.record_evaluation(make_evaluation(evaluation_id="mid",
                                            created_at=base_time + timedelta(hours=1)),
                            cache_key=None)
    store.record_evaluation(make_evaluation(evaluation_id="late",
                                            created_at=base_time + timedelta(hours=2),
                                            status=Status.ERROR), cache_key=None)
    store.record_evaluation(make_evaluation(evaluation_id="early", created_at=base_time),
                            cache_key=None)
    store.record_evaluation(make_evaluation(evaluation_id="other-opp", opportunity_id="opp-2",
                                            created_at=base_time + timedelta(days=1)),
                            cache_key=None)
    latest = store.latest_for_opportunity("opp-1")
    assert latest is not None and latest.evaluation_id == "late"  # errors included


def test_latest_for_opportunity_tie_goes_to_last_recorded(store: SqliteStore,
                                                          make_evaluation: MakeEvaluation,
                                                          base_time: datetime) -> None:
    for evaluation_id in ("first", "second"):
        store.record_evaluation(make_evaluation(evaluation_id=evaluation_id,
                                                created_at=base_time), cache_key=None)
    latest = store.latest_for_opportunity("opp-1")
    assert latest is not None and latest.evaluation_id == "second"


def test_timezones_are_normalised_for_ordering(store: SqliteStore,
                                               make_evaluation: MakeEvaluation,
                                               base_time: datetime) -> None:
    plus_five = timezone(timedelta(hours=5))
    # 16:00+05:00 is 11:00 UTC: earlier than base_time (12:00 UTC) despite the larger clock.
    store.record_evaluation(make_evaluation(evaluation_id="utc", created_at=base_time),
                            cache_key=None)
    store.record_evaluation(make_evaluation(
        evaluation_id="offset", created_at=datetime(2026, 10, 8, 16, 0, tzinfo=plus_five)),
        cache_key=None)
    latest = store.latest_for_opportunity("opp-1")
    assert latest is not None and latest.evaluation_id == "utc"
    offset = store.get_evaluation("offset")
    assert offset is not None and offset.created_at.utcoffset() == timedelta(hours=5)
    assert store.audit_rows()[0]["created_at"] == "2026-10-08T11:00:00.000000+00:00"


def test_evaluations_for_posting(store: SqliteStore, make_evaluation: MakeEvaluation,
                                 base_time: datetime) -> None:
    def add(evaluation_id: str, opp: str, hours: int, posting: str = "posting-1") -> None:
        store.record_evaluation(make_evaluation(evaluation_id=evaluation_id,
                                                opportunity_id=opp, posting_id=posting,
                                                created_at=base_time + timedelta(hours=hours)),
                                cache_key=None)

    add("a1", "opp-a", 1)
    add("b1", "opp-b", 2)
    add("a2", "opp-a", 3)
    add("c1", "opp-c", 0, posting="posting-2")
    latest = [e.evaluation_id for e in store.evaluations_for_posting("posting-1")]
    assert latest == ["b1", "a2"]
    every = [e.evaluation_id for e in store.evaluations_for_posting("posting-1",
                                                                     latest_only=False)]
    assert every == ["a1", "b1", "a2"]
    assert [e.evaluation_id for e in store.evaluations_for_posting("posting-2")] == ["c1"]


def test_evaluations_for_posting_skips_purged_candidates(store: SqliteStore,
                                                         make_evaluation: MakeEvaluation
                                                         ) -> None:
    store.record_evaluation(make_evaluation(opportunity_id="opp-a"), cache_key=None)
    store.record_evaluation(make_evaluation(opportunity_id="opp-b"), cache_key=None)
    store.purge_candidate("opp-a")
    assert [e.opportunity_id for e in store.evaluations_for_posting("posting-1")] == ["opp-b"]


# ---------------------------------------------------------------------------- audit rows


def test_audit_rows_are_pseudonymous(store: SqliteStore, make_evaluation: MakeEvaluation,
                                     sensitive_texts: tuple[str, ...]) -> None:
    store.record_evaluation(make_evaluation(), cache_key="k",
                            generation_config={"thinking_level": "low", "seed": 7},
                            redacted_text="[CANDIDATE] resume body")
    rows = store.audit_rows()
    dumped = json.dumps(rows)
    for text in (*sensitive_texts, "[CANDIDATE] resume body"):
        assert text not in dumped
    assert "candidate_name" not in rows[0]
    raw = store._conn.execute("SELECT * FROM audit_log").fetchone()
    raw_text = json.dumps(list(raw))
    for text in sensitive_texts:
        assert text not in raw_text


def test_audit_row_contents(store: SqliteStore, make_evaluation: MakeEvaluation) -> None:
    evaluation = make_evaluation(evaluation_id="e-1")
    store.record_evaluation(evaluation, cache_key="k", generation_config={"seed": 7})
    row = store.audit_rows()[0]
    assert row["evaluation_id"] == "e-1"
    assert row["mode"] == "dry_run"
    assert row["status"] == "needs_review"
    assert row["score"] == 74 and row["tier"] == "B" and row["holistic_score"] == 80
    assert row["flag_codes"] == ["hidden_text", "model_red_flag"]
    assert row["review_reasons"] == ["hidden_text"]
    assert row["validation"]["quotes_grounded"] == 2
    assert row["usage"]["total_tokens"] == 1550
    assert row["generation_config"] == {"seed": 7}
    assert row["rationale_publishable"] is False
    assert row["file_hmac"] == "f" * 64
    assert row["criteria_hash"] == "c" * 64
    assert row["created_at"] == "2026-10-08T12:01:00.000000+00:00"


def test_audit_rows_filter_and_order(store: SqliteStore, make_evaluation: MakeEvaluation,
                                     base_time: datetime) -> None:
    store.record_evaluation(make_evaluation(evaluation_id="b", posting_id="p1",
                                            created_at=base_time + timedelta(hours=1)),
                            cache_key=None)
    store.record_evaluation(make_evaluation(evaluation_id="a", posting_id="p1",
                                            created_at=base_time), cache_key=None)
    store.record_evaluation(make_evaluation(evaluation_id="c", posting_id="p2"),
                            cache_key=None)
    assert [r["evaluation_id"] for r in store.audit_rows(posting_id="p1")] == ["a", "b"]
    assert {r["evaluation_id"] for r in store.audit_rows()} == {"a", "b", "c"}
    assert store.audit_rows(posting_id="none") == []


def test_audit_rows_survive_candidate_purge(store: SqliteStore,
                                            make_evaluation: MakeEvaluation) -> None:
    store.record_evaluation(make_evaluation(evaluation_id="e-1"), cache_key=None)
    store.purge_candidate("opp-1")
    assert store.get_evaluation("e-1") is None
    assert [r["evaluation_id"] for r in store.audit_rows()] == ["e-1"]
