"""Legal holds, candidate purge (deletion handling), deletions a hold suspended, retention
purge, and what a purge leaves in the database files."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from applicant_ranking.domain import Evaluation, WritebackResult
from applicant_ranking.store import PurgeResult, SqliteStore

MakeEvaluation = Callable[..., Evaluation]


def count(store: SqliteStore, table: str) -> int:
    return store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def seed_candidate(store: SqliteStore, make_evaluation: MakeEvaluation, opp: str,
                   posting: str | None = "posting-1", n: int = 2,
                   created_at: datetime | None = None) -> None:
    for _ in range(n):
        overrides = {"opportunity_id": opp, "posting_id": posting}
        if created_at is not None:
            overrides["created_at"] = created_at
        evaluation = make_evaluation(**overrides)
        store.record_evaluation(evaluation, cache_key=f"{opp}-key")
        store.record_writeback(WritebackResult(opportunity_id=opp,
                                               evaluation_id=evaluation.evaluation_id,
                                               executed=False),
                               now=evaluation.created_at)
    store.set_note_id(opp, f"note-{opp}", now=datetime(2026, 10, 8))


# --------------------------------------------------------------------------------- holds


def test_hold_set_check_clear(store: SqliteStore, base_time: datetime) -> None:
    assert not store.is_on_hold(opportunity_id="opp-1")
    store.set_legal_hold(scope="opportunity", scope_id="opp-1", reason="EEOC charge",
                         set_by="Legal", now=base_time)
    assert store.is_on_hold(opportunity_id="opp-1")
    assert not store.is_on_hold(opportunity_id="opp-2")
    assert store.clear_legal_hold(scope="opportunity", scope_id="opp-1")
    assert not store.clear_legal_hold(scope="opportunity", scope_id="opp-1")
    assert not store.is_on_hold(opportunity_id="opp-1")


def test_setting_a_hold_twice_updates_it(store: SqliteStore, base_time: datetime) -> None:
    store.set_legal_hold(scope="posting", scope_id="p1", reason="first", set_by="A",
                         now=base_time)
    store.set_legal_hold(scope="posting", scope_id="p1", reason="second", set_by="B",
                         now=base_time)
    rows = store._conn.execute("SELECT reason, set_by FROM legal_holds").fetchall()
    assert [tuple(r) for r in rows] == [("second", "B")]


def test_posting_hold_covers_its_opportunities(store: SqliteStore,
                                               make_evaluation: MakeEvaluation,
                                               base_time: datetime) -> None:
    seed_candidate(store, make_evaluation, "opp-1", posting="p1")
    store.set_legal_hold(scope="posting", scope_id="p1", reason="charge filed",
                         set_by="Legal", now=base_time)
    assert store.is_on_hold(posting_id="p1")
    assert store.is_on_hold(opportunity_id="opp-1")  # via its evaluations
    assert store.is_on_hold(opportunity_id="opp-unknown", posting_id="p1")
    assert not store.is_on_hold(opportunity_id="opp-unknown")
    assert not store.is_on_hold(posting_id="p2")


def test_hold_scopes_are_separate(store: SqliteStore, base_time: datetime) -> None:
    store.set_legal_hold(scope="posting", scope_id="x", reason="r", set_by="L", now=base_time)
    assert not store.is_on_hold(opportunity_id="x")
    assert not store.clear_legal_hold(scope="opportunity", scope_id="x")


def test_hold_arguments_are_validated(store: SqliteStore, base_time: datetime) -> None:
    with pytest.raises(ValueError, match="scope"):
        store.set_legal_hold(scope="candidate", scope_id="x", reason="r",  # type: ignore[arg-type]
                             set_by="L", now=base_time)
    with pytest.raises(ValueError, match="scope_id"):
        store.set_legal_hold(scope="posting", scope_id="", reason="r", set_by="L",
                             now=base_time)
    with pytest.raises(ValueError, match="opportunity_id and/or posting_id"):
        store.is_on_hold()


# ------------------------------------------------------------------------ purge_candidate


def test_purge_candidate_deletes_pii_writebacks_and_note(store: SqliteStore,
                                                         make_evaluation: MakeEvaluation
                                                         ) -> None:
    seed_candidate(store, make_evaluation, "opp-1")
    seed_candidate(store, make_evaluation, "opp-2")
    result = store.purge_candidate("opp-1")
    assert result == PurgeResult(opportunity_id="opp-1", on_hold=False, pii_rows_deleted=2,
                                 audit_rows_kept=2)
    assert store.latest_for_opportunity("opp-1") is None
    assert store.writebacks_for_opportunity("opp-1") == []
    assert store.get_note_id("opp-1") is None
    assert len(store.audit_rows()) == 4
    # Other candidates are untouched.
    assert store.latest_for_opportunity("opp-2") is not None
    assert len(store.writebacks_for_opportunity("opp-2")) == 2
    assert store.get_note_id("opp-2") == "note-opp-2"


def test_purge_candidate_is_idempotent(store: SqliteStore,
                                       make_evaluation: MakeEvaluation) -> None:
    seed_candidate(store, make_evaluation, "opp-1")
    store.purge_candidate("opp-1")
    again = store.purge_candidate("opp-1")
    assert again.pii_rows_deleted == 0 and again.audit_rows_kept == 2


def test_purge_unknown_candidate(store: SqliteStore) -> None:
    assert store.purge_candidate("nobody") == PurgeResult(opportunity_id="nobody",
                                                          on_hold=False)


@pytest.mark.parametrize(("scope", "scope_id", "posting_arg"), [
    ("opportunity", "opp-1", None),  # the opportunity itself
    ("posting", "posting-1", None),  # a posting it has PII rows for
    ("posting", "posting-9", "posting-9"),  # the posting passed in
])
def test_purge_candidate_respects_holds(store: SqliteStore, make_evaluation: MakeEvaluation,
                                        base_time: datetime, scope: str, scope_id: str,
                                        posting_arg: str | None) -> None:
    seed_candidate(store, make_evaluation, "opp-1", posting="posting-1")
    store.set_legal_hold(scope=scope, scope_id=scope_id, reason="charge",  # type: ignore[arg-type]
                         set_by="Legal", now=base_time)
    result = store.purge_candidate("opp-1", posting_id=posting_arg)
    assert result == PurgeResult(opportunity_id="opp-1", on_hold=True, pii_rows_deleted=0,
                                 audit_rows_kept=2)
    assert count(store, "pii_evaluations") == 2
    assert len(store.writebacks_for_opportunity("opp-1")) == 2
    assert store.get_note_id("opp-1") == "note-opp-1"

    store.clear_legal_hold(scope=scope, scope_id=scope_id)  # type: ignore[arg-type]
    assert store.purge_candidate("opp-1", posting_id=posting_arg).pii_rows_deleted == 2


def test_hold_found_through_audit_rows_after_pii_purge(store: SqliteStore,
                                                       make_evaluation: MakeEvaluation,
                                                       base_time: datetime) -> None:
    seed_candidate(store, make_evaluation, "opp-1", posting="posting-1")
    store.purge_candidate("opp-1")
    store.set_legal_hold(scope="posting", scope_id="posting-1", reason="charge",
                         set_by="Legal", now=base_time)
    assert store.is_on_hold(opportunity_id="opp-1")


# -------------------------------------------------------- deletions suspended by a hold


def test_clearing_a_hold_completes_the_deletions_it_suspended(
        store: SqliteStore, make_evaluation: MakeEvaluation, base_time: datetime) -> None:
    """candidateDeleted during a posting hold leaves a tombstone and the PII. Releasing the
    hold honours the deletion at once instead of leaving the data until retention."""
    seed_candidate(store, make_evaluation, "opp-deleted", posting="posting-1")
    seed_candidate(store, make_evaluation, "opp-active", posting="posting-1")
    store.set_legal_hold(scope="posting", scope_id="posting-1", reason="charge",
                         set_by="Legal", now=base_time)
    store.record_deletion("opp-deleted", source="webhook", deleted_at=base_time)
    assert store.purge_candidate("opp-deleted").on_hold
    assert store.latest_for_opportunity("opp-deleted") is not None

    assert store.clear_legal_hold(scope="posting", scope_id="posting-1")

    assert store.latest_for_opportunity("opp-deleted") is None
    assert store.writebacks_for_opportunity("opp-deleted") == []
    assert store.get_note_id("opp-deleted") is None
    assert store.is_deleted("opp-deleted")
    assert len(store.audit_rows()) == 4  # pseudonymous rows stay, as after any purge
    # A hold release is not a purge: candidates who were not deleted keep their data.
    assert store.latest_for_opportunity("opp-active") is not None
    assert len(store.writebacks_for_opportunity("opp-active")) == 2
    assert store.get_note_id("opp-active") == "note-opp-active"


def test_purge_tombstoned_respects_remaining_holds_and_reports_each(
        store: SqliteStore, make_evaluation: MakeEvaluation, base_time: datetime) -> None:
    for opp in ("opp-1", "opp-2", "opp-3"):
        seed_candidate(store, make_evaluation, opp, posting="posting-1")
    store.set_legal_hold(scope="posting", scope_id="posting-1", reason="charge",
                         set_by="Legal", now=base_time)
    store.set_legal_hold(scope="opportunity", scope_id="opp-2", reason="subpoena",
                         set_by="Legal", now=base_time)
    store.record_deletion("opp-2", source="webhook", deleted_at=base_time)
    store.record_deletion("opp-1", source="reconcile",
                          deleted_at=base_time + timedelta(hours=1))
    assert store.purge_candidate("opp-1").on_hold and store.purge_candidate("opp-2").on_hold
    # The caller that wants to report what was purged runs purge_tombstoned itself.
    assert store.clear_legal_hold(scope="posting", scope_id="posting-1", purge_deleted=False)
    assert store.latest_for_opportunity("opp-1") is not None

    results = store.purge_tombstoned()

    assert results == [  # oldest deletion first
        PurgeResult(opportunity_id="opp-2", on_hold=True, audit_rows_kept=2),
        PurgeResult(opportunity_id="opp-1", on_hold=False, pii_rows_deleted=2,
                    audit_rows_kept=2),
    ]
    assert store.latest_for_opportunity("opp-1") is None
    assert store.latest_for_opportunity("opp-2") is not None  # its own hold still applies
    assert store.latest_for_opportunity("opp-3") is not None  # never deleted
    assert store.purge_tombstoned() == [PurgeResult(opportunity_id="opp-2", on_hold=True,
                                                    audit_rows_kept=2)]
    assert store.clear_legal_hold(scope="opportunity", scope_id="opp-2")
    assert store.latest_for_opportunity("opp-2") is None
    assert store.purge_tombstoned() == []  # completed deletions are not reported again


def test_purge_tombstoned_clears_leftover_writebacks_and_notes(store: SqliteStore,
                                                               base_time: datetime) -> None:
    store.set_note_id("opp-9", "note-9", now=base_time)
    store.record_writeback(WritebackResult(opportunity_id="opp-8", evaluation_id="e-8",
                                           executed=True), now=base_time)
    store.record_deletion("opp-9", source="webhook", deleted_at=base_time)
    store.record_deletion("opp-8", source="webhook", deleted_at=base_time + timedelta(1))
    store.record_deletion("opp-clean", source="webhook", deleted_at=base_time)
    assert store.purge_tombstoned() == [PurgeResult(opportunity_id="opp-9", on_hold=False),
                                        PurgeResult(opportunity_id="opp-8", on_hold=False)]
    assert store.get_note_id("opp-9") is None
    assert store.writebacks_for_opportunity("opp-8") == []
    assert store.purge_tombstoned() == []


# -------------------------------------------------------------------------- purge_expired


def test_purge_expired_counts_and_boundary(store: SqliteStore,
                                           make_evaluation: MakeEvaluation,
                                           base_time: datetime) -> None:
    now = base_time + timedelta(days=1461)
    seed_candidate(store, make_evaluation, "opp-old", n=2,
                   created_at=base_time - timedelta(microseconds=1))
    seed_candidate(store, make_evaluation, "opp-edge", n=1, created_at=base_time)
    seed_candidate(store, make_evaluation, "opp-new", n=1,
                   created_at=base_time + timedelta(days=10))
    removed = store.purge_expired(now=now, retention_days=1461)
    assert removed == 2
    remaining = {r["opportunity_id"] for r in store.audit_rows()}
    assert remaining == {"opp-edge", "opp-new"}  # exactly retention_days old is kept
    assert count(store, "pii_evaluations") == 2
    assert store.writebacks_for_opportunity("opp-old") == []
    assert len(store.writebacks_for_opportunity("opp-edge")) == 1
    assert store.purge_expired(now=now, retention_days=1461) == 0


def test_purge_expired_keeps_newer_writeback_of_purged_evaluation(
        store: SqliteStore, make_evaluation: MakeEvaluation, base_time: datetime) -> None:
    """A cached result can be written to Lever long after it was produced: the write-back
    record keeps its own retention clock instead of going with the evaluation."""
    now = base_time + timedelta(days=1461)
    old = make_evaluation(opportunity_id="opp-old", posting_id="posting-1",
                          created_at=base_time - timedelta(days=1))
    store.record_evaluation(old, cache_key="k")
    store.record_writeback(WritebackResult(opportunity_id="opp-old",
                                           evaluation_id=old.evaluation_id, executed=True),
                           now=now - timedelta(days=30))
    assert store.purge_expired(now=now, retention_days=1461) == 1
    assert store.audit_rows() == []
    assert len(store.writebacks_for_opportunity("opp-old")) == 1
    assert store.purge_expired(now=now + timedelta(days=1461), retention_days=1461) == 0
    assert store.writebacks_for_opportunity("opp-old") == []


def test_purge_expired_removes_audit_rows_left_after_candidate_purge(
        store: SqliteStore, make_evaluation: MakeEvaluation, base_time: datetime) -> None:
    seed_candidate(store, make_evaluation, "opp-1", n=2, created_at=base_time)
    store.purge_candidate("opp-1")
    assert store.purge_expired(now=base_time + timedelta(days=400), retention_days=365) == 2
    assert store.audit_rows() == []


@pytest.mark.parametrize(("scope", "scope_id"), [
    ("opportunity", "opp-old"),
    ("posting", "posting-old"),
    ("posting", "posting-current"),  # the opportunity also has a row for a held posting
])
def test_purge_expired_respects_holds(store: SqliteStore, make_evaluation: MakeEvaluation,
                                      base_time: datetime, scope: str, scope_id: str) -> None:
    now = base_time + timedelta(days=800)
    seed_candidate(store, make_evaluation, "opp-old", posting="posting-old", n=2,
                   created_at=base_time)
    seed_candidate(store, make_evaluation, "opp-old", posting="posting-current", n=1,
                   created_at=now)
    seed_candidate(store, make_evaluation, "opp-other", posting="posting-x", n=1,
                   created_at=base_time)
    store.set_legal_hold(scope=scope, scope_id=scope_id, reason="charge",  # type: ignore[arg-type]
                         set_by="Legal", now=base_time)
    assert store.purge_expired(now=now, retention_days=365) == 1  # only opp-other
    assert {r["opportunity_id"] for r in store.audit_rows()} == {"opp-old"}
    assert count(store, "pii_evaluations") == 3


def test_purge_expired_validates_retention(store: SqliteStore, base_time: datetime) -> None:
    with pytest.raises(ValueError, match="retention_days"):
        store.purge_expired(now=base_time, retention_days=0)


def test_purge_expired_on_empty_store(store: SqliteStore, base_time: datetime) -> None:
    assert store.purge_expired(now=base_time, retention_days=365) == 0


# ------------------------------------------------------- what a purge leaves in the files


def files_containing(db_path: Path, needle: str) -> list[str]:
    """Names of the database, -wal and -shm files that hold ``needle``'s bytes."""
    candidates = [db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")]
    return [p.name for p in candidates if p.exists() and needle.encode() in p.read_bytes()]


@pytest.mark.parametrize("how", ["purge_candidate", "purge_expired"])
def test_purged_candidate_data_leaves_the_database_files(db_path: Path,
                                                         make_evaluation: MakeEvaluation,
                                                         base_time: datetime, how: str) -> None:
    """Deleting rows only unlinks them in SQLite by default: a copy or backup of the file
    would still hold the name, the evidence and the resume text. secure_delete zeroes them,
    and the WAL checkpoint after a purge drops the page images written before it."""
    name, resume = "Zephyrine Quokkabottom", "Built the Quokkabottom lakehouse on Dataflow."
    store = SqliteStore(db_path, store_resume_text=True)
    try:
        assert store._conn.execute("PRAGMA secure_delete").fetchone()[0] == 1
        for _ in range(3):
            evaluation = make_evaluation(opportunity_id="opp-x", candidate_name=name,
                                         created_at=base_time)
            store.record_evaluation(evaluation, cache_key="k", redacted_text=resume)
        assert files_containing(db_path, name) and files_containing(db_path, resume)

        if how == "purge_candidate":
            assert store.purge_candidate("opp-x").pii_rows_deleted == 3
        else:
            assert store.purge_expired(now=base_time + timedelta(days=400),
                                       retention_days=365) == 3

        assert store.latest_for_opportunity("opp-x") is None
        assert files_containing(db_path, name) == []
        assert files_containing(db_path, resume) == []
    finally:
        store.close()
    assert files_containing(db_path, name) == []
    assert files_containing(db_path, resume) == []
