"""Write-back records, Lever note ids, the webhook replay cache and deletion tombstones."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from applicant_ranking.domain import WritebackResult
from applicant_ranking.store import SqliteStore

# ----------------------------------------------------------------------------- writebacks


def test_writebacks_are_recorded_in_order(store: SqliteStore, base_time: datetime) -> None:
    planned = WritebackResult(opportunity_id="opp-1", evaluation_id="e-1", executed=False,
                              dry_run_reason="dry run")
    sent = WritebackResult(opportunity_id="opp-1", evaluation_id="e-1", executed=True,
                           note_id="note-9", responses=[{"path": "/x", "status": 200}])
    store.record_writeback(sent, now=base_time + timedelta(minutes=1))
    store.record_writeback(planned, now=base_time)
    store.record_writeback(WritebackResult(opportunity_id="opp-2", evaluation_id="e-2",
                                           executed=False))
    assert store.writebacks_for_opportunity("opp-1") == [planned, sent]
    assert len(store.writebacks_for_opportunity("opp-2")) == 1
    assert store.writebacks_for_opportunity("opp-3") == []


def test_same_evaluation_can_be_written_back_twice(store: SqliteStore) -> None:
    result = WritebackResult(opportunity_id="opp-1", evaluation_id="e-1", executed=False)
    store.record_writeback(result)
    store.record_writeback(result)
    assert len(store.writebacks_for_opportunity("opp-1")) == 2


# ---------------------------------------------------------------------------- note ids


def test_note_id_set_and_update(store: SqliteStore, base_time: datetime) -> None:
    assert store.get_note_id("opp-1") is None
    store.set_note_id("opp-1", "note-1", now=base_time)
    assert store.get_note_id("opp-1") == "note-1"
    store.set_note_id("opp-1", "note-2", now=base_time + timedelta(minutes=5))
    assert store.get_note_id("opp-1") == "note-2"
    assert store.get_note_id("opp-2") is None
    row = store._conn.execute("SELECT updated_at FROM lever_notes").fetchone()
    assert row[0] == "2026-10-08T12:05:00.000000+00:00"


# ------------------------------------------------------------------------ webhook tokens


def test_webhook_token_replay_within_ttl(store: SqliteStore, base_time: datetime) -> None:
    assert store.remember_webhook_token("tok-1", now=base_time, ttl_seconds=600)
    assert not store.remember_webhook_token("tok-1", now=base_time + timedelta(seconds=1),
                                            ttl_seconds=600)
    assert not store.remember_webhook_token("tok-1", now=base_time + timedelta(seconds=600),
                                            ttl_seconds=600)
    assert store.remember_webhook_token("tok-2", now=base_time, ttl_seconds=600)


def test_webhook_token_is_new_again_after_ttl(store: SqliteStore, base_time: datetime) -> None:
    assert store.remember_webhook_token("tok-1", now=base_time, ttl_seconds=600)
    later = base_time + timedelta(seconds=601)
    assert store.remember_webhook_token("tok-1", now=later, ttl_seconds=600)
    assert not store.remember_webhook_token("tok-1", now=later + timedelta(seconds=1),
                                            ttl_seconds=600)


def test_expired_tokens_are_pruned(store: SqliteStore, base_time: datetime) -> None:
    for i in range(5):
        store.remember_webhook_token(f"old-{i}", now=base_time, ttl_seconds=60)
    store.remember_webhook_token("fresh", now=base_time + timedelta(hours=1), ttl_seconds=60)
    assert store._conn.execute("SELECT COUNT(*) FROM webhook_tokens").fetchone()[0] == 1


def test_tokens_are_stored_hashed(store: SqliteStore, base_time: datetime) -> None:
    store.remember_webhook_token("secret-looking-token", now=base_time, ttl_seconds=60)
    stored = store._conn.execute("SELECT token FROM webhook_tokens").fetchone()[0]
    assert "secret-looking-token" not in stored
    assert len(stored) == 64


def test_webhook_ttl_must_be_positive(store: SqliteStore, base_time: datetime) -> None:
    with pytest.raises(ValueError, match="ttl_seconds"):
        store.remember_webhook_token("t", now=base_time, ttl_seconds=0)


def test_naive_now_is_treated_as_utc(store: SqliteStore) -> None:
    naive = datetime(2026, 10, 8, 12, 0)
    assert store.remember_webhook_token("t", now=naive, ttl_seconds=60)
    stored = store._conn.execute("SELECT seen_at FROM webhook_tokens").fetchone()[0]
    assert stored == "2026-10-08T12:00:00.000000+00:00"


# ----------------------------------------------------------------------------- deletions


def test_deletion_tombstone(store: SqliteStore, base_time: datetime) -> None:
    assert not store.is_deleted("opp-1")
    store.record_deletion("opp-1", source="webhook", deleted_at=base_time)
    assert store.is_deleted("opp-1")
    assert not store.is_deleted("opp-2")


def test_deletion_is_idempotent_and_first_wins(store: SqliteStore,
                                               base_time: datetime) -> None:
    store.record_deletion("opp-1", source="webhook", deleted_at=base_time)
    store.record_deletion("opp-1", source="reconcile", deleted_at=base_time + timedelta(days=1))
    rows = store._conn.execute("SELECT deleted_at, source FROM deletions").fetchall()
    assert [tuple(r) for r in rows] == [("2026-10-08T12:00:00.000000+00:00", "webhook")]


def test_tombstones_survive_purges(store: SqliteStore, base_time: datetime) -> None:
    store.record_deletion("opp-1", source="webhook", deleted_at=base_time)
    store.purge_candidate("opp-1")
    store.purge_expired(now=base_time + timedelta(days=5000), retention_days=365)
    assert store.is_deleted("opp-1")
