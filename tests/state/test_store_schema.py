"""SqliteStore construction: files, WAL, migrations, closing."""

from __future__ import annotations

import sqlite3
import stat
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pytest

from applicant_ranking.domain import Evaluation
from applicant_ranking.store import MIGRATIONS, SCHEMA_VERSION, SqliteStore, Store, StoreError


def test_memory_store_works(make_evaluation: Callable[..., Evaluation]) -> None:
    with SqliteStore(":memory:") as store:
        evaluation = make_evaluation()
        store.record_evaluation(evaluation, cache_key="k")
        assert store.get_evaluation(evaluation.evaluation_id) == evaluation
        assert store.schema_version() == SCHEMA_VERSION


def test_path_object_for_memory_works() -> None:
    with SqliteStore(Path(":memory:")) as store:
        assert store.audit_rows() == []


def test_satisfies_store_protocol() -> None:
    with SqliteStore(":memory:") as store:
        typed: Store = store
        for name in ("record_evaluation", "get_evaluation", "find_cached",
                     "latest_for_opportunity", "evaluations_for_posting", "audit_rows",
                     "record_writeback", "get_note_id", "set_note_id",
                     "remember_webhook_token", "record_deletion", "is_deleted",
                     "purge_candidate", "set_legal_hold", "clear_legal_hold", "is_on_hold",
                     "purge_expired", "close"):
            assert callable(getattr(typed, name))


def test_file_store_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "c" / "state.sqlite3"
    with SqliteStore(path):
        pass
    assert path.is_file()


def test_string_path_works(tmp_path: Path) -> None:
    path = tmp_path / "x" / "state.sqlite3"
    with SqliteStore(str(path)):
        pass
    assert path.is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_new_database_file_is_owner_only(db_path: Path) -> None:
    with SqliteStore(db_path):
        pass
    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600


def test_file_store_uses_wal(store: SqliteStore, db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()


def test_schema_is_created_once_and_survives_reopen(
        db_path: Path, make_evaluation: Callable[..., Evaluation]) -> None:
    evaluation = make_evaluation()
    with SqliteStore(db_path) as first:
        first.record_evaluation(evaluation, cache_key="k")
    with SqliteStore(db_path) as second:
        assert second.get_evaluation(evaluation.evaluation_id) == evaluation
        assert second.schema_version() == SCHEMA_VERSION
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
        assert rows == [(version,) for version, _ in MIGRATIONS]
        triggers = conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'"
                                ).fetchall()
        assert ("audit_log_append_only",) in triggers
    finally:
        conn.close()


def test_newer_schema_is_refused(db_path: Path) -> None:
    with SqliteStore(db_path):
        pass
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute("INSERT INTO schema_migrations (version, applied_at) VALUES (99, 'x')")
    conn.close()
    with pytest.raises(StoreError, match="schema version 99"):
        SqliteStore(db_path)


def test_not_a_database_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    path.write_bytes(b"this is not a sqlite database, just some text " * 20)
    with pytest.raises(sqlite3.DatabaseError):
        SqliteStore(path)


def test_close_is_idempotent_and_blocks_further_use(store: SqliteStore,
                                                     base_time: datetime) -> None:
    store.close()
    store.close()
    with pytest.raises(StoreError, match="closed"):
        store.is_deleted("opp-1")
    with pytest.raises(StoreError, match="closed"):
        store.record_deletion("opp-1", source="webhook", deleted_at=base_time)
