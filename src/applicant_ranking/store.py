"""Local state: an append-only audit log, a separately purgeable PII store, idempotency
cache, write-back bookkeeping, webhook replay cache, deletions and legal holds.

SQLite (stdlib ``sqlite3``) with WAL; one connection guarded by a lock
(``check_same_thread=False``) so FastAPI worker threads and asyncio code can share it.
Cloud Run's disk is ephemeral: production needs a durable backend (Firestore/BigQuery
with insert-only IAM for the audit table). Keep this class's public surface the
``Store`` protocol so a second implementation can drop in.

Tables:
- ``audit_log`` (append-only; a trigger aborts any UPDATE): evaluation_id PK, created_at,
  mode, posting_id, opportunity_id, application_id, resume_id, criteria_role_id,
  criteria_version, criteria_hash, prompt_version, redaction_version, model_id,
  model_version, generation_config_json, file_hmac, redacted_text_hmac, status, score,
  tier, holistic_score, flag_codes_json, review_reasons_json, validation_json (without
  the model-written ``unknown_ids_dropped`` strings; their count stays), usage_json,
  rationale_publishable. Pseudonymous only: no names, no quotes, no text.
  Rows are deleted only by ``purge_expired`` (retention), never on candidate deletion
  (Legal decides otherwise later).
- ``pii_evaluations``: evaluation_id PK, opportunity_id, posting_id, cache_key,
  created_at, evaluation_json (full Evaluation incl. candidate name and evidence),
  redacted_text (only when ``store_resume_text``). Deleted by ``purge_candidate``.
- ``writebacks``: evaluation_id, opportunity_id, executed, created_at, result_json.
- ``lever_notes``: opportunity_id PK, note_id, updated_at.
- ``webhook_tokens``: token PK, seen_at.
- ``deletions``: opportunity_id PK, deleted_at, source.
- ``legal_holds``: (scope, scope_id) PK, reason, set_by, set_at.
All timestamps stored as ISO-8601 UTC text.

Legal holds: an opportunity is on hold when it is held itself, or when any posting it
has evaluations for (or the ``posting_id`` passed in) is held. Holds suspend both
``purge_candidate`` and ``purge_expired``. A deletion that a hold suspended is still on
record (its tombstone), so clearing the hold completes it (``purge_tombstoned``, also run
by the ``purge`` command) instead of leaving the PII until the retention purge.

Webhook tokens are stored as SHA-256 digests, never as received. The database file is
created owner-only (0600) because it holds candidate data, deleted rows are overwritten
with zeros (``PRAGMA secure_delete``), and every purge checkpoints and truncates the WAL
so the page images written before the delete do not keep the data alive in the ``-wal``
file. Crypto-shredding is the plan for the production backend (design.md).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from applicant_ranking.domain import Evaluation, RunMode, Status, WritebackResult, utcnow

logger = logging.getLogger(__name__)

HoldScope = Literal["posting", "opportunity"]
_HOLD_SCOPES = ("posting", "opportunity")

_SCHEMA_V1 = """
CREATE TABLE audit_log (
    evaluation_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    posting_id TEXT,
    opportunity_id TEXT NOT NULL,
    application_id TEXT,
    resume_id TEXT,
    criteria_role_id TEXT,
    criteria_version INTEGER,
    criteria_hash TEXT,
    prompt_version TEXT,
    redaction_version TEXT,
    model_id TEXT,
    model_version TEXT,
    generation_config_json TEXT,
    file_hmac TEXT,
    redacted_text_hmac TEXT,
    status TEXT NOT NULL,
    score INTEGER,
    tier TEXT,
    holistic_score INTEGER,
    flag_codes_json TEXT NOT NULL,
    review_reasons_json TEXT NOT NULL,
    validation_json TEXT NOT NULL,
    usage_json TEXT,
    rationale_publishable INTEGER NOT NULL
);
CREATE INDEX audit_log_created ON audit_log (created_at);
CREATE INDEX audit_log_posting ON audit_log (posting_id, created_at);
CREATE INDEX audit_log_opportunity ON audit_log (opportunity_id);
CREATE TRIGGER audit_log_append_only BEFORE UPDATE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only');
END;

CREATE TABLE pii_evaluations (
    evaluation_id TEXT PRIMARY KEY REFERENCES audit_log (evaluation_id),
    opportunity_id TEXT NOT NULL,
    posting_id TEXT,
    cache_key TEXT,
    created_at TEXT NOT NULL,
    evaluation_json TEXT NOT NULL,
    redacted_text TEXT
);
CREATE INDEX pii_cache_key ON pii_evaluations (cache_key, created_at);
CREATE INDEX pii_opportunity ON pii_evaluations (opportunity_id, created_at);
CREATE INDEX pii_posting ON pii_evaluations (posting_id, created_at);

CREATE TABLE writebacks (
    evaluation_id TEXT NOT NULL,
    opportunity_id TEXT NOT NULL,
    executed INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    result_json TEXT NOT NULL
);
CREATE INDEX writebacks_opportunity ON writebacks (opportunity_id, created_at);
CREATE INDEX writebacks_evaluation ON writebacks (evaluation_id);

CREATE TABLE lever_notes (
    opportunity_id TEXT PRIMARY KEY,
    note_id TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE webhook_tokens (
    token TEXT PRIMARY KEY,
    seen_at TEXT NOT NULL
);
CREATE INDEX webhook_tokens_seen ON webhook_tokens (seen_at);

CREATE TABLE deletions (
    opportunity_id TEXT PRIMARY KEY,
    deleted_at TEXT NOT NULL,
    source TEXT NOT NULL
);

CREATE TABLE legal_holds (
    scope TEXT NOT NULL CHECK (scope IN ('posting', 'opportunity')),
    scope_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    set_by TEXT NOT NULL,
    set_at TEXT NOT NULL,
    PRIMARY KEY (scope, scope_id)
);
"""

# The audit record carries the jurisdiction regimes applied and
# the gate outcome for the run that produced it.
_SCHEMA_V2 = """
ALTER TABLE audit_log ADD COLUMN policy_regimes_json TEXT;
ALTER TABLE audit_log ADD COLUMN policy_allowed INTEGER;
"""

# (version, DDL). Append new migrations; never edit an applied one.
MIGRATIONS: tuple[tuple[int, str], ...] = ((1, _SCHEMA_V1), (2, _SCHEMA_V2))
SCHEMA_VERSION = MIGRATIONS[-1][0]

_AUDIT_COLUMNS = (
    "evaluation_id", "created_at", "mode", "posting_id", "opportunity_id", "application_id",
    "resume_id", "criteria_role_id", "criteria_version", "criteria_hash", "prompt_version",
    "redaction_version", "model_id", "model_version", "generation_config_json", "file_hmac",
    "redacted_text_hmac", "status", "score", "tier", "holistic_score", "flag_codes_json",
    "review_reasons_json", "validation_json", "usage_json", "rationale_publishable",
    "policy_regimes_json", "policy_allowed",
)  # fmt: skip
_AUDIT_JSON_COLUMNS = (
    "generation_config_json", "flag_codes_json", "review_reasons_json", "validation_json",
    "usage_json", "policy_regimes_json",
)  # fmt: skip
# Built from the constant column list above, never from input.
_AUDIT_INSERT_SQL = (
    f"INSERT INTO audit_log ({', '.join(_AUDIT_COLUMNS)}) "  # noqa: S608
    f"VALUES ({', '.join(':' + c for c in _AUDIT_COLUMNS)})"
)
_AUDIT_SELECT_SQL = f"SELECT {', '.join(_AUDIT_COLUMNS)} FROM audit_log"  # noqa: S608

# True when an opportunity (:opp) or a posting (:posting, or any posting the opportunity
# has rows for) is on legal hold.
_HELD_SQL = """
SELECT EXISTS (
    SELECT 1 FROM legal_holds h
    WHERE (h.scope = 'opportunity' AND h.scope_id = :opp)
       OR (h.scope = 'posting' AND (
               h.scope_id = :posting
            OR h.scope_id IN (SELECT a.posting_id FROM audit_log a
                              WHERE a.opportunity_id = :opp)
            OR h.scope_id IN (SELECT p.posting_id FROM pii_evaluations p
                              WHERE p.opportunity_id = :opp)))
)
"""

_EXPIRED_SQL = """
SELECT a.evaluation_id FROM audit_log a
WHERE a.created_at < :cutoff
  AND NOT EXISTS (
    SELECT 1 FROM legal_holds h
    WHERE (h.scope = 'opportunity' AND h.scope_id = a.opportunity_id)
       OR (h.scope = 'posting' AND (
               h.scope_id = a.posting_id
            OR h.scope_id IN (SELECT a2.posting_id FROM audit_log a2
                              WHERE a2.opportunity_id = a.opportunity_id)
            OR h.scope_id IN (SELECT p.posting_id FROM pii_evaluations p
                              WHERE p.opportunity_id = a.opportunity_id))))
ORDER BY a.created_at
"""

# Write-back records older than the cutoff whose evaluation is gone (purged now or
# earlier) and whose opportunity is not on hold. A write-back can be much newer than its
# evaluation (a cached result written to Lever later), so it expires on its own clock.
_EXPIRED_WRITEBACKS_SQL = """
DELETE FROM writebacks
WHERE created_at < :cutoff
  AND NOT EXISTS (SELECT 1 FROM audit_log a WHERE a.evaluation_id = writebacks.evaluation_id)
  AND NOT EXISTS (
    SELECT 1 FROM legal_holds h
    WHERE (h.scope = 'opportunity' AND h.scope_id = writebacks.opportunity_id)
       OR (h.scope = 'posting' AND (
               h.scope_id IN (SELECT a2.posting_id FROM audit_log a2
                              WHERE a2.opportunity_id = writebacks.opportunity_id)
            OR h.scope_id IN (SELECT p.posting_id FROM pii_evaluations p
                              WHERE p.opportunity_id = writebacks.opportunity_id))))
"""


# Deleted (tombstoned) opportunities that still have purgeable rows: a legal hold stopped
# the purge when the deletion arrived, or the purge never ran.
_TOMBSTONED_SQL = """
SELECT d.opportunity_id FROM deletions d
WHERE EXISTS (SELECT 1 FROM pii_evaluations p WHERE p.opportunity_id = d.opportunity_id)
   OR EXISTS (SELECT 1 FROM writebacks w WHERE w.opportunity_id = d.opportunity_id)
   OR EXISTS (SELECT 1 FROM lever_notes n WHERE n.opportunity_id = d.opportunity_id)
ORDER BY d.deleted_at, d.opportunity_id
"""


class StoreError(RuntimeError):
    """The store can't do what was asked (closed, newer schema, ...)."""


class DuplicateEvaluationError(StoreError):
    """An evaluation_id was recorded before; the audit log is append-only."""


class PurgeResult(BaseModel):
    opportunity_id: str
    on_hold: bool
    pii_rows_deleted: int = 0
    audit_rows_kept: int = 0
    # False when a candidateDeleted webhook named an opportunity Lever still serves as
    # live: nothing was tombstoned or purged (pipeline.handle_candidate_deleted).
    confirmed: bool = True


class Store(Protocol):
    def record_evaluation(self, evaluation: Evaluation, *, cache_key: str | None,
                          generation_config: dict[str, Any] | None = None,
                          redacted_text: str | None = None,
                          policy_regimes: Sequence[str] | None = None,
                          policy_allowed: bool | None = None) -> None: ...
    def get_evaluation(self, evaluation_id: str) -> Evaluation | None: ...
    def find_cached(self, cache_key: str, *,
                    evaluation_mode: bool | None = None) -> Evaluation | None: ...
    def latest_for_opportunity(self, opportunity_id: str) -> Evaluation | None: ...
    def evaluations_for_posting(self, posting_id: str, *,
                                latest_only: bool = True) -> list[Evaluation]: ...
    def audit_rows(self, *, posting_id: str | None = None) -> list[dict[str, Any]]: ...
    def record_writeback(self, result: WritebackResult, *,
                         now: datetime | None = None) -> None: ...
    def get_note_id(self, opportunity_id: str) -> str | None: ...
    def set_note_id(self, opportunity_id: str, note_id: str, *, now: datetime) -> None: ...
    def remember_webhook_token(self, token: str, *, now: datetime, ttl_seconds: int) -> bool: ...
    def record_deletion(self, opportunity_id: str, *, source: str,
                        deleted_at: datetime) -> None: ...
    def is_deleted(self, opportunity_id: str) -> bool: ...
    def purge_candidate(self, opportunity_id: str, *,
                        posting_id: str | None = None) -> PurgeResult: ...
    def purge_tombstoned(self) -> list[PurgeResult]: ...
    def set_legal_hold(self, *, scope: HoldScope, scope_id: str, reason: str, set_by: str,
                       now: datetime) -> None: ...
    def clear_legal_hold(self, *, scope: HoldScope, scope_id: str) -> bool: ...
    def is_on_hold(self, *, opportunity_id: str | None = None,
                   posting_id: str | None = None) -> bool: ...
    def purge_expired(self, *, now: datetime, retention_days: int) -> int: ...
    def close(self) -> None: ...


def _iso(value: datetime) -> str:
    """Fixed-width ISO-8601 UTC, so text comparison orders correctly. Naive = UTC."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _split_sql(script: str) -> list[str]:
    """Split a DDL script into statements (trigger bodies stay whole)."""
    statements: list[str] = []
    buffer = ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statements.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        raise ValueError("migration ends with an incomplete SQL statement")
    return statements


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _create_private_file(path: Path) -> None:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return
    os.close(fd)


def _audit_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Audit row with JSON columns decoded and renamed without the ``_json`` suffix
    (``flag_codes``, ``review_reasons``, ``validation``, ``usage``,
    ``generation_config``, ``policy_regimes``), ``rationale_publishable`` as a bool and
    ``policy_allowed`` as a bool or None (rows recorded before schema version 2)."""
    out: dict[str, Any] = {}
    for column in _AUDIT_COLUMNS:
        value = row[column]
        if column in _AUDIT_JSON_COLUMNS:
            out[column.removesuffix("_json")] = None if value is None else json.loads(value)
        elif column == "rationale_publishable":
            out[column] = bool(value)
        elif column == "policy_allowed":
            out[column] = None if value is None else bool(value)
        else:
            out[column] = value
    return out


class SqliteStore:
    def __init__(self, path: Path | str, *, store_resume_text: bool = False) -> None:
        """Creates parent dirs and the schema. ``":memory:"`` works for tests."""
        self._store_resume_text = store_resume_text
        self._lock = threading.Lock()
        self._closed = False
        target = str(path)
        if target != ":memory:":
            file = Path(path)
            file.parent.mkdir(parents=True, exist_ok=True)
            _create_private_file(file)
        self.path = target
        self._conn = sqlite3.connect(target, check_same_thread=False, isolation_level=None,
                                     timeout=10.0)
        try:
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            # Per connection: deleted rows (names, evidence, resume text) are zeroed, not
            # just unlinked, so a purge removes them from the file, not only from queries.
            self._conn.execute("PRAGMA secure_delete=ON")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._migrate()
        except BaseException:
            self._conn.close()
            raise

    def __enter__(self) -> SqliteStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -------------------------------------------------------------------- connection

    def _migrate(self) -> None:
        """Apply pending migrations, each in its own transaction. The version is re-read
        under the write lock, so two processes opening a new file don't both apply it."""
        conn = self._conn
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations ("
                     "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
        for version, ddl in MIGRATIONS:
            conn.execute("BEGIN IMMEDIATE")
            try:
                current = conn.execute(
                    "SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()[0]
                if current > SCHEMA_VERSION:
                    raise StoreError(f"state database {self.path} has schema version "
                                     f"{current}; this build supports up to {SCHEMA_VERSION}")
                applied = version > current
                if applied:
                    for statement in _split_sql(ddl):
                        conn.execute(statement)
                    conn.execute("INSERT INTO schema_migrations (version, applied_at) "
                                 "VALUES (?, ?)", (version, _iso(utcnow())))
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
            if applied:
                logger.info("state database %s migrated to schema version %d", self.path,
                            version)

    def _connection(self) -> sqlite3.Connection:
        if self._closed:
            raise StoreError("store is closed")
        return self._conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._connection()
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    @contextmanager
    def _reading(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            yield self._connection()

    def schema_version(self) -> int:
        with self._reading() as conn:
            return conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]

    # ------------------------------------------------------------------- evaluations

    def record_evaluation(self, evaluation: Evaluation, *, cache_key: str | None,
                          generation_config: dict[str, Any] | None = None,
                          redacted_text: str | None = None,
                          policy_regimes: Sequence[str] | None = None,
                          policy_allowed: bool | None = None) -> None:
        """Insert one audit row and one PII row in a single transaction. Re-recording an
        existing evaluation_id is an error (append-only). ``policy_regimes`` and
        ``policy_allowed`` record the jurisdiction decision of the run (None = no gate,
        e.g. a local evaluation)."""
        e = evaluation
        created_at = _iso(e.created_at)
        audit: dict[str, Any] = {
            "evaluation_id": e.evaluation_id,
            "created_at": created_at,
            "mode": e.mode.value,
            "posting_id": e.posting_id,
            "opportunity_id": e.opportunity_id,
            "application_id": e.application_id,
            "resume_id": e.resume_id,
            "criteria_role_id": e.criteria_role_id,
            "criteria_version": e.criteria_version,
            "criteria_hash": e.criteria_hash,
            "prompt_version": e.prompt_version,
            "redaction_version": e.redaction_version,
            "model_id": e.model_id,
            "model_version": e.model_version,
            "generation_config_json": (
                None if generation_config is None
                else json.dumps(generation_config, sort_keys=True, default=str)
            ),
            "file_hmac": e.file_hmac,
            "redacted_text_hmac": e.redacted_text_hmac,
            "status": e.status.value,
            "score": e.score,
            "tier": e.tier.value if e.tier is not None else None,
            "holistic_score": e.holistic_score,
            "flag_codes_json": json.dumps([flag.code.value for flag in e.flags]),
            "review_reasons_json": json.dumps(list(e.review_reasons)),
            # Unknown assessment ids are model-written text: the audit row keeps their
            # count, the strings stay in the PII row (purged with the candidate).
            "validation_json": e.validation.model_dump_json(exclude={"unknown_ids_dropped"}),
            "usage_json": e.usage.model_dump_json() if e.usage is not None else None,
            "rationale_publishable": int(e.rationale_publishable),
            "policy_regimes_json": (
                None if policy_regimes is None else json.dumps(list(policy_regimes))
            ),
            "policy_allowed": None if policy_allowed is None else int(policy_allowed),
        }
        pii = {
            "evaluation_id": e.evaluation_id,
            "opportunity_id": e.opportunity_id,
            "posting_id": e.posting_id,
            "cache_key": cache_key,
            "created_at": created_at,
            "evaluation_json": e.model_dump_json(),
            "redacted_text": redacted_text if self._store_resume_text else None,
        }
        with self._transaction() as conn:
            exists = conn.execute("SELECT 1 FROM audit_log WHERE evaluation_id = ?",
                                  (e.evaluation_id,)).fetchone()
            if exists:
                raise DuplicateEvaluationError(
                    f"evaluation {e.evaluation_id} is already recorded (append-only)"
                )
            conn.execute(_AUDIT_INSERT_SQL, audit)
            conn.execute(
                "INSERT INTO pii_evaluations (evaluation_id, opportunity_id, posting_id, "
                "cache_key, created_at, evaluation_json, redacted_text) VALUES "
                "(:evaluation_id, :opportunity_id, :posting_id, :cache_key, :created_at, "
                ":evaluation_json, :redacted_text)",
                pii,
            )

    def get_evaluation(self, evaluation_id: str) -> Evaluation | None:
        with self._reading() as conn:
            row = conn.execute("SELECT evaluation_json FROM pii_evaluations "
                               "WHERE evaluation_id = ?", (evaluation_id,)).fetchone()
        return Evaluation.model_validate_json(row[0]) if row else None

    def get_redacted_text(self, evaluation_id: str) -> str | None:
        """The stored redacted resume text (only kept when ``store_resume_text``)."""
        with self._reading() as conn:
            row = conn.execute("SELECT redacted_text FROM pii_evaluations "
                               "WHERE evaluation_id = ?", (evaluation_id,)).fetchone()
        return row[0] if row else None

    def find_cached(self, cache_key: str, *,
                    evaluation_mode: bool | None = None) -> Evaluation | None:
        """Most recent evaluation with this cache key whose status is not ERROR (errors
        are re-tried) and whose PII row still exists. ``evaluation_mode`` True/False
        restricts the search to EVALUATION-mode rows / to DRY_RUN and LIVE rows, so an
        offline-evaluation result is never reused for a real run or the reverse."""
        mode_sql = ""
        params: list[Any] = [cache_key, Status.ERROR.value]
        if evaluation_mode is not None:
            mode_sql = " AND a.mode = ?" if evaluation_mode else " AND a.mode != ?"
            params.append(RunMode.EVALUATION.value)
        with self._reading() as conn:
            row = conn.execute(
                "SELECT p.evaluation_json FROM pii_evaluations p "  # noqa: S608 - constant
                "JOIN audit_log a ON a.evaluation_id = p.evaluation_id "
                "WHERE p.cache_key = ? AND a.status != ?" + mode_sql + " "
                "ORDER BY p.created_at DESC, p.rowid DESC LIMIT 1",
                params,
            ).fetchone()
        return Evaluation.model_validate_json(row[0]) if row else None

    def latest_for_opportunity(self, opportunity_id: str) -> Evaluation | None:
        """Newest evaluation (any status) whose PII row still exists; ties go to the one
        recorded last."""
        with self._reading() as conn:
            row = conn.execute(
                "SELECT evaluation_json FROM pii_evaluations WHERE opportunity_id = ? "
                "ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (opportunity_id,),
            ).fetchone()
        return Evaluation.model_validate_json(row[0]) if row else None

    def evaluations_for_posting(self, posting_id: str, *,
                                latest_only: bool = True) -> list[Evaluation]:
        """Oldest first. ``latest_only`` keeps the newest evaluation per opportunity."""
        if latest_only:
            sql = (
                "SELECT evaluation_json FROM ("
                " SELECT evaluation_json, created_at, rowid AS rid, ROW_NUMBER() OVER ("
                "  PARTITION BY opportunity_id ORDER BY created_at DESC, rowid DESC) AS rn"
                " FROM pii_evaluations WHERE posting_id = ?"
                ") WHERE rn = 1 ORDER BY created_at, rid"
            )
        else:
            sql = ("SELECT evaluation_json FROM pii_evaluations WHERE posting_id = ? "
                   "ORDER BY created_at, rowid")
        with self._reading() as conn:
            rows = conn.execute(sql, (posting_id,)).fetchall()
        return [Evaluation.model_validate_json(row[0]) for row in rows]

    def audit_rows(self, *, posting_id: str | None = None) -> list[dict[str, Any]]:
        """Pseudonymous audit rows, oldest first; see ``_audit_dict`` for the shape."""
        sql = _AUDIT_SELECT_SQL
        params: tuple[Any, ...] = ()
        if posting_id is not None:
            sql += " WHERE posting_id = ?"
            params = (posting_id,)
        sql += " ORDER BY created_at, rowid"
        with self._reading() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_audit_dict(row) for row in rows]

    # -------------------------------------------------------------------- write-back

    def record_writeback(self, result: WritebackResult, *, now: datetime | None = None) -> None:
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO writebacks (evaluation_id, opportunity_id, executed, created_at, "
                "result_json) VALUES (?, ?, ?, ?, ?)",
                (result.evaluation_id, result.opportunity_id, int(result.executed),
                 _iso(now or utcnow()), result.model_dump_json()),
            )

    def writebacks_for_opportunity(self, opportunity_id: str) -> list[WritebackResult]:
        """Recorded write-back results for an opportunity, oldest first."""
        with self._reading() as conn:
            rows = conn.execute(
                "SELECT result_json FROM writebacks WHERE opportunity_id = ? "
                "ORDER BY created_at, rowid",
                (opportunity_id,),
            ).fetchall()
        return [WritebackResult.model_validate_json(row[0]) for row in rows]

    def get_note_id(self, opportunity_id: str) -> str | None:
        with self._reading() as conn:
            row = conn.execute("SELECT note_id FROM lever_notes WHERE opportunity_id = ?",
                               (opportunity_id,)).fetchone()
        return row[0] if row else None

    def set_note_id(self, opportunity_id: str, note_id: str, *, now: datetime) -> None:
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO lever_notes (opportunity_id, note_id, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT (opportunity_id) DO UPDATE SET note_id = excluded.note_id, "
                "updated_at = excluded.updated_at",
                (opportunity_id, note_id, _iso(now)),
            )

    # ---------------------------------------------------------------------- webhooks

    def remember_webhook_token(self, token: str, *, now: datetime, ttl_seconds: int) -> bool:
        """True if the token is new (and records it); False if seen within the TTL.
        Expired tokens are pruned opportunistically."""
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        cutoff = _iso(now - timedelta(seconds=ttl_seconds))
        digest = _token_digest(token)
        with self._transaction() as conn:
            conn.execute("DELETE FROM webhook_tokens WHERE seen_at < ?", (cutoff,))
            if conn.execute("SELECT 1 FROM webhook_tokens WHERE token = ?",
                            (digest,)).fetchone():
                return False
            conn.execute("INSERT INTO webhook_tokens (token, seen_at) VALUES (?, ?)",
                         (digest, _iso(now)))
            return True

    # ------------------------------------------------------------- deletions & holds

    def record_deletion(self, opportunity_id: str, *, source: str,
                        deleted_at: datetime) -> None:
        """Tombstone (idempotent; the first recorded deletion wins). Never purged."""
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO deletions (opportunity_id, deleted_at, source) VALUES (?, ?, ?) "
                "ON CONFLICT (opportunity_id) DO NOTHING",
                (opportunity_id, _iso(deleted_at), source),
            )

    def is_deleted(self, opportunity_id: str) -> bool:
        with self._reading() as conn:
            return conn.execute("SELECT 1 FROM deletions WHERE opportunity_id = ?",
                                (opportunity_id,)).fetchone() is not None

    def purge_candidate(self, opportunity_id: str, *, posting_id: str | None = None) -> PurgeResult:
        """Delete PII rows, write-back records and the note mapping for the opportunity
        unless it (or ``posting_id``, or any posting it has PII rows for) is on legal
        hold. Audit rows are kept (pseudonymous). The WAL is checkpointed afterwards so
        the deleted rows' page images leave the files too."""
        with self._transaction() as conn:
            audit_kept = conn.execute("SELECT COUNT(*) FROM audit_log WHERE opportunity_id = ?",
                                      (opportunity_id,)).fetchone()[0]
            if self._held(conn, opportunity_id=opportunity_id, posting_id=posting_id):
                logger.info("purge of opportunity %s skipped: legal hold", opportunity_id)
                return PurgeResult(opportunity_id=opportunity_id, on_hold=True,
                                   audit_rows_kept=audit_kept)
            deleted = conn.execute("DELETE FROM pii_evaluations WHERE opportunity_id = ?",
                                   (opportunity_id,)).rowcount
            conn.execute("DELETE FROM writebacks WHERE opportunity_id = ?", (opportunity_id,))
            conn.execute("DELETE FROM lever_notes WHERE opportunity_id = ?", (opportunity_id,))
        self._checkpoint()
        logger.info("purged opportunity %s: %d PII row(s) deleted, %d audit row(s) kept",
                    opportunity_id, deleted, audit_kept)
        return PurgeResult(opportunity_id=opportunity_id, on_hold=False,
                           pii_rows_deleted=deleted, audit_rows_kept=audit_kept)

    def purge_tombstoned(self) -> list[PurgeResult]:
        """``purge_candidate`` for every deleted (tombstoned) opportunity that still has
        PII rows, write-back records or a note mapping, oldest deletion first: completes
        the deletions a legal hold suspended once the hold is gone. Opportunities still
        on hold come back with ``on_hold=True`` and are left as they are."""
        with self._reading() as conn:
            pending = [row[0] for row in conn.execute(_TOMBSTONED_SQL)]
        results = [self.purge_candidate(opportunity_id) for opportunity_id in pending]
        if results:
            logger.info("deleted candidates with data left: %d purged, %d kept on legal hold",
                        sum(1 for r in results if not r.on_hold),
                        sum(1 for r in results if r.on_hold))
        return results

    def set_legal_hold(self, *, scope: HoldScope, scope_id: str, reason: str, set_by: str,
                       now: datetime) -> None:
        """Set (or replace the reason of) a hold. Holds suspend every purge."""
        self._check_scope(scope, scope_id)
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO legal_holds (scope, scope_id, reason, set_by, set_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT (scope, scope_id) DO UPDATE SET "
                "reason = excluded.reason, set_by = excluded.set_by, set_at = excluded.set_at",
                (scope, scope_id, reason, set_by, _iso(now)),
            )
        logger.info("legal hold set on %s %s", scope, scope_id)

    def clear_legal_hold(self, *, scope: HoldScope, scope_id: str,
                         purge_deleted: bool = True) -> bool:
        """Release a hold; False when none was set. The deletions it suspended are then
        completed (``purge_tombstoned``), unless ``purge_deleted`` is False because the
        caller runs that itself, e.g. to report what was purged."""
        self._check_scope(scope, scope_id)
        with self._transaction() as conn:
            cleared = conn.execute("DELETE FROM legal_holds WHERE scope = ? AND scope_id = ?",
                                   (scope, scope_id)).rowcount > 0
        if cleared:
            logger.info("legal hold cleared on %s %s", scope, scope_id)
            if purge_deleted:
                self.purge_tombstoned()
        return cleared

    def is_on_hold(self, *, opportunity_id: str | None = None,
                   posting_id: str | None = None) -> bool:
        """True if the opportunity, the posting, or any posting the opportunity has
        evaluations for is on hold."""
        if opportunity_id is None and posting_id is None:
            raise ValueError("pass opportunity_id and/or posting_id")
        with self._reading() as conn:
            return self._held(conn, opportunity_id=opportunity_id, posting_id=posting_id)

    def purge_expired(self, *, now: datetime, retention_days: int) -> int:
        """Delete audit + PII rows older than retention unless on hold. Returns the number
        of evaluations removed. The only code path that deletes audit rows.

        Rows exactly ``retention_days`` old are kept. Write-back records of removed
        evaluations go too once they are themselves older than the retention period (a
        newer one is kept until a later purge); deletion tombstones are never purged."""
        if retention_days < 1:
            raise ValueError("retention_days must be at least 1")
        cutoff = _iso(now - timedelta(days=retention_days))
        with self._transaction() as conn:
            ids = [(row[0],) for row in conn.execute(_EXPIRED_SQL, {"cutoff": cutoff})]
            conn.executemany("DELETE FROM pii_evaluations WHERE evaluation_id = ?", ids)
            conn.executemany("DELETE FROM audit_log WHERE evaluation_id = ?", ids)
            conn.execute(_EXPIRED_WRITEBACKS_SQL, {"cutoff": cutoff})
        self._checkpoint()
        logger.info("retention purge (older than %s): %d evaluation(s) removed", cutoff,
                    len(ids))
        return len(ids)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                finally:
                    self._conn.close()

    # --------------------------------------------------------------------- helpers

    def _checkpoint(self) -> None:
        """Fold the WAL into the database file and truncate it, so page images written
        before a purge (still holding the deleted rows) do not outlive it. When another
        connection is mid-transaction the checkpoint reports busy and the next one, or
        ``close``, finishes the job."""
        with self._lock:
            self._connection().execute("PRAGMA wal_checkpoint(TRUNCATE)")

    @staticmethod
    def _check_scope(scope: str, scope_id: str) -> None:
        if scope not in _HOLD_SCOPES:
            raise ValueError(f"scope must be one of {_HOLD_SCOPES}, not {scope!r}")
        if not scope_id:
            raise ValueError("scope_id must not be empty")

    @staticmethod
    def _held(conn: sqlite3.Connection, *, opportunity_id: str | None,
              posting_id: str | None) -> bool:
        row = conn.execute(_HELD_SQL, {"opp": opportunity_id, "posting": posting_id})
        return bool(row.fetchone()[0])
