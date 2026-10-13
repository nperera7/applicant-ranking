"""Schema v2: the audit log records the jurisdiction decision, and the cache never mixes
EVALUATION-mode results with real (DRY_RUN/LIVE) runs."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from applicant_ranking.domain import Evaluation, RunMode, Status
from applicant_ranking.store import _SCHEMA_V1, SqliteStore


def test_audit_row_records_policy_decision(
        store: SqliteStore, make_evaluation: Callable[..., Evaluation]) -> None:
    gated = make_evaluation(opportunity_id="opp-a")
    local = make_evaluation(opportunity_id="opp-b")
    store.record_evaluation(gated, cache_key=None, policy_regimes=["IL_HRA", "CA_FEHA"],
                            policy_allowed=True)
    store.record_evaluation(local, cache_key=None)
    rows = {r["opportunity_id"]: r for r in store.audit_rows()}
    assert rows["opp-a"]["policy_regimes"] == ["IL_HRA", "CA_FEHA"]
    assert rows["opp-a"]["policy_allowed"] is True
    assert rows["opp-b"]["policy_regimes"] is None
    assert rows["opp-b"]["policy_allowed"] is None


def test_find_cached_filters_by_mode(
        store: SqliteStore, make_evaluation: Callable[..., Evaluation]) -> None:
    real = make_evaluation(mode=RunMode.DRY_RUN, status=Status.SCORED)
    offline = make_evaluation(mode=RunMode.EVALUATION, status=Status.SCORED)  # newer
    store.record_evaluation(real, cache_key="k")
    store.record_evaluation(offline, cache_key="k")

    assert store.find_cached("k").evaluation_id == offline.evaluation_id
    assert store.find_cached("k", evaluation_mode=True).evaluation_id == offline.evaluation_id
    # The older real-run result is still found for a real run.
    assert store.find_cached("k", evaluation_mode=False).evaluation_id == real.evaluation_id


def test_find_cached_mode_filter_misses_when_only_other_kind_exists(
        store: SqliteStore, make_evaluation: Callable[..., Evaluation]) -> None:
    store.record_evaluation(make_evaluation(mode=RunMode.EVALUATION, status=Status.SCORED),
                            cache_key="k")
    assert store.find_cached("k", evaluation_mode=False) is None


def test_v1_database_is_migrated_in_place(
        tmp_path: Path, make_evaluation: Callable[..., Evaluation]) -> None:
    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, "
                 "applied_at TEXT NOT NULL)")
    conn.executescript(_SCHEMA_V1)
    conn.execute("INSERT INTO schema_migrations VALUES (1, '2026-10-01T00:00:00+00:00')")
    conn.commit()
    conn.close()

    with SqliteStore(path) as store:
        assert store.schema_version() == 2
        store.record_evaluation(make_evaluation(), cache_key=None, policy_regimes=["IL_HRA"],
                                policy_allowed=True)
        assert store.audit_rows()[0]["policy_regimes"] == ["IL_HRA"]
