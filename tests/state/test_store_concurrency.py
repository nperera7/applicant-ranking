"""One connection shared by many threads (FastAPI workers + asyncio)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timedelta

from applicant_ranking.domain import Evaluation, WritebackResult
from applicant_ranking.store import SqliteStore

THREADS = 8
PER_THREAD = 25


def test_concurrent_writes_from_eight_threads(store: SqliteStore,
                                              make_evaluation: Callable[..., Evaluation],
                                              base_time: datetime) -> None:
    # Build evaluations up front: the factory's counter is not thread-safe.
    batches = [
        [make_evaluation(evaluation_id=f"t{t}-e{i}", opportunity_id=f"opp-{t}-{i % 5}",
                         posting_id=f"posting-{t % 2}") for i in range(PER_THREAD)]
        for t in range(THREADS)
    ]
    barrier = threading.Barrier(THREADS)
    errors: list[BaseException] = []
    fresh_tokens: list[bool] = []
    tokens_lock = threading.Lock()

    def work(t: int) -> None:
        try:
            barrier.wait()
            for i, evaluation in enumerate(batches[t]):
                store.record_evaluation(evaluation, cache_key=f"key-{t}-{i}")
                store.record_writeback(WritebackResult(
                    opportunity_id=evaluation.opportunity_id,
                    evaluation_id=evaluation.evaluation_id, executed=False))
                store.set_note_id(evaluation.opportunity_id, f"note-{t}-{i}", now=base_time)
                # Every thread races on the same shared tokens: exactly one wins each.
                fresh = store.remember_webhook_token(f"shared-{i}",
                                                     now=base_time + timedelta(seconds=i),
                                                     ttl_seconds=3600)
                with tokens_lock:
                    fresh_tokens.append(fresh)
                assert store.find_cached(f"key-{t}-{i}") is not None
                store.is_on_hold(opportunity_id=evaluation.opportunity_id)
        except BaseException as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(t,)) for t in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(store.audit_rows()) == THREADS * PER_THREAD
    assert store._conn.execute("SELECT COUNT(*) FROM pii_evaluations").fetchone()[0] == (
        THREADS * PER_THREAD)
    assert store._conn.execute("SELECT COUNT(*) FROM writebacks").fetchone()[0] == (
        THREADS * PER_THREAD)
    assert sum(fresh_tokens) == PER_THREAD
    assert len(store.evaluations_for_posting("posting-0")) == (THREADS // 2) * 5


def test_concurrent_duplicate_records_admit_exactly_one(
        store: SqliteStore, make_evaluation: Callable[..., Evaluation]) -> None:
    evaluation = make_evaluation(evaluation_id="same")
    barrier = threading.Barrier(THREADS)
    outcomes: list[str] = []
    lock = threading.Lock()

    def work() -> None:
        barrier.wait()
        try:
            store.record_evaluation(evaluation, cache_key="k")
            outcome = "ok"
        except Exception as exc:
            outcome = type(exc).__name__
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=work) for _ in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count("ok") == 1
    assert outcomes.count("DuplicateEvaluationError") == THREADS - 1
