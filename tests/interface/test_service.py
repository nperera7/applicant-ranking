"""Webhook service: verification, replay protection, dispatch and lifespan."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from applicant_ranking.config import Settings
from applicant_ranking.domain import RunMode
from applicant_ranking.lever.webhooks import compute_signature
from applicant_ranking.service import app as app_module
from applicant_ranking.service.app import MAX_BODY_BYTES, create_app
from applicant_ranking.wiring import Components

from .conftest import NOW, SECRETS, WEBHOOK_TOKENS, FakeRanker

URL = "/webhooks/lever"
MAX_AGE = 600


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def now_ms(clock: Clock) -> int:
    return int(clock.now.timestamp() * 1000)


def envelope(event: str, data: dict[str, Any] | None = None, *, token: str | None = None,
             triggered_at: int | None = None, signing_token: str | None = None,
             event_id: str | None = None, signature: str | None = None) -> dict[str, Any]:
    token = token or secrets.token_hex(24)
    triggered_at = triggered_at if triggered_at is not None else int(NOW.timestamp() * 1000)
    signing = signing_token or WEBHOOK_TOKENS.get(event, "unconfigured")
    body: dict[str, Any] = {
        "event": event,
        "triggeredAt": triggered_at,
        "token": token,
        "signature": signature or compute_signature(token, triggered_at, signing),
        "data": {} if data is None else data,
    }
    if event_id is not None:
        body["id"] = event_id
    return body


def created(opportunity_id: str = "opp-1", **kwargs: Any) -> dict[str, Any]:
    return envelope("applicationCreated", {"opportunityId": opportunity_id,
                                           "applicationId": "app-1",
                                           "candidateId": opportunity_id}, **kwargs)


@dataclass
class Service:
    client: TestClient
    app: FastAPI
    ranker: FakeRanker
    components: Components
    clock: Clock

    def post(self, body: Any) -> Any:
        return self.client.post(URL, json=body)

    def drain(self) -> None:
        self.client.portal.call(self.app.state.queue.join)


@pytest.fixture
def start_service(iface_settings: Settings, make_components: Callable[..., Components],
                  ) -> Iterator[Callable[..., Service]]:
    clients: list[TestClient] = []

    def _start(settings: Settings | None = None, **app_kwargs: Any) -> Service:
        settings = settings or iface_settings
        ranker = FakeRanker()
        components = make_components(settings, ranker=ranker)
        clock = Clock(NOW)
        app = create_app(settings, components=components, clock=clock, **app_kwargs)
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return Service(client=client, app=app, ranker=ranker, components=components,
                       clock=clock)

    yield _start
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def service(start_service: Callable[..., Service]) -> Service:
    return start_service()


# ------------------------------------------------------------------------------ health


def test_healthz_reports_status_without_secrets(service: Service) -> None:
    response = service.client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "gemini-3.1-pro-preview",
                               "write_enabled": False}
    for secret in SECRETS:
        assert secret not in response.text


def test_no_docs_or_schema_routes(service: Service) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert service.client.get(path).status_code == 404


# -------------------------------------------------------------------------- dispatch


def test_application_created_is_queued_and_evaluated_once(service: Service) -> None:
    response = service.post(created("opp-42"))

    assert response.status_code == 202
    assert response.json() == {"status": "queued"}
    service.drain()
    assert service.ranker.called("evaluate_opportunity") == [
        (("opp-42",), {"mode": RunMode.DRY_RUN, "force": False, "defer_pending": True})]


def test_resume_still_processing_is_retried_then_scored(
        service: Service, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "RETRY_DELAYS", (0.0, 0.0, 0.0))
    service.ranker.pending_rows = 2  # deferred twice, then Lever has processed the file

    response = service.post(created("opp-7"))
    service.drain()

    assert response.status_code == 202
    calls = service.ranker.called("evaluate_opportunity")
    assert [(args[0], kwargs["defer_pending"]) for args, kwargs in calls] == [
        ("opp-7", True), ("opp-7", True), ("opp-7", True)]
    queue = service.app.state.queue
    assert queue.completed == 3 and queue.failed == 0 and queue.dropped == 0


def test_resume_still_processing_after_every_retry_is_scored_undeferred(
        service: Service, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "RETRY_DELAYS", (0.0, 0.0, 0.0))
    service.ranker.pending_rows = 100  # Lever never finishes

    assert service.post(created("opp-8")).status_code == 202
    service.drain()

    calls = service.ranker.called("evaluate_opportunity")
    # Three deferred attempts (one per delay), then one that takes the usual ERROR path.
    assert [kwargs["defer_pending"] for _, kwargs in calls] == [True, True, True, False]
    assert all(kwargs["mode"] is RunMode.DRY_RUN for _, kwargs in calls)
    assert service.app.state.queue.completed == 4


def test_write_enabled_runs_live(start_service: Callable[..., Service],
                                 iface_settings: Settings) -> None:
    service = start_service(iface_settings.model_copy(update={"write_enabled": True}))

    assert service.post(created("opp-1")).status_code == 202
    service.drain()

    [(_, kwargs)] = service.ranker.called("evaluate_opportunity")
    assert kwargs["mode"] == RunMode.LIVE
    assert service.client.get("/healthz").json()["write_enabled"] is True


def test_candidate_deleted_purges(service: Service) -> None:
    response = service.post(envelope("candidateDeleted", {"candidateId": "opp-9"}))

    assert response.status_code == 202
    service.drain()
    assert service.ranker.called("handle_candidate_deleted") == [
        (("opp-9",), {"source": "webhook"})]
    assert service.ranker.called("evaluate_opportunity") == []


def test_verify_connection_ping_is_ignored(service: Service) -> None:
    response = service.post(envelope("candidateHired", {}))

    assert response.status_code == 200
    assert response.json() == {"status": "ignored"}
    service.drain()
    assert service.ranker.calls == []


def test_unknown_event_without_token_is_ignored(service: Service) -> None:
    response = service.post(envelope("candidateStageChange", {"opportunityId": "opp-1"},
                                     signature="anything"))

    assert response.status_code == 200
    assert response.json() == {"status": "ignored"}
    assert service.ranker.calls == []


def test_unknown_event_with_token_is_verified_then_ignored(
        start_service: Callable[..., Service], iface_settings: Settings) -> None:
    tokens = {**WEBHOOK_TOKENS, "candidateStageChange": "whsec-stage"}
    service = start_service(iface_settings.model_copy(update={
        "lever_webhook_tokens": {event: SecretStr(t) for event, t in tokens.items()}}))

    bad = service.post(envelope("candidateStageChange", {"opportunityId": "opp-1"},
                                signing_token="wrong"))
    good = service.post(envelope("candidateStageChange", {"opportunityId": "opp-1"},
                                 signing_token="whsec-stage"))

    assert bad.status_code == 401
    assert good.status_code == 200
    assert good.json() == {"status": "ignored"}


@pytest.mark.parametrize("data", [{}, {"applicationId": "app-1"}])
def test_application_created_without_opportunity_is_ignored(service: Service,
                                                            data: dict[str, Any]) -> None:
    response = service.post(envelope("applicationCreated", data))

    assert response.status_code == 200
    assert response.json()["status"] == "ignored"
    assert service.ranker.calls == []


@pytest.mark.parametrize("opportunity_id", ["../../postings", "a b", "x" * 300, "-lead"])
def test_malformed_opportunity_id_is_ignored(service: Service, opportunity_id: str) -> None:
    response = service.post(created(opportunity_id))

    assert response.status_code == 200
    assert response.json()["status"] == "ignored"
    service.drain()
    assert service.ranker.calls == []


# ---------------------------------------------------------------------- verification


def test_bad_signature_is_rejected(service: Service) -> None:
    response = service.post(created(signing_token="not-the-secret"))

    assert response.status_code == 401
    assert response.json() == {"status": "rejected", "detail": "invalid signature"}
    service.drain()
    assert service.ranker.calls == []


def test_missing_signature_is_rejected(service: Service) -> None:
    body = created()
    del body["signature"]

    response = service.post(body)

    assert response.status_code == 401
    assert response.json()["detail"] == "missing signature"


def test_stale_envelope_is_rejected(service: Service) -> None:
    stale = now_ms(service.clock) - (MAX_AGE + 1) * 1000

    response = service.post(created(triggered_at=stale))

    assert response.status_code == 401
    assert "stale" in response.json()["detail"]
    assert service.ranker.calls == []


def test_future_envelope_is_rejected(service: Service) -> None:
    future = now_ms(service.clock) + (MAX_AGE + 1) * 1000

    response = service.post(created(triggered_at=future))

    assert response.status_code == 401
    assert "future" in response.json()["detail"]


def test_freshness_follows_the_injected_clock(service: Service) -> None:
    body = created()
    service.clock.now = NOW + timedelta(seconds=MAX_AGE + 5)

    response = service.post(body)

    assert response.status_code == 401
    service.app.state.clock = lambda: NOW  # app.state.clock can be swapped too
    assert service.post(body).status_code == 202


def test_handled_event_without_configured_token_is_rejected(
        start_service: Callable[..., Service], iface_settings: Settings) -> None:
    service = start_service(iface_settings.model_copy(update={"lever_webhook_tokens": {}}))

    response = service.post(created())

    assert response.status_code == 401
    assert response.json()["detail"] == "no signing token configured for this event"
    assert service.ranker.calls == []


# --------------------------------------------------------------------------- replays


def test_replayed_token_is_a_duplicate_and_not_requeued(service: Service) -> None:
    body = created("opp-1")

    first = service.post(body)
    service.drain()
    second = service.post(body)
    service.drain()

    assert first.status_code == 202
    assert second.status_code == 200
    assert second.json() == {"status": "duplicate"}
    assert len(service.ranker.called("evaluate_opportunity")) == 1


def test_rejected_envelope_does_not_consume_its_token(service: Service) -> None:
    token = secrets.token_hex(24)

    forged = service.post(created(token=token, signing_token="wrong"))
    genuine = service.post(created(token=token))

    assert forged.status_code == 401
    assert genuine.status_code == 202


def test_replay_window_covers_future_skew(service: Service) -> None:
    """A token first seen while ``triggeredAt`` was ahead of our clock must still be
    remembered when the same envelope is replayed at the far edge of the window."""
    ahead = now_ms(service.clock) + MAX_AGE * 1000
    body = created(triggered_at=ahead)

    assert service.post(body).status_code == 202
    service.clock.now = NOW + timedelta(seconds=2 * MAX_AGE)
    replay = service.post(body)

    assert replay.status_code == 200
    assert replay.json() == {"status": "duplicate"}


def test_same_event_id_is_not_queued_twice_while_running(service: Service) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def hold(_: str) -> None:
        started.set()
        await release.wait()

    service.ranker.on_evaluate = hold
    first = service.post(created("opp-1", event_id="evt-1"))
    service.client.portal.call(started.wait)
    second = service.post(created("opp-1", event_id="evt-1"))  # new token, same event id
    service.client.portal.call(release.set)
    service.drain()

    assert first.status_code == 202
    assert second.status_code == 200
    assert second.json() == {"status": "duplicate"}
    assert len(service.ranker.called("evaluate_opportunity")) == 1


def test_full_queue_drops_with_2xx(start_service: Callable[..., Service]) -> None:
    service = start_service(queue_workers=1, queue_maxsize=1)
    started = asyncio.Event()
    release = asyncio.Event()

    async def hold(_: str) -> None:
        started.set()
        await release.wait()

    service.ranker.on_evaluate = hold
    running = service.post(created("opp-1"))
    service.client.portal.call(started.wait)
    waiting = service.post(created("opp-2"))
    dropped = service.post(created("opp-3"))
    service.client.portal.call(release.set)
    service.drain()

    assert running.status_code == waiting.status_code == 202
    assert dropped.status_code == 202
    assert dropped.json()["status"] == "dropped"
    called = [args[0] for args, _ in service.ranker.called("evaluate_opportunity")]
    assert called == ["opp-1", "opp-2"]


def test_failing_job_does_not_stop_the_worker(service: Service) -> None:
    service.ranker.error = RuntimeError("Lever unavailable")
    assert service.post(created("opp-1")).status_code == 202
    service.drain()

    service.ranker.error = None
    assert service.post(created("opp-2")).status_code == 202
    service.drain()

    queue = service.app.state.queue
    assert queue.failed == 1 and queue.completed == 1
    called = [args[0] for args, _ in service.ranker.called("evaluate_opportunity")]
    assert called == ["opp-1", "opp-2"]


# --------------------------------------------------------------------- bad requests


@pytest.mark.parametrize(("content", "detail"), [
    (b"{not json", "invalid JSON"),
    (b"\xff\xfe\x00garbage", "invalid JSON"),
    (b"[1, 2]", "expected a JSON object"),
    (b"[" * 100_000 + b"]" * 100_000, "invalid JSON"),
])
def test_invalid_json_is_400(service: Service, content: bytes, detail: str) -> None:
    response = service.client.post(URL, content=content,
                                   headers={"content-type": "application/json"})

    assert response.status_code == 400
    assert response.json()["detail"] == detail


def test_incomplete_envelope_is_400(service: Service) -> None:
    body = created()
    del body["triggeredAt"]
    body["token"] = {"not": "a string"}

    response = service.post(body)

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid webhook envelope: token, triggeredAt"


def test_oversized_body_is_413(service: Service) -> None:
    body = created()
    body["data"]["padding"] = "x" * MAX_BODY_BYTES

    response = service.post(body)

    assert response.status_code == 413
    assert service.ranker.calls == []


def test_oversized_streamed_body_without_length_is_413(service: Service) -> None:
    def chunks() -> Iterator[bytes]:
        for _ in range(5):
            yield b" " * (MAX_BODY_BYTES // 4)

    response = service.client.post(URL, content=chunks(),
                                   headers={"content-type": "application/json"})

    assert response.status_code == 413


# ------------------------------------------------------------------------- logging


def test_logs_never_contain_tokens_or_signatures(service: Service,
                                                 caplog: pytest.LogCaptureFixture) -> None:
    bodies = [created("opp-1"), created("opp-2", signing_token="wrong"),
              created(triggered_at=0), envelope("candidateHired", {})]
    with caplog.at_level(logging.DEBUG):
        for body in bodies:
            service.post(body)
        service.post(bodies[0])  # replay
        service.drain()

    assert "webhook applicationCreated rejected: invalid signature" in caplog.text
    for body in bodies:
        assert body["token"] not in caplog.text
        assert body["signature"] not in caplog.text
    for secret in SECRETS:
        assert secret not in caplog.text


# ------------------------------------------------------------------------ lifespan


def test_injected_components_are_not_closed(start_service: Callable[..., Service]) -> None:
    service = start_service()
    service.client.__exit__(None, None, None)

    # The store is still usable: the caller owns injected components.
    assert service.components.store.is_deleted("opp-1") is False
    assert not service.app.state.queue.accepting


def test_lifespan_builds_and_closes_components(
        monkeypatch: pytest.MonkeyPatch, iface_settings: Settings,
        make_components: Callable[..., Components]) -> None:
    ranker = FakeRanker()
    events: list[str] = []

    def build(settings: Settings) -> Components:
        events.append("build")
        assert settings is iface_settings
        return make_components(settings, ranker=ranker)

    async def close(components: Components) -> None:
        events.append("close")

    monkeypatch.setattr(app_module, "_build_components", build)
    monkeypatch.setattr(app_module, "_close_components", close)
    app = create_app(iface_settings, clock=lambda: NOW)
    assert events == []  # nothing is built before startup

    with TestClient(app) as client:
        assert events == ["build"]
        assert client.post(URL, json=created("opp-5")).status_code == 202
        client.portal.call(app.state.queue.join)

    assert events == ["build", "close"]
    assert ranker.called("evaluate_opportunity")[0][0] == ("opp-5",)


def test_shutdown_drains_queued_jobs(start_service: Callable[..., Service]) -> None:
    service = start_service(queue_workers=1)
    for i in range(3):
        assert service.post(created(f"opp-{i}")).status_code == 202

    service.client.__exit__(None, None, None)

    assert len(service.ranker.called("evaluate_opportunity")) == 3


def test_settings_default_to_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
                                         iface_settings: Settings,
                                         make_components: Callable[..., Components]) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AR_GEMINI_MODEL", "gemini-from-env")
    monkeypatch.setattr(app_module, "_build_components",
                        lambda settings: make_components(iface_settings, ranker=FakeRanker()))
    monkeypatch.setattr(app_module, "_close_components", _noop_close)

    app = create_app()

    with TestClient(app) as client:
        assert client.get("/healthz").json()["model"] == "gemini-from-env"


async def _noop_close(components: Components) -> None:
    return None


def test_settings_taken_from_injected_components(make_components: Callable[..., Components],
                                                 iface_settings: Settings) -> None:
    components = make_components(iface_settings.model_copy(update={"gemini_model": "m-x"}),
                                 ranker=FakeRanker())

    app = create_app(components=components)

    with TestClient(app) as client:
        assert client.get("/healthz").json()["model"] == "m-x"


def test_json_bodies_are_only_read_once(service: Service) -> None:
    """Sanity check for the raw-body path: a valid request with extra envelope fields."""
    body = created("opp-1")
    body["extra"] = {"nested": [1, 2, 3]}

    response = service.client.post(URL, content=json.dumps(body).encode(),
                                   headers={"content-type": "application/json"})

    assert response.status_code == 202
