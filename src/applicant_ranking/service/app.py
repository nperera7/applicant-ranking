"""FastAPI webhook receiver.

Routes:
- ``GET /healthz`` -> {"status": "ok", "model": ..., "write_enabled": ...} (no secrets).
- ``POST /webhooks/lever``: parse the JSON envelope (invalid JSON -> 400); look up the
  signing token for ``envelope.event`` (none configured -> 401 if the event is handled,
  else 200 ignore); ``verify_envelope`` (reject -> 401 with the reason); replay check via
  ``store.remember_webhook_token`` (seen -> 200 {"status": "duplicate"}, Lever must not
  retry); then:
  * applicationCreated with an opportunityId -> enqueue ``ranker.evaluate_opportunity``
    (mode: LIVE if settings.write_enabled else DRY_RUN) keyed by the envelope's
    idempotency key -> 202 {"status": "queued"}. While Lever is still processing the
    resume the ranker returns a SKIPPED row (``SKIP_RESUME_PENDING``) and the job is
    re-queued after ``RETRY_DELAYS[attempt]`` seconds under ``<key>:retry<n>``; the last
    attempt runs with ``defer_pending=False`` and ends in the normal ERROR/"Not scored"
    path if the resume still is not processed;
  * candidateDeleted -> enqueue ``ranker.handle_candidate_deleted`` -> 202 (the ranker
    confirms the deletion with GET /opportunities/{id} before any tombstone or purge,
    because the opportunity id comes from the unsigned ``data``);
  * anything else (including the verify-connection ping with data {}) -> 200
    {"status": "ignored"}.
  Handler work stays under ~1 s: no Lever or LLM calls inline.
The app owns its Components via the lifespan (built from settings unless injected) and
closes them on shutdown. Logs never include resume text or tokens.

Also:
- Bodies over ``MAX_BODY_BYTES`` -> 413; JSON that is not an object, or an envelope
  missing ``event``/``triggeredAt``/``token`` -> 400 (field names only in the reply).
- Replay tokens are remembered for twice the freshness window plus a margin, because
  ``verify_envelope`` accepts ``triggeredAt`` up to the window on either side of now.
- An opportunity id from the (unsigned) ``data`` that is not id-shaped -> 200 ignored.
- A job the queue refuses: still queued/running under the same idempotency key -> 200
  {"status": "duplicate"}; queue full or stopping -> 202 {"status": "dropped"} (the
  token is already remembered, so a Lever retry could not help; re-running ``rank``
  catches up).
- Only components the lifespan built are closed by it; injected ones belong to the
  caller. ``app.state.clock`` (UTC datetimes) drives the freshness check.
- No OpenAPI schema or docs routes are served.
"""

from __future__ import annotations

import functools
import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from applicant_ranking import wiring
from applicant_ranking.config import Settings
from applicant_ranking.domain import RunMode, utcnow
from applicant_ranking.lever.webhooks import (
    HANDLED_EVENTS,
    WebhookEnvelope,
    WebhookRejected,
    verify_envelope,
)
from applicant_ranking.pipeline import SKIP_RESUME_PENDING
from applicant_ranking.service.jobs import JobFn, JobQueue
from applicant_ranking.wiring import Components

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 256 * 1024
REPLAY_TTL_MARGIN_SECONDS = 60
DEFAULT_DRAIN_SECONDS = 8.0  # Cloud Run allows 10 s between SIGTERM and SIGKILL
# Seconds before each re-run of an applicationCreated job whose resume Lever was still
# processing (Lever parses most files within minutes). After the last delay the job runs
# once more without deferring.
RETRY_DELAYS: tuple[float, ...] = (120.0, 600.0, 1800.0)
# Same shape the Lever client accepts for ids it puts into a path.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")


def _build_components(settings: Settings) -> Components:
    return wiring.build_components(settings)


async def _close_components(components: Components) -> None:
    await wiring.close_components(components)


def create_app(settings: Settings | None = None, *,
               components: Components | None = None,
               clock: Callable[[], datetime] = utcnow,
               queue_workers: int = 2,
               queue_maxsize: int = 1000,
               drain_timeout: float = DEFAULT_DRAIN_SECONDS) -> FastAPI:
    if settings is None:
        settings = components.settings if components is not None else Settings()
    resolved: Settings = settings

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = components is None
        comps = components if components is not None else _build_components(resolved)
        try:
            queue = JobQueue(workers=queue_workers, maxsize=queue_maxsize)
            app.state.components = comps
            app.state.queue = queue
            await queue.start()
            try:
                yield
            finally:
                await queue.stop(drain_timeout=drain_timeout)
        finally:
            if owned:
                await _close_components(comps)

    app = FastAPI(title="Applicant Ranking webhooks", lifespan=lifespan, docs_url=None,
                  redoc_url=None, openapi_url=None)
    app.state.settings = resolved
    app.state.clock = clock

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok", "model": resolved.gemini_model,
                "write_enabled": resolved.write_enabled}

    @app.post("/webhooks/lever")
    async def lever_webhook(request: Request) -> JSONResponse:
        return await _handle_lever_webhook(request)

    return app


def _reply(status_code: int, status: str, detail: str | None = None) -> JSONResponse:
    body: dict[str, str] = {"status": status}
    if detail:
        body["detail"] = detail
    return JSONResponse(body, status_code=status_code)


async def _read_body(request: Request, limit: int) -> bytes | None:
    """The request body, or None when it is larger than ``limit`` bytes."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        return None
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            return None
    return bytes(body)


def _now(clock: Callable[[], datetime]) -> tuple[datetime, int]:
    """The clock's time (naive = UTC) and the same instant in epoch milliseconds."""
    now = clock()
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now, int(now.timestamp() * 1000)


async def _handle_lever_webhook(request: Request) -> JSONResponse:
    state = request.app.state
    settings: Settings = state.settings

    body = await _read_body(request, MAX_BODY_BYTES)
    if body is None:
        return _reply(413, "rejected", "payload too large")
    try:
        payload = json.loads(body)
    except (ValueError, RecursionError):
        return _reply(400, "rejected", "invalid JSON")
    if not isinstance(payload, dict):
        return _reply(400, "rejected", "expected a JSON object")
    try:
        envelope = WebhookEnvelope.model_validate(payload)
    except ValidationError as exc:
        fields = sorted({str(err["loc"][0]) for err in exc.errors() if err["loc"]})
        return _reply(400, "rejected", "invalid webhook envelope: " + ", ".join(fields))

    event = envelope.event
    signing_token = settings.webhook_token(event)
    if signing_token is None:
        if event in HANDLED_EVENTS:
            logger.warning("webhook %s rejected: no signing token configured", event)
            return _reply(401, "rejected", "no signing token configured for this event")
        logger.info("webhook event %r ignored (not handled, no signing token)", event[:64])
        return _reply(200, "ignored")

    now, now_ms = _now(state.clock)
    try:
        verify_envelope(envelope, signing_token=signing_token, now_ms=now_ms,
                        max_age_seconds=settings.webhook_max_age_seconds)
    except WebhookRejected as exc:
        logger.warning("webhook %s rejected: %s", event, exc.reason)
        return _reply(401, "rejected", exc.reason)

    components: Components = state.components
    ttl = 2 * settings.webhook_max_age_seconds + REPLAY_TTL_MARGIN_SECONDS
    fresh = await run_in_threadpool(components.store.remember_webhook_token, envelope.token,
                                    now=now, ttl_seconds=ttl)
    if not fresh:
        logger.info("webhook %s replayed: token already seen", event)
        return _reply(200, "duplicate")
    return _dispatch(envelope, components, state.queue, settings)


def _dispatch(envelope: WebhookEnvelope, components: Components, queue: JobQueue,
              settings: Settings) -> JSONResponse:
    event = envelope.event
    if event not in HANDLED_EVENTS:
        logger.info("webhook %s ignored (not handled)", event)
        return _reply(200, "ignored")
    opportunity_id = envelope.opportunity_id
    if not opportunity_id:
        logger.info("webhook %s ignored: no opportunity id in data", event)
        return _reply(200, "ignored", "no opportunityId")
    if not _ID_RE.match(opportunity_id):
        logger.warning("webhook %s ignored: malformed opportunity id", event)
        return _reply(200, "ignored", "malformed opportunityId")

    ranker = components.ranker
    key = envelope.idempotency_key
    job: JobFn
    if event == "applicationCreated":
        mode = RunMode.LIVE if settings.write_enabled else RunMode.DRY_RUN
        job = _evaluation_job(ranker, queue, key, opportunity_id, mode)
    else:  # candidateDeleted
        job = functools.partial(ranker.handle_candidate_deleted, opportunity_id,
                                source="webhook")

    if queue.is_pending(key):
        logger.info("webhook %s for %s: job %s already queued or running", event,
                    opportunity_id, key)
        return _reply(200, "duplicate")
    if not queue.submit(key, job):
        logger.warning("webhook %s for %s dropped: job queue full or stopping", event,
                       opportunity_id)
        return _reply(202, "dropped", "job queue full or stopping")
    logger.info("webhook %s for %s queued as %s", event, opportunity_id, key)
    return _reply(202, "queued")


def _evaluation_job(ranker: Any, queue: JobQueue, key: str, opportunity_id: str,
                    mode: RunMode) -> JobFn:
    """The applicationCreated job. ``evaluate_opportunity`` defers while retries remain;
    a row skipped with ``SKIP_RESUME_PENDING`` is re-queued after ``RETRY_DELAYS[attempt]``
    seconds as ``<key>:retry<attempt + 1>`` (its own idempotency key). The last attempt
    runs with ``defer_pending=False``."""

    async def run(attempt: int = 0) -> Any:
        defer = attempt < len(RETRY_DELAYS)
        row = await ranker.evaluate_opportunity(opportunity_id, mode=mode, defer_pending=defer)
        if defer and row.evaluation.review_reasons == [SKIP_RESUME_PENDING]:
            delay = RETRY_DELAYS[attempt]
            retry_key = f"{key}:retry{attempt + 1}"
            if queue.submit_later(retry_key, functools.partial(run, attempt + 1), delay):
                logger.info("opportunity %s: resume still processing in Lever; retry %d of %d "
                            "scheduled in %.0f s as %s", opportunity_id, attempt + 1,
                            len(RETRY_DELAYS), delay, retry_key)
            else:
                logger.warning("opportunity %s: resume still processing in Lever and the "
                               "retry could not be scheduled; the next rank run scores it",
                               opportunity_id)
        return row

    return run
