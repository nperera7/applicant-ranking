"""Async client for the Lever Data API v1 (API-key / HTTP Basic auth).

Facts this relies on (verified against hire.lever.co/developer/documentation, 2026-10-08):
- Auth: HTTP Basic, API key as username, blank password, sent pre-emptively
  (Lever's 401 carries no WWW-Authenticate challenge).
- Lists: ``{"data": [...], "next": "<opaque>", "hasNext": bool}``; pass ``next`` back as
  ``offset``; ``limit`` 1-100. Tolerate a missing ``hasNext``/``next`` (= end) and a bare
  JSON array body.
- Rate limit: token bucket per key, 10 rps steady / 20 burst; 429 on overload; 503 during
  maintenance. Retry-After is undocumented: honour it if present, never depend on it.
- Errors: JSON ``{"code", "message"}`` (body may also be non-dict, e.g. ``null``).
- 301 from the API means the wrong regional base URL. Never follow redirects on API
  calls: httpx drops Authorization cross-origin and turns POST into GET.
- Resume bytes: GET /opportunities/{opp}/resumes/{id}/download (422 = Lever could not
  process the file). ``file.downloadUrl`` is the web app; never use it. Content type and
  redirect behaviour are undocumented: follow at most one redirect for downloads only,
  to https, without auth on the cross-origin hop (httpx's default stripping).
- ``include=`` RESTRICTS the response to the named fields; never send it. Use ``expand``.

RANK-ONLY GUARANTEE: this client deliberately has no method that changes an
opportunity's stage, archive state or any field other than tags and notes. ``execute``
refuses any write whose path is not on ``WRITE_PATH_ALLOWLIST``. A unit test asserts both.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
    Sequence,
)
from contextlib import aclosing
from typing import Any, Self, cast
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ValidationError

from applicant_ranking.config import Settings
from applicant_ranking.domain import PlannedRequest
from applicant_ranking.lever.models import (
    ArchiveReason,
    DeletedApplication,
    DeletedOpportunity,
    DownloadedFile,
    Opportunity,
    Posting,
    Resume,
    Stage,
    User,
)
from applicant_ranking.lever.ratelimit import AsyncTokenBucket

logger = logging.getLogger(__name__)

# Only these write endpoints may ever be called (POST/PUT). Anything else raises.
WRITE_PATH_ALLOWLIST: tuple[re.Pattern[str], ...] = (
    re.compile(r"^/opportunities/[^/]+/addTags$"),
    re.compile(r"^/opportunities/[^/]+/removeTags$"),
    re.compile(r"^/opportunities/[^/]+/notes$"),
    re.compile(r"^/opportunities/[^/]+/notes/[^/]+$"),
)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
# A write answered with one of these was not processed, so sending it again is safe.
WRITE_RETRYABLE_STATUS = frozenset({429, 503})
REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})
MAX_PAGE_LIMIT = 100
# GET /applications/deleted accepts windows of at most 30 days (start inclusive, end exclusive).
DELETED_WINDOW_MS = 30 * 24 * 60 * 60 * 1000

USER_AGENT = "applicant-ranking (Egen internal)"

# Values Lever accepts for the ``confidentiality`` filter on list endpoints. Unspecified
# means non-confidential only, so a confidential posting's applicants are invisible.
CONFIDENTIALITY_VALUES: frozenset[str] = frozenset({"confidential", "non-confidential", "all"})
CONFIDENTIAL = "confidential"
# Without distributionChannel, GET /postings?state=published returns only postings on the
# public job site; both channels are needed to see internal-only published postings.
DEFAULT_DISTRIBUTION_CHANNELS: tuple[str, ...] = ("public", "internal")

# The method each allowlisted path may be written with (same order as the allowlist).
_WRITE_RULES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    zip(WRITE_PATH_ALLOWLIST, ("POST", "POST", "POST", "PUT"), strict=True)
)
# Lever ids are UUIDs or short slugs ("lead-new", "imp-..."). Anything else is refused
# before it can be interpolated into a path (ids may come from unsigned webhook data).
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.:~-]+$")
# A cursor made only of these is already URL-safe (possibly percent-encoded, as in the
# file_actions example) and is sent verbatim; anything else is percent-encoded once.
_VERBATIM_CURSOR_RE = re.compile(r"^(?:[A-Za-z0-9\-._~!$'()*,;:@]|%[0-9A-Fa-f]{2})+$")
_ERROR_BODY_LIMIT = 64 * 1024
_DOWNLOAD_HEADERS = {"Accept": "*/*", "Accept-Encoding": "identity"}


class LeverError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, code: str | None = None,
                 path: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.path = path


class LeverAuthError(LeverError):
    """401/403: bad key, or the key lacks permission for this endpoint."""


class LeverNotFound(LeverError):
    """404."""


class LeverUnprocessable(LeverError):
    """422 on a file download: Lever could not process the file."""


class LeverRedirectError(LeverError):
    """301/302 on an API call: probably the wrong regional base URL (api.eu.lever.co).

    Also raised for any 3xx on an API call, and for a download redirect that is not https
    or is followed by a second redirect."""


class LeverWriteNotAllowed(LeverError):
    """A write outside WRITE_PATH_ALLOWLIST was attempted (programming error)."""


class LeverInvalidRequest(LeverError, ValueError):
    """An id or path that is not safe to send (raised before any I/O)."""


class LeverConnectionError(LeverError):
    """The request failed below HTTP (connect/read error) and was not, or no longer, retried."""


class LeverTimeout(LeverConnectionError):
    """The request timed out. A timed-out write may still have been applied by Lever."""


class LeverFileTooLarge(LeverError):
    """A download exceeded ``max_bytes``; reading stopped at the limit."""


class _RetryableStatus(Exception):
    """Internal signal: a response that may be retried. Never escapes the client."""

    def __init__(self, error: LeverError, retry_after: float | None) -> None:
        super().__init__(str(error))
        self.error = error
        self.retry_after = retry_after


class _RedactQueryFilter(logging.Filter):
    """httpx logs every request URL at INFO. Query strings can carry signed-download
    signatures and personal data (``email=``), so they are dropped from those records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple) and any(
                isinstance(arg, httpx.URL) and arg.query for arg in record.args):
            record.args = tuple(
                arg.copy_with(query=None) if isinstance(arg, httpx.URL) and arg.query else arg
                for arg in record.args
            )
        return True


_HTTPX_LOG_FILTER = _RedactQueryFilter()


def _install_httpx_log_filter() -> None:
    httpx_logger = logging.getLogger("httpx")
    if _HTTPX_LOG_FILTER not in httpx_logger.filters:
        httpx_logger.addFilter(_HTTPX_LOG_FILTER)


def _full_jitter(ceiling: float) -> float:
    return random.uniform(0.0, ceiling)  # noqa: S311 - backoff jitter, not security


class LeverClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.lever.co/v1",
        rate_per_second: float = 8.0,
        burst: int = 15,
        timeout_seconds: float = 30.0,
        max_retries: int = 5,
        perform_as: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        bucket: AsyncTokenBucket | None = None,
        backoff_base_seconds: float = 1.0,
        backoff_cap_seconds: float = 60.0,
        sleep: Callable[[float], Awaitable[object]] | None = None,
        jitter: Callable[[float], float] | None = None,
    ) -> None:
        """``transport`` lets tests inject httpx.MockTransport/respx. Backoff values are
        parameters so tests can make retries instant.

        ``sleep`` (async) and ``jitter`` (maps a backoff ceiling to the delay actually
        used; default uniform in [0, ceiling], i.e. full jitter) are injectable for tests.
        """
        if not api_key:
            raise ValueError("api_key is required")
        base = base_url.rstrip("/")
        if not base.startswith("https://"):
            raise ValueError("base_url must be https")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if backoff_base_seconds < 0 or backoff_cap_seconds < 0:
            raise ValueError("backoff values must be >= 0")
        self._max_retries = max_retries
        self._perform_as = perform_as or None
        self._backoff_base = backoff_base_seconds
        self._backoff_cap = backoff_cap_seconds
        self._sleep = sleep or asyncio.sleep
        self._jitter = jitter or _full_jitter
        self._bucket = bucket or AsyncTokenBucket(rate_per_second, burst)
        # Postings this client fetched and found confidential: their opportunities are
        # listed with confidentiality=all (see list_opportunities).
        self._confidential_postings: set[str] = set()
        _install_httpx_log_filter()
        self._http = httpx.AsyncClient(
            base_url=base,
            auth=httpx.BasicAuth(api_key, ""),
            follow_redirects=False,
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
            headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        )

    @classmethod
    def from_settings(cls, settings: Settings,
                      transport: httpx.AsyncBaseTransport | None = None) -> Self:
        return cls(
            api_key=settings.require_lever_key(),
            base_url=settings.lever_base_url,
            rate_per_second=settings.lever_rate_per_second,
            burst=settings.lever_burst,
            timeout_seconds=settings.lever_timeout_seconds,
            max_retries=settings.lever_max_retries,
            perform_as=settings.lever_perform_as,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ low level

    async def request_json(self, method: str, path: str, *, params: dict[str, Any] | None = None,
                           json_body: dict[str, Any] | None = None,
                           retry_on_timeout: bool = True) -> Any:
        """One API call with rate limiting, retries (429/5xx/connect errors; timeouts only
        when ``retry_on_timeout``), exponential backoff with full jitter, and error mapping.
        Returns the decoded JSON body (may be dict, list or None). Writes pass
        ``retry_on_timeout=False``: a timed-out POST may have been applied.

        Writes (POST/PUT) are only sent to ``WRITE_PATH_ALLOWLIST`` paths, are retried
        only on 429/503 and connect errors, and never after a timeout whatever
        ``retry_on_timeout`` says. ``Retry-After`` (seconds) replaces the computed backoff,
        capped at ``backoff_cap_seconds``.
        """
        return await self._request_json(method, path, params=params, json_body=json_body,
                                        retry_on_timeout=retry_on_timeout)

    async def paginate(self, path: str, params: dict[str, Any] | None = None,
                       *, limit: int = 100) -> AsyncIterator[dict[str, Any]]:
        """Yield items across pages (offset=next while hasNext)."""
        if not 1 <= limit <= MAX_PAGE_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_PAGE_LIMIT}")
        query = {k: v for k, v in (params or {}).items() if k not in ("limit", "offset")}
        query["limit"] = limit
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            body = await self._request_json(
                "GET", path, params=query, json_body=None, retry_on_timeout=True,
                raw_params={"offset": cursor} if cursor is not None else None,
            )
            if isinstance(body, list):  # some examples are a bare array: a single page
                for item in body:
                    if isinstance(item, dict):
                        yield item
                return
            if not isinstance(body, dict):
                return
            data = body.get("data")
            for item in data if isinstance(data, list) else []:
                if isinstance(item, dict):
                    yield item
            nxt = body.get("next")
            if body.get("hasNext") is not True or not isinstance(nxt, str) or not nxt:
                return
            if nxt in seen:
                raise LeverError(f"Lever GET {path} returned a pagination cursor twice",
                                 path=path)
            seen.add(nxt)
            cursor = nxt

    # --------------------------------------------------------------------- reads

    async def get_posting(self, posting_id: str) -> Posting:
        path = f"/postings/{_id(posting_id, 'posting')}"
        posting = _parse_one(Posting, await self.request_json("GET", path), path)
        if (posting.confidentiality or "").strip().lower() == CONFIDENTIAL:
            self._confidential_postings.add(posting.id)
        return posting

    async def list_postings(
        self,
        *,
        state: str | None = None,
        distribution_channels: Sequence[str] | None = DEFAULT_DISTRIBUTION_CHANNELS,
        confidentiality: str | None = None,
    ) -> AsyncIterator[Posting]:
        """GET /postings[?state=..][&distributionChannel=..&distributionChannel=..]
        [&confidentiality=..]. ``distribution_channels`` is sent (as a repeated parameter)
        only for published postings (``state`` None or "published"), where Lever would
        otherwise return public-site postings only. ``confidentiality`` (one of
        ``CONFIDENTIALITY_VALUES``) needs a key with confidential access; unspecified means
        non-confidential postings only."""
        params: dict[str, Any] = {}
        if state:
            params["state"] = state
        if distribution_channels and state in (None, "published"):
            params["distributionChannel"] = list(distribution_channels)
        params["confidentiality"] = _confidentiality_param(confidentiality)
        async for posting in self._list(Posting, "/postings", params):
            yield posting

    async def list_opportunities(
        self,
        *,
        posting_id: str,
        archived: bool | None = False,
        created_at_start_ms: int | None = None,
        confidentiality: str | None = None,
    ) -> AsyncIterator[Opportunity]:
        """GET /opportunities?posting_id=..&expand=applications[&archived=..][&created_at_start=..]
        [&confidentiality=..]. ``archived=None`` returns both active and archived.

        Lever lists non-confidential opportunities unless ``confidentiality`` says otherwise
        (``CONFIDENTIALITY_VALUES``). When it is not given and ``get_posting`` has already
        shown this posting to be confidential, "all" is sent: that read proved the key has
        confidential access, so the posting's applicants are not silently invisible.

        A record that fails validation is yielded as ``Opportunity.unreadable`` when it
        carries a usable id, so the applicant still gets a (skipped) row."""
        if not posting_id:
            raise ValueError("posting_id is required")
        params: dict[str, Any] = {"posting_id": posting_id, "expand": "applications"}
        if archived is not None:
            params["archived"] = "true" if archived else "false"
        if created_at_start_ms is not None:
            params["created_at_start"] = int(created_at_start_ms)
        if confidentiality is None and posting_id in self._confidential_postings:
            confidentiality = "all"
        params["confidentiality"] = _confidentiality_param(confidentiality)
        count = 0
        async for opportunity in self._list(Opportunity, "/opportunities", params):
            count += 1
            yield opportunity
        if count == 0:
            logger.warning(
                "Lever listed no %sopportunities for posting %s (confidentiality=%s): if the "
                "posting has applicants, check the archived filter and whether the API key "
                "has confidential access", "" if archived is None else
                ("archived " if archived else "active "), posting_id,
                confidentiality or "unspecified (non-confidential)",
            )

    async def get_opportunity(self, opportunity_id: str) -> Opportunity:
        """GET /opportunities/{id}?expand=applications."""
        path = f"/opportunities/{_id(opportunity_id, 'opportunity')}"
        body = await self.request_json("GET", path, params={"expand": "applications"})
        return _parse_one(Opportunity, body, path)

    async def list_resumes(self, opportunity_id: str) -> list[Resume]:
        path = f"/opportunities/{_id(opportunity_id, 'opportunity')}/resumes"
        return [resume async for resume in self._list(Resume, path)]

    async def download_resume(self, opportunity_id: str, resume_id: str, *,
                              max_bytes: int) -> DownloadedFile:
        """Raises LeverUnprocessable on 422 and LeverError if the body exceeds max_bytes
        (stream and stop reading; don't buffer an unbounded body).

        The size limit raises ``LeverFileTooLarge`` (a LeverError). A 3xx is followed at
        most once, only to https, as a fresh request without the Authorization header.
        """
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")
        path = (f"/opportunities/{_id(opportunity_id, 'opportunity')}"
                f"/resumes/{_id(resume_id, 'resume')}/download")
        _check_request("GET", path)

        async def attempt() -> DownloadedFile:
            request = self._http.build_request("GET", path, headers=_DOWNLOAD_HEADERS)
            response = await self._http.send(request, stream=True)
            try:
                if response.status_code in REDIRECT_STATUS:
                    target = _redirect_target(response, path)
                    await response.aclose()
                    return await self._download_redirect(target, path, max_bytes)
                return await self._read_download(response, path, max_bytes)
            finally:
                await response.aclose()

        return await self._with_retries("GET", path, attempt, is_write=False,
                                        retry_on_timeout=True)

    async def list_stages(self) -> list[Stage]:
        return [stage async for stage in self._list(Stage, "/stages")]

    async def list_archive_reasons(self) -> list[ArchiveReason]:
        return [reason async for reason in self._list(ArchiveReason, "/archive_reasons")]

    async def find_user_by_email(self, email: str) -> User | None:
        """GET /users?email=.. (Lever matches the canonicalised address). Returns the user
        whose email equals ``email`` case-insensitively, else None."""
        email = email.strip()
        if not email:
            raise ValueError("email is required")
        wanted = email.casefold()
        async with aclosing(self._list(User, "/users", {"email": email})) as users:
            async for user in users:
                if user.email and user.email.strip().casefold() == wanted:
                    return user
        return None

    async def list_deleted_applications(self, *, start_ms: int,
                                        end_ms: int) -> AsyncIterator[DeletedApplication]:
        """GET /applications/deleted. Lever caps each window at 30 days (start inclusive,
        end exclusive), so split longer ranges into consecutive <=30-day windows."""
        window_start = int(start_ms)
        end = int(end_ms)
        while window_start < end:
            window_end = min(window_start + DELETED_WINDOW_MS, end)
            params = {"deleted_at_start": window_start, "deleted_at_end": window_end}
            async for deleted in self._list(DeletedApplication, "/applications/deleted", params):
                yield deleted
            window_start = window_end

    async def list_deleted_opportunities(self, *, start_ms: int,
                                         end_ms: int) -> AsyncIterator[DeletedOpportunity]:
        """GET /opportunities/deleted?deleted_at_start=..&deleted_at_end=..: the candidates
        Lever deleted in the range (both bounds inclusive, no window cap). Each row's
        ``id`` is the deleted opportunity's id."""
        params = {"deleted_at_start": int(start_ms), "deleted_at_end": int(end_ms)}
        async for deleted in self._list(DeletedOpportunity, "/opportunities/deleted", params):
            yield deleted

    # -------------------------------------------------------------------- writes
    # perform_as is optional for tags and notes; sent only when configured.

    async def add_tags(self, opportunity_id: str, tags: list[str]) -> None:
        await self._change_tags("addTags", opportunity_id, tags)

    async def remove_tags(self, opportunity_id: str, tags: list[str]) -> None:
        await self._change_tags("removeTags", opportunity_id, tags)

    async def create_note(self, opportunity_id: str, *, value: str, secret: bool = False,
                          notify_followers: bool = False) -> str:
        """POST /opportunities/{id}/notes -> 201 {"data": {"noteId": ...}}. Returns noteId.
        Never sets ``score`` (Lever's 1-4 thumbs scale belongs to human reviewers)."""
        if not value:
            raise ValueError("note value is required")
        path = f"/opportunities/{_id(opportunity_id, 'opportunity')}/notes"
        body = await self.request_json(
            "POST", path, params=self._write_params(),
            json_body={"value": value, "secret": secret, "notifyFollowers": notify_followers},
            retry_on_timeout=False,
        )
        note_id = _note_id(body)
        if note_id is None:
            raise LeverError(f"Lever POST {path} returned no noteId", path=path)
        return note_id

    async def update_note(self, opportunity_id: str, note_id: str, *, value: str,
                          secret: bool = False) -> str:
        """PUT /opportunities/{id}/notes/{noteId} with {"values": [{"value": ...}],
        "secret": ...}; only notes created through the API can be updated. Returns noteId.
        Raises LeverNotFound if the note no longer exists."""
        if not value:
            raise ValueError("note value is required")
        path = (f"/opportunities/{_id(opportunity_id, 'opportunity')}"
                f"/notes/{_id(note_id, 'note')}")
        body = await self.request_json(
            "PUT", path, params=self._write_params(),
            json_body={"values": [{"value": value}], "secret": secret},
            retry_on_timeout=False,
        )
        return _note_id(body) or note_id

    async def execute(self, request: PlannedRequest) -> Any:
        """Send a planned request. Non-GET requests must match WRITE_PATH_ALLOWLIST or
        LeverWriteNotAllowed is raised before anything is sent."""
        return await self.request_json(
            request.method, request.path, params=dict(request.params) or None,
            json_body=request.json_body, retry_on_timeout=request.method == "GET",
        )

    # ----------------------------------------------------------------- internals

    def _write_params(self) -> dict[str, str] | None:
        return {"perform_as": self._perform_as} if self._perform_as else None

    async def _change_tags(self, action: str, opportunity_id: str, tags: list[str]) -> None:
        if isinstance(tags, str) or not all(isinstance(t, str) and t.strip() for t in tags):
            raise ValueError("tags must be a list of non-empty strings")
        unique = list(dict.fromkeys(tags))
        if not unique:
            return
        path = f"/opportunities/{_id(opportunity_id, 'opportunity')}/{action}"
        await self.request_json("POST", path, params=self._write_params(),
                                json_body={"tags": unique}, retry_on_timeout=False)

    async def _list[M: BaseModel](self, model: type[M], path: str,
                                  params: dict[str, Any] | None = None) -> AsyncGenerator[M]:
        """Validated items, page by page. An opportunity that fails validation but has a
        usable id becomes ``Opportunity.unreadable`` (never dropped); any other invalid
        item is skipped. Field values may be personal data, so the log carries the shape
        of the problem only."""
        async for item in self.paginate(path, params):
            try:
                yield model.model_validate(item)
            except ValidationError as exc:
                placeholder = _placeholder(model, item)
                if placeholder is None:
                    logger.warning("Skipping a %s item from GET %s that failed validation "
                                   "(%d errors)", model.__name__, path, exc.error_count())
                    continue
                logger.warning("Opportunity %s from GET %s failed validation (%d errors); "
                               "kept as an unreadable record so the applicant is not dropped",
                               placeholder.id, path, exc.error_count())
                yield placeholder

    async def _request_json(self, method: str, path: str, *, params: Mapping[str, Any] | None,
                            json_body: dict[str, Any] | None, retry_on_timeout: bool,
                            raw_params: Mapping[str, str] | None = None) -> Any:
        method = _check_request(method, path)
        is_write = method != "GET"
        url = _build_url(path, params, raw_params)

        async def attempt() -> Any:
            response = await self._http.request(method, url, json=json_body)
            if not response.is_success:
                raise _failure(response.status_code, response.headers, response.content,
                               method, path, is_write=is_write)
            if response.status_code == 204 or not response.content:
                return None
            try:
                return response.json()
            except ValueError as exc:
                raise LeverError(f"Lever {method} {path} returned invalid JSON",
                                 status=response.status_code, path=path) from exc

        if is_write:
            logger.info("Lever %s %s", method, path)
        return await self._with_retries(method, path, attempt, is_write=is_write,
                                        retry_on_timeout=retry_on_timeout and not is_write)

    async def _with_retries[T](self, method: str, path: str,
                               attempt: Callable[[], Awaitable[T]], *,
                               is_write: bool, retry_on_timeout: bool) -> T:
        retries = 0
        while True:
            await self._bucket.acquire()
            try:
                return await attempt()
            except _RetryableStatus as exc:
                if retries >= self._max_retries:
                    raise exc.error from None
                delay = self._backoff(retries, exc.retry_after)
                reason = f"HTTP {exc.error.status}"
            except httpx.TimeoutException as exc:
                if not retry_on_timeout or retries >= self._max_retries:
                    raise LeverTimeout(f"Lever {method} {path} timed out ({type(exc).__name__})",
                                       path=path) from exc
                delay = self._backoff(retries, None)
                reason = type(exc).__name__
            except httpx.TransportError as exc:
                # A connect error means nothing was sent. Other network errors may hit a
                # request Lever already received: only reads are safe to repeat.
                safe = isinstance(exc, httpx.ConnectError) or (
                    not is_write and isinstance(exc, httpx.NetworkError | httpx.RemoteProtocolError)
                )
                if not safe or retries >= self._max_retries:
                    raise LeverConnectionError(
                        f"Lever {method} {path} failed ({type(exc).__name__})", path=path
                    ) from exc
                delay = self._backoff(retries, None)
                reason = type(exc).__name__
            except httpx.HTTPError as exc:
                raise LeverError(f"Lever {method} {path} failed ({type(exc).__name__})",
                                 path=path) from exc
            retries += 1
            logger.warning("Lever %s %s: %s; retry %d of %d in %.2fs", method, path, reason,
                           retries, self._max_retries, delay)
            await self._sleep(delay)

    def _backoff(self, retry_index: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self._backoff_cap)
        ceiling = min(self._backoff_cap, self._backoff_base * 2.0 ** min(retry_index, 32))
        return min(max(0.0, self._jitter(ceiling)), ceiling)

    async def _download_redirect(self, target: httpx.URL, path: str,
                                 max_bytes: int) -> DownloadedFile:
        await self._bucket.acquire()
        request = self._http.build_request("GET", target, headers=_DOWNLOAD_HEADERS)
        request.headers.pop("Authorization", None)
        # auth=None: no credentials on this hop, wherever it points.
        response = await self._http.send(request, stream=True, auth=None)
        try:
            if response.status_code in REDIRECT_STATUS:
                raise LeverRedirectError(
                    f"Lever GET {path}: download redirected more than once",
                    status=response.status_code, path=path,
                )
            return await self._read_download(response, path, max_bytes)
        finally:
            await response.aclose()

    async def _read_download(self, response: httpx.Response, path: str,
                             max_bytes: int) -> DownloadedFile:
        if not response.is_success:
            body, _ = await _read_limited(response, _ERROR_BODY_LIMIT)
            raise _failure(response.status_code, response.headers, body, "GET", path,
                           is_write=False)
        declared = response.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > max_bytes:
            raise LeverFileTooLarge(
                f"Lever GET {path}: file is {declared} bytes, over the {max_bytes}-byte limit",
                status=response.status_code, path=path,
            )
        content, truncated = await _read_limited(response, max_bytes)
        if truncated:
            raise LeverFileTooLarge(
                f"Lever GET {path}: file exceeds the {max_bytes}-byte limit",
                status=response.status_code, path=path,
            )
        return DownloadedFile(content=content, content_type=response.headers.get("content-type"))


# --------------------------------------------------------------------------- helpers


def _id(value: str, what: str) -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        shown = repr(value)[:60]
        raise LeverInvalidRequest(f"invalid Lever {what} id: {shown}")
    return value


def _confidentiality_param(value: str | None) -> str | None:
    """Validate a ``confidentiality`` filter value (None = let Lever default)."""
    if value is None:
        return None
    if value not in CONFIDENTIALITY_VALUES:
        raise ValueError(f"confidentiality must be one of "
                         f"{', '.join(sorted(CONFIDENTIALITY_VALUES))}, not {value!r}")
    return value


def _placeholder[M: BaseModel](model: type[M], item: dict[str, Any]) -> M | None:
    """The stand-in for an invalid list item: only opportunities with a usable id get one
    (an applicant must not vanish from the ranking); everything else is skipped."""
    raw_id = item.get("id")
    if model is Opportunity and isinstance(raw_id, str) and _ID_RE.match(raw_id):
        return cast(M, Opportunity.unreadable(raw_id))
    return None


def _check_request(method: str, path: str) -> str:
    """Normalise the method and refuse unsafe paths and non-allowlisted writes (no I/O)."""
    method = method.upper()
    safe_path = path.startswith("/") and all(
        _SEGMENT_RE.match(segment) and segment not in (".", "..")
        for segment in path[1:].split("/")
    )
    if method == "GET":
        if not safe_path:
            raise LeverInvalidRequest(f"unsafe Lever path: {path[:80]!r}", path=path)
        return method
    if safe_path and any(rule.match(path) and method == allowed for rule, allowed in _WRITE_RULES):
        return method
    raise LeverWriteNotAllowed(
        f"{method} {path[:80]!r} is not an allowed Lever write (rank-only client)", path=path
    )


def _build_url(path: str, params: Mapping[str, Any] | None,
               raw_params: Mapping[str, str] | None) -> httpx.URL:
    parts: list[str] = []
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    if clean:
        parts.append(str(httpx.QueryParams(clean)))
    for key, value in (raw_params or {}).items():
        encoded = value if _VERBATIM_CURSOR_RE.match(value) else quote(value, safe="")
        parts.append(f"{quote(key, safe='')}={encoded}")
    if not parts:
        return httpx.URL(path)
    return httpx.URL(path, query="&".join(parts).encode("ascii"))


def _failure(status: int, headers: httpx.Headers, body: bytes, method: str, path: str, *,
             is_write: bool) -> Exception:
    """The exception for a non-2xx response: a ``_RetryableStatus`` wrapper when the
    status may be retried for this kind of request, else the mapped LeverError."""
    error = _error_for(status, headers, body, method, path)
    retryable = WRITE_RETRYABLE_STATUS if is_write else RETRYABLE_STATUS
    if status in retryable:
        return _RetryableStatus(error, _retry_after(headers))
    return error


def _error_for(status: int, headers: httpx.Headers, body: bytes, method: str,
               path: str) -> LeverError:
    code, message = _parse_error_body(body)
    if status in REDIRECT_STATUS:
        location = _without_query(headers.get("location", "")) or "<no Location>"
        return LeverRedirectError(
            f"Lever {method} {path} returned HTTP {status} redirecting to {location}: "
            "probably the wrong regional base URL (AR_LEVER_BASE_URL; EU accounts use "
            "https://api.eu.lever.co/v1). API redirects are never followed.",
            status=status, code=code, path=path,
        )
    cls: type[LeverError] = {
        401: LeverAuthError, 403: LeverAuthError, 404: LeverNotFound, 422: LeverUnprocessable,
    }.get(status, LeverError)
    detail = ": ".join(part for part in (code, message) if part)
    text = f"Lever {method} {path} failed with HTTP {status}"
    return cls(f"{text} ({detail})" if detail else text, status=status, code=code, path=path)


def _parse_error_body(body: bytes) -> tuple[str | None, str | None]:
    try:
        decoded = json.loads(body) if body else None
    except ValueError:
        return None, None
    if not isinstance(decoded, dict):
        return None, None
    code = decoded.get("code")
    message = decoded.get("message")
    return (
        str(code)[:100] if isinstance(code, str | int) and not isinstance(code, bool) else None,
        message[:300] if isinstance(message, str) else None,
    )


def _retry_after(headers: httpx.Headers) -> float | None:
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None  # an HTTP-date; Lever doesn't document either form
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _without_query(url: str) -> str:
    return url.split("?", 1)[0].split("#", 1)[0][:200]


def _redirect_target(response: httpx.Response, path: str) -> httpx.URL:
    location = response.headers.get("location")
    if not location:
        raise LeverRedirectError(f"Lever GET {path}: redirect without a Location",
                                 status=response.status_code, path=path)
    target = response.request.url.join(location)
    if target.scheme != "https":
        raise LeverRedirectError(
            f"Lever GET {path}: refusing a non-https download redirect ({target.scheme})",
            status=response.status_code, path=path,
        )
    return target


async def _read_limited(response: httpx.Response, limit: int) -> tuple[bytes, bool]:
    """Read at most ``limit`` bytes. Returns (content, True) if the body was longer."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        if total + len(chunk) > limit:
            chunks.append(chunk[: limit - total])
            return b"".join(chunks), True
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks), False


def _parse_one[M: BaseModel](model: type[M], body: Any, path: str) -> M:
    data = body.get("data") if isinstance(body, dict) else None
    if data is None and isinstance(body, dict) and "data" not in body and "id" in body:
        data = body
    if not isinstance(data, dict):
        raise LeverError(f"Lever GET {path} returned an unexpected body", path=path)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise LeverError(f"Lever GET {path} returned an invalid {model.__name__} "
                         f"({exc.error_count()} errors)", path=path) from None


def _note_id(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    data = body.get("data")
    note_id = data.get("noteId") if isinstance(data, dict) else body.get("noteId")
    return str(note_id) if isinstance(note_id, str | int) and note_id != "" else None
