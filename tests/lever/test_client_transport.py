"""Auth, error mapping, redirects, retries and backoff of the low-level request path."""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from applicant_ranking.config import ConfigError, Settings
from applicant_ranking.lever.client import (
    LeverAuthError,
    LeverClient,
    LeverConnectionError,
    LeverError,
    LeverNotFound,
    LeverRedirectError,
    LeverTimeout,
    LeverUnprocessable,
)

OK = httpx.Response(200, json={"data": {"id": "x"}})


def _error(status: int, body: Any = None, *,
           headers: dict[str, str] | None = None) -> httpx.Response:
    if isinstance(body, bytes):
        return httpx.Response(status, content=body, headers=headers)
    return httpx.Response(status, json=body, headers=headers)


# --------------------------------------------------------------------------- auth/config


async def test_basic_auth_is_sent_on_the_first_request(client: LeverClient, lever: Any,
                                                       api_key: str) -> None:
    lever.on("GET", "/stages", httpx.Response(200, json={"data": []}))
    await client.list_stages()
    assert len(lever.requests) == 1  # no challenge round trip
    request = lever.requests[0]
    expected = base64.b64encode(f"{api_key}:".encode()).decode()
    assert request.headers["authorization"] == f"Basic {expected}"
    assert request.headers["accept"] == "application/json"
    assert request.url.scheme == "https"
    assert str(request.url).startswith("https://api.lever.co/v1/stages")


async def test_from_settings(make_settings: Callable[..., Settings], lever: Any) -> None:
    settings = make_settings(lever_base_url="https://api.eu.lever.co/v1/",
                             lever_perform_as="user-1", lever_max_retries=0)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": []})

    async with LeverClient.from_settings(settings, transport=httpx.MockTransport(handler)) as c:
        await c.add_tags("opp-1", ["AI Rank: Tier A"])
    assert seen[0].url.host == "api.eu.lever.co"
    assert seen[0].url.path == "/v1/opportunities/opp-1/addTags"
    assert seen[0].url.params["perform_as"] == "user-1"
    assert seen[0].headers["authorization"] == (
        "Basic " + base64.b64encode(b"test-lever-key:").decode()
    )


def test_from_settings_requires_a_key(make_settings: Callable[..., Settings]) -> None:
    with pytest.raises(ConfigError):
        LeverClient.from_settings(make_settings(lever_api_key=None))


@pytest.mark.parametrize("kwargs", [
    {"api_key": ""},
    {"api_key": "k", "base_url": "http://api.lever.co/v1"},
    {"api_key": "k", "max_retries": -1},
    {"api_key": "k", "backoff_base_seconds": -1},
])
def test_constructor_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        LeverClient(**kwargs)


async def test_context_manager_closes_the_http_client(lever: Any) -> None:
    async with LeverClient(api_key="k", transport=httpx.MockTransport(lever.handler)) as c:
        pass
    assert c._http.is_closed


# --------------------------------------------------------------------------- error mapping


@pytest.mark.parametrize(("status", "exc_type"), [
    (401, LeverAuthError), (403, LeverAuthError), (404, LeverNotFound),
    (422, LeverUnprocessable), (400, LeverError),
])
async def test_status_mapping(client: LeverClient, lever: Any, status: int,
                              exc_type: type[LeverError]) -> None:
    lever.on("GET", "/stages/s1", _error(status, {"code": "SomeCode", "message": "went wrong"}))
    with pytest.raises(exc_type) as excinfo:
        await client.request_json("GET", "/stages/s1")
    err = excinfo.value
    assert type(err) is exc_type
    assert (err.status, err.code, err.path) == (status, "SomeCode", "/stages/s1")
    assert "went wrong" in str(err)
    assert lever.count("GET", "/stages/s1") == 1  # not retried


@pytest.mark.parametrize("body", [None, [], "just text", {"code": None, "message": 5},
                                  b"<html>oops</html>", b""])
async def test_error_body_is_parsed_defensively(client: LeverClient, lever: Any,
                                                body: Any) -> None:
    lever.on("GET", "/stages", _error(400, body))
    with pytest.raises(LeverError) as excinfo:
        await client.request_json("GET", "/stages")
    assert excinfo.value.status == 400
    assert excinfo.value.code is None


async def test_long_error_message_is_clipped(client: LeverClient, lever: Any) -> None:
    lever.on("GET", "/stages", _error(400, {"code": "Bad", "message": "x" * 5000}))
    with pytest.raises(LeverError) as excinfo:
        await client.request_json("GET", "/stages")
    assert len(str(excinfo.value)) < 500


@pytest.mark.parametrize("status", [301, 302, 307, 308])
async def test_api_redirect_is_an_error_naming_the_location(client: LeverClient, lever: Any,
                                                            status: int) -> None:
    lever.on("GET", "/postings", httpx.Response(
        status, json=None, headers={"Location": "https://api.lever.co/v1/postings?x=secret"}))
    with pytest.raises(LeverRedirectError) as excinfo:
        await client.request_json("GET", "/postings")
    message = str(excinfo.value)
    assert "https://api.lever.co/v1/postings" in message
    assert "secret" not in message
    assert "base URL" in message
    assert excinfo.value.status == status
    assert len(lever.requests) == 1  # never followed


async def test_redirect_on_a_write_is_not_followed(client: LeverClient, lever: Any) -> None:
    lever.on("POST", "/opportunities/o1/addTags",
             httpx.Response(301, headers={"Location": "https://api.lever.co/v1/x"}))
    with pytest.raises(LeverRedirectError):
        await client.add_tags("o1", ["t"])
    assert [r.method for r in lever.requests] == ["POST"]


async def test_success_bodies(client: LeverClient, lever: Any) -> None:
    lever.on("GET", "/a", httpx.Response(200, json=[1, 2]))
    lever.on("GET", "/b", httpx.Response(204))
    lever.on("GET", "/c", httpx.Response(200, json=None))
    lever.on("GET", "/d", httpx.Response(200, content=b"{not json"))
    assert await client.request_json("GET", "/a") == [1, 2]
    assert await client.request_json("GET", "/b") is None
    assert await client.request_json("GET", "/c") is None
    with pytest.raises(LeverError, match="invalid JSON"):
        await client.request_json("GET", "/d")


async def test_params_are_encoded_and_none_values_dropped(client: LeverClient,
                                                          lever: Any) -> None:
    lever.on("GET", "/users", httpx.Response(200, json={"data": []}))
    await client.request_json("GET", "/users",
                              params={"email": "a+b@example.com", "skip": None,
                                      "tag": ["x y", "z"]})
    params = lever.requests[0].url.params
    assert params["email"] == "a+b@example.com"
    assert "skip" not in params
    assert params.get_list("tag") == ["x y", "z"]


# --------------------------------------------------------------------------------- retries


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_reads_retry_transient_statuses(client: LeverClient, lever: Any, bucket: Any,
                                              backoff: Any, status: int) -> None:
    lever.on("GET", "/stages", _error(status, {"code": "Busy"}), _error(status), OK)
    assert await client.request_json("GET", "/stages") == {"data": {"id": "x"}}
    assert lever.count("GET", "/stages") == 3
    assert bucket.acquired == 3  # every attempt takes a token
    assert backoff.delays == [0.5, 1.0]


async def test_backoff_is_exponential_and_capped(make_client: Callable[..., LeverClient],
                                                 lever: Any, backoff: Any) -> None:
    client = make_client(max_retries=6, backoff_base_seconds=1.0, backoff_cap_seconds=10.0)
    lever.on("GET", "/stages", _error(503))
    with pytest.raises(LeverError) as excinfo:
        await client.request_json("GET", "/stages")
    assert excinfo.value.status == 503
    assert lever.count("GET", "/stages") == 7
    assert backoff.delays == [1.0, 2.0, 4.0, 8.0, 10.0, 10.0]


async def test_default_jitter_is_full_jitter(make_client: Callable[..., LeverClient],
                                             lever: Any, backoff: Any) -> None:
    client = make_client(jitter=None, max_retries=5, backoff_base_seconds=1.0,
                         backoff_cap_seconds=4.0)
    lever.on("GET", "/stages", _error(500))
    with pytest.raises(LeverError):
        await client.request_json("GET", "/stages")
    ceilings = [1.0, 2.0, 4.0, 4.0, 4.0]
    assert len(backoff.delays) == len(ceilings)
    assert all(0.0 <= d <= c for d, c in zip(backoff.delays, ceilings, strict=True))


async def test_no_retries_when_max_retries_is_zero(make_client: Callable[..., LeverClient],
                                                   lever: Any, backoff: Any) -> None:
    client = make_client(max_retries=0)
    lever.on("GET", "/stages", _error(429))
    with pytest.raises(LeverError) as excinfo:
        await client.request_json("GET", "/stages")
    assert excinfo.value.status == 429
    assert lever.count("GET", "/stages") == 1
    assert backoff.delays == []


@pytest.mark.parametrize(("header", "expected"), [
    ("7", 7.0), ("1.5", 1.5), ("0", 0.0),
    ("1000", 8.0),  # capped at backoff_cap_seconds
    ("Wed, 21 Oct 2015 07:28:00 GMT", 0.5), ("-3", 0.5), ("nan", 0.5),  # ignored
])
async def test_retry_after_is_honoured(client: LeverClient, lever: Any, backoff: Any,
                                       header: str, expected: float) -> None:
    lever.on("GET", "/stages", _error(429, headers={"Retry-After": header}), OK)
    await client.request_json("GET", "/stages")
    assert backoff.delays == [expected]


async def test_connect_errors_are_retried(client: LeverClient, lever: Any) -> None:
    lever.on("GET", "/stages", httpx.ConnectError("refused"), OK)
    assert await client.request_json("GET", "/stages") == {"data": {"id": "x"}}
    assert lever.count("GET", "/stages") == 2


async def test_read_errors_are_retried_for_reads(client: LeverClient, lever: Any) -> None:
    lever.on("GET", "/stages", httpx.ReadError("reset"),
             httpx.RemoteProtocolError("bad"), OK)
    await client.request_json("GET", "/stages")
    assert lever.count("GET", "/stages") == 3


async def test_persistent_connect_error_raises_lever_connection_error(
        client: LeverClient, lever: Any) -> None:
    lever.on("GET", "/stages", httpx.ConnectError("refused"))
    with pytest.raises(LeverConnectionError) as excinfo:
        await client.request_json("GET", "/stages")
    assert isinstance(excinfo.value.__cause__, httpx.ConnectError)
    assert lever.count("GET", "/stages") == 4  # 1 + max_retries


async def test_timeouts_are_retried_for_reads_by_default(client: LeverClient,
                                                         lever: Any) -> None:
    lever.on("GET", "/stages", httpx.ReadTimeout("slow"), OK)
    await client.request_json("GET", "/stages")
    assert lever.count("GET", "/stages") == 2


async def test_timeouts_are_not_retried_when_disabled(client: LeverClient, lever: Any) -> None:
    lever.on("GET", "/stages", httpx.ReadTimeout("slow"), OK)
    with pytest.raises(LeverTimeout):
        await client.request_json("GET", "/stages", retry_on_timeout=False)
    assert lever.count("GET", "/stages") == 1


async def test_other_httpx_errors_are_wrapped(client: LeverClient, lever: Any) -> None:
    lever.on("GET", "/stages", httpx.DecodingError("bad gzip"))
    with pytest.raises(LeverError):
        await client.request_json("GET", "/stages")
    assert lever.count("GET", "/stages") == 1


# ------------------------------------------------------------------------- write retries


@pytest.mark.parametrize("status", [429, 503])
async def test_writes_retry_only_unprocessed_statuses(client: LeverClient, lever: Any,
                                                      status: int) -> None:
    lever.on("POST", "/opportunities/o1/addTags", _error(status), httpx.Response(200, json={}))
    await client.add_tags("o1", ["t"])
    assert lever.count("POST", "/opportunities/o1/addTags") == 2


@pytest.mark.parametrize("status", [500, 502, 504])
async def test_writes_are_not_retried_on_other_5xx(client: LeverClient, lever: Any,
                                                   status: int) -> None:
    lever.on("POST", "/opportunities/o1/notes", _error(status), httpx.Response(201, json={}))
    with pytest.raises(LeverError) as excinfo:
        await client.create_note("o1", value="hello")
    assert excinfo.value.status == status
    assert lever.count("POST", "/opportunities/o1/notes") == 1


@pytest.mark.parametrize("exc", [httpx.ReadTimeout("slow"), httpx.WriteTimeout("slow"),
                                 httpx.ConnectTimeout("slow")])
async def test_writes_are_never_retried_after_a_timeout(client: LeverClient, lever: Any,
                                                        exc: Exception) -> None:
    lever.on("POST", "/opportunities/o1/notes", exc, httpx.Response(201, json={}))
    with pytest.raises(LeverTimeout):
        # Even an explicit retry_on_timeout=True can't make a write repeat after a timeout.
        await client.request_json("POST", "/opportunities/o1/notes",
                                  json_body={"value": "v"}, retry_on_timeout=True)
    assert lever.count("POST", "/opportunities/o1/notes") == 1


async def test_writes_retry_connect_errors(client: LeverClient, lever: Any) -> None:
    lever.on("PUT", "/opportunities/o1/notes/n1", httpx.ConnectError("refused"),
             httpx.Response(200, json={"data": {"noteId": "n1"}}))
    assert await client.update_note("o1", "n1", value="v") == "n1"
    assert lever.count("PUT", "/opportunities/o1/notes/n1") == 2


async def test_writes_do_not_retry_read_errors(client: LeverClient, lever: Any) -> None:
    lever.on("POST", "/opportunities/o1/addTags", httpx.ReadError("reset"),
             httpx.Response(200, json={}))
    with pytest.raises(LeverConnectionError):
        await client.add_tags("o1", ["t"])
    assert lever.count("POST", "/opportunities/o1/addTags") == 1


# ------------------------------------------------------------------------------- logging


async def test_logs_never_contain_the_key_or_auth_header(client: LeverClient, lever: Any,
                                                         api_key: str,
                                                         caplog: pytest.LogCaptureFixture
                                                         ) -> None:
    caplog.set_level(logging.DEBUG)
    lever.on("GET", "/stages", _error(429), _error(401, {"code": "UnauthorizedError"}))
    with pytest.raises(LeverAuthError) as excinfo:
        await client.list_stages()
    assert "retry 1 of 3" in caplog.text
    encoded = base64.b64encode(f"{api_key}:".encode()).decode()
    for text in (caplog.text, str(excinfo.value), repr(excinfo.value)):
        assert api_key not in text
        assert encoded not in text
