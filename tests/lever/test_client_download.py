"""Resume downloads: streaming size cap, 422, and the single auth-less redirect."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from applicant_ranking.lever.client import (
    LeverClient,
    LeverError,
    LeverFileTooLarge,
    LeverRedirectError,
    LeverUnprocessable,
)

DOWNLOAD = "/opportunities/o1/resumes/r1/download"
PDF = b"%PDF-1.7\n" + b"x" * 991


class ChunkStream(httpx.AsyncByteStream):
    """A response body that records how many chunks were actually pulled."""

    def __init__(self, chunk: bytes, count: int) -> None:
        self.chunk = chunk
        self.count = count
        self.pulled = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for _ in range(self.count):
            self.pulled += 1
            yield self.chunk

    async def aclose(self) -> None:
        self.closed = True


async def test_download_returns_bytes_and_content_type(client: LeverClient, lever: Any) -> None:
    lever.on("GET", DOWNLOAD, httpx.Response(
        200, content=PDF, headers={"Content-Type": "application/pdf"}))
    downloaded = await client.download_resume("o1", "r1", max_bytes=10_000)
    assert downloaded.content == PDF
    assert downloaded.content_type == "application/pdf"
    request = lever.requests[0]
    assert request.headers["authorization"].startswith("Basic ")
    assert request.headers["accept-encoding"] == "identity"


async def test_body_exactly_at_the_limit_is_accepted(client: LeverClient, lever: Any) -> None:
    lever.on("GET", DOWNLOAD, httpx.Response(200, content=PDF))
    downloaded = await client.download_resume("o1", "r1", max_bytes=len(PDF))
    assert downloaded.content == PDF
    assert downloaded.content_type is None


async def test_streaming_stops_once_the_limit_is_exceeded(client: LeverClient,
                                                          lever: Any) -> None:
    stream = ChunkStream(b"a" * 1024, count=1000)  # ~1 MB, no Content-Length
    lever.on("GET", DOWNLOAD, lambda request: httpx.Response(200, stream=stream))
    with pytest.raises(LeverFileTooLarge) as excinfo:
        await client.download_resume("o1", "r1", max_bytes=4096)
    assert isinstance(excinfo.value, LeverError)
    assert stream.pulled == 5  # stopped right after crossing 4 KiB
    assert stream.closed
    assert len(lever.requests) == 1  # not retried


async def test_declared_length_over_the_limit_is_refused_unread(client: LeverClient,
                                                                lever: Any) -> None:
    stream = ChunkStream(b"a" * 1024, count=100)
    lever.on("GET", DOWNLOAD, lambda request: httpx.Response(
        200, stream=stream, headers={"Content-Length": str(100 * 1024)}))
    with pytest.raises(LeverFileTooLarge, match="102400 bytes"):
        await client.download_resume("o1", "r1", max_bytes=4096)
    assert stream.pulled == 0


async def test_422_is_unprocessable(client: LeverClient, lever: Any) -> None:
    lever.on("GET", DOWNLOAD, httpx.Response(
        422, json={"code": "UnprocessableEntity", "message": "File could not be processed"}))
    with pytest.raises(LeverUnprocessable) as excinfo:
        await client.download_resume("o1", "r1", max_bytes=10_000)
    assert excinfo.value.status == 422
    assert excinfo.value.code == "UnprocessableEntity"
    assert len(lever.requests) == 1


async def test_large_error_body_is_read_only_partially(client: LeverClient, lever: Any) -> None:
    stream = ChunkStream(b"e" * 1024, count=10_000)
    lever.on("GET", DOWNLOAD, lambda request: httpx.Response(404, stream=stream))
    with pytest.raises(LeverError) as excinfo:
        await client.download_resume("o1", "r1", max_bytes=10_000)
    assert excinfo.value.status == 404
    assert stream.pulled <= 65


async def test_transient_errors_are_retried(client: LeverClient, lever: Any) -> None:
    lever.on("GET", DOWNLOAD, httpx.Response(503), httpx.ReadTimeout("slow"),
             httpx.Response(200, content=PDF))
    downloaded = await client.download_resume("o1", "r1", max_bytes=10_000)
    assert downloaded.content == PDF
    assert len(lever.requests) == 3


async def test_redirect_is_followed_once_without_authorization(
        client: LeverClient, lever: Any, bucket: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    signed = "https://files.example-storage.com/resume.pdf?X-Signature=s3cr3t"
    lever.on("GET", DOWNLOAD, httpx.Response(302, headers={"Location": signed}))
    lever.on("GET", signed, httpx.Response(200, content=PDF,
                                           headers={"Content-Type": "application/pdf"}))
    downloaded = await client.download_resume("o1", "r1", max_bytes=10_000)
    assert downloaded.content == PDF
    first, second = lever.requests
    assert "authorization" in first.headers
    assert "authorization" not in second.headers
    assert second.url == httpx.URL(signed)
    assert bucket.acquired == 2
    assert "s3cr3t" not in caplog.text


async def test_relative_redirect_to_the_api_host_also_drops_auth(client: LeverClient,
                                                                 lever: Any) -> None:
    lever.on("GET", DOWNLOAD, httpx.Response(307, headers={"Location": "/v1/files/r1"}))
    lever.on("GET", "/files/r1", httpx.Response(200, content=PDF))
    assert (await client.download_resume("o1", "r1", max_bytes=10_000)).content == PDF
    assert "authorization" not in lever.requests[1].headers


async def test_only_one_redirect_is_followed(client: LeverClient, lever: Any) -> None:
    lever.on("GET", DOWNLOAD, httpx.Response(302, headers={"Location": "https://a.example/1"}))
    lever.on("GET", "https://a.example/1",
             httpx.Response(302, headers={"Location": "https://b.example/2"}))
    with pytest.raises(LeverRedirectError, match="more than once"):
        await client.download_resume("o1", "r1", max_bytes=10_000)
    assert len(lever.requests) == 2


@pytest.mark.parametrize("location", ["http://files.example.com/resume.pdf", None])
async def test_unsafe_or_missing_redirect_target_is_refused(client: LeverClient, lever: Any,
                                                            location: str | None) -> None:
    headers = {"Location": location} if location else {}
    lever.on("GET", DOWNLOAD, httpx.Response(302, headers=headers))
    with pytest.raises(LeverRedirectError):
        await client.download_resume("o1", "r1", max_bytes=10_000)
    assert len(lever.requests) == 1


async def test_size_cap_applies_after_a_redirect(client: LeverClient, lever: Any) -> None:
    lever.on("GET", DOWNLOAD, httpx.Response(302, headers={"Location": "https://a.example/1"}))
    lever.on("GET", "https://a.example/1", httpx.Response(200, content=b"x" * 5000))
    with pytest.raises(LeverFileTooLarge):
        await client.download_resume("o1", "r1", max_bytes=4096)


async def test_invalid_arguments(client: LeverClient, lever: Any) -> None:
    with pytest.raises(ValueError):
        await client.download_resume("o1", "r1", max_bytes=0)
    with pytest.raises(ValueError):
        await client.download_resume("o1", "../r1", max_bytes=10)
    assert lever.requests == []
