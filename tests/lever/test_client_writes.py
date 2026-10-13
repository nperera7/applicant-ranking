"""Tag and note write-back, perform_as, execute(), and the rank-only guarantee."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from applicant_ranking.domain import PlannedRequest
from applicant_ranking.lever.client import (
    WRITE_PATH_ALLOWLIST,
    LeverClient,
    LeverError,
    LeverNotFound,
    LeverWriteNotAllowed,
)


def _body(request: httpx.Request) -> Any:
    return json.loads(request.content)


# ---------------------------------------------------------------------------------- tags


@pytest.mark.parametrize(("method_name", "action"), [("add_tags", "addTags"),
                                                     ("remove_tags", "removeTags")])
async def test_tag_payload_without_perform_as(client: LeverClient, lever: Any,
                                              method_name: str, action: str) -> None:
    lever.on("POST", f"/opportunities/o1/{action}", httpx.Response(200, json={"data": {}}))
    await getattr(client, method_name)("o1", ["AI Rank: Tier A", "AI Rank: Tier A",
                                              "AI Rank: Verify knockout"])
    request = lever.requests[0]
    assert request.method == "POST"
    assert request.headers["content-type"] == "application/json"
    assert _body(request) == {"tags": ["AI Rank: Tier A", "AI Rank: Verify knockout"]}
    assert "perform_as" not in request.url.params


async def test_tags_with_perform_as(make_client: Callable[..., LeverClient],
                                    lever: Any) -> None:
    client = make_client(perform_as="8d49b010-cc6a-4f40-ace5-e86061c677ed")
    lever.on("POST", "/opportunities/o1/addTags", httpx.Response(200, json={}))
    lever.on("POST", "/opportunities/o1/removeTags", httpx.Response(200, json={}))
    await client.add_tags("o1", ["a"])
    await client.remove_tags("o1", ["b"])
    assert [r.url.params["perform_as"] for r in lever.requests] == [
        "8d49b010-cc6a-4f40-ace5-e86061c677ed"] * 2


async def test_empty_tag_list_sends_nothing(client: LeverClient, lever: Any) -> None:
    await client.add_tags("o1", [])
    await client.remove_tags("o1", [])
    assert lever.requests == []


@pytest.mark.parametrize("tags", [[""], ["  "], ["ok", 3], "AI Rank: Tier A"])
async def test_invalid_tags_are_refused(client: LeverClient, lever: Any, tags: Any) -> None:
    with pytest.raises(ValueError):
        await client.add_tags("o1", tags)
    assert lever.requests == []


# --------------------------------------------------------------------------------- notes


async def test_create_note_payload(client: LeverClient, lever: Any,
                                   payload: Callable[[str], Any]) -> None:
    lever.on("POST", "/opportunities/o1/notes",
             httpx.Response(201, json=payload("note_created.json")))
    note_id = await client.create_note("o1", value="AI ranking (advisory)\n- line")
    assert note_id == "915a6cef-4f91-4f3d-b8ff-a49e1eae2c30"
    request = lever.requests[0]
    assert _body(request) == {"value": "AI ranking (advisory)\n- line", "secret": False,
                              "notifyFollowers": False}
    assert "score" not in _body(request)  # Lever's thumbs scale is for humans
    assert "perform_as" not in request.url.params


async def test_create_note_options_and_perform_as(make_client: Callable[..., LeverClient],
                                                  lever: Any) -> None:
    client = make_client(perform_as="user-1")
    lever.on("POST", "/opportunities/o1/notes",
             httpx.Response(201, json={"data": {"noteId": "n9"}}))
    assert await client.create_note("o1", value="v", secret=True, notify_followers=True) == "n9"
    request = lever.requests[0]
    assert _body(request) == {"value": "v", "secret": True, "notifyFollowers": True}
    assert request.url.params["perform_as"] == "user-1"


@pytest.mark.parametrize("body", [{}, {"data": {}}, None, [], {"data": {"noteId": ""}}])
async def test_create_note_without_note_id_is_an_error(client: LeverClient, lever: Any,
                                                       body: Any) -> None:
    lever.on("POST", "/opportunities/o1/notes", httpx.Response(201, json=body))
    with pytest.raises(LeverError, match="noteId"):
        await client.create_note("o1", value="v")


async def test_update_note_payload(client: LeverClient, lever: Any,
                                   payload: Callable[[str], Any]) -> None:
    note = "b1dbbcfe-281c-46a0-8f51-57049158d3f3"
    lever.on("PUT", f"/opportunities/o1/notes/{note}",
             httpx.Response(200, json=payload("note_updated.json")))
    assert await client.update_note("o1", note, value="new text") == note
    request = lever.requests[0]
    assert request.method == "PUT"
    assert _body(request) == {"values": [{"value": "new text"}], "secret": False}
    assert "perform_as" not in request.url.params


async def test_update_note_with_perform_as_and_bare_response(
        make_client: Callable[..., LeverClient], lever: Any) -> None:
    client = make_client(perform_as="user-1")
    lever.on("PUT", "/opportunities/o1/notes/n1", httpx.Response(200))
    assert await client.update_note("o1", "n1", value="v", secret=True) == "n1"
    assert lever.requests[0].url.params["perform_as"] == "user-1"
    assert _body(lever.requests[0])["secret"] is True


async def test_update_missing_note_raises_not_found(client: LeverClient, lever: Any) -> None:
    lever.on("PUT", "/opportunities/o1/notes/gone",
             httpx.Response(404, json={"code": "ResourceNotFound", "message": "note not found"}))
    with pytest.raises(LeverNotFound):
        await client.update_note("o1", "gone", value="v")
    assert len(lever.requests) == 1


async def test_empty_note_is_refused(client: LeverClient, lever: Any) -> None:
    with pytest.raises(ValueError):
        await client.create_note("o1", value="")
    with pytest.raises(ValueError):
        await client.update_note("o1", "n1", value="")
    assert lever.requests == []


async def test_writes_are_logged_without_payloads(client: LeverClient, lever: Any,
                                                  caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    lever.on("POST", "/opportunities/o1/notes", httpx.Response(201, json={"data": {"noteId": "n"}}))
    await client.create_note("o1", value="Candidate rationale text")
    assert "POST /opportunities/o1/notes" in caplog.text
    assert "Candidate rationale text" not in caplog.text


# ------------------------------------------------------------------------------- execute


async def test_execute_sends_allowlisted_writes_and_gets(client: LeverClient,
                                                         lever: Any) -> None:
    lever.on("POST", "/opportunities/o1/removeTags", httpx.Response(200, json={}))
    lever.on("PUT", "/opportunities/o1/notes/n1",
             httpx.Response(200, json={"data": {"noteId": "n1"}}))
    lever.on("GET", "/opportunities/o1", httpx.Response(200, json={"data": {"id": "o1"}}))
    await client.execute(PlannedRequest(method="POST", path="/opportunities/o1/removeTags",
                                        params={"perform_as": "u1"},
                                        json_body={"tags": ["AI Rank: Tier B"]}))
    result = await client.execute(PlannedRequest(method="PUT", path="/opportunities/o1/notes/n1",
                                                 json_body={"values": [{"value": "v"}]}))
    assert result == {"data": {"noteId": "n1"}}
    await client.execute(PlannedRequest(method="GET", path="/opportunities/o1",
                                        params={"expand": "applications"}))
    first, second, third = lever.requests
    assert first.url.params["perform_as"] == "u1"
    assert _body(first) == {"tags": ["AI Rank: Tier B"]}
    assert second.method == "PUT"
    assert third.url.params["expand"] == "applications"


async def test_execute_write_is_not_retried_after_timeout(client: LeverClient,
                                                          lever: Any) -> None:
    lever.on("POST", "/opportunities/o1/notes", httpx.ReadTimeout("slow"),
             httpx.Response(201, json={"data": {"noteId": "n"}}))
    with pytest.raises(LeverError):
        await client.execute(PlannedRequest(method="POST", path="/opportunities/o1/notes",
                                            json_body={"value": "v"}))
    assert len(lever.requests) == 1


@pytest.mark.parametrize(("method", "path"), [
    ("PUT", "/opportunities/x/stage"),
    ("POST", "/opportunities/x/archived"),
    ("PUT", "/opportunities/x/archived"),
    ("POST", "/opportunities/x/links"),
    ("POST", "/opportunities/x/sources"),
    ("POST", "/opportunities"),
    ("POST", "/opportunities/x/forms"),
    ("POST", "/opportunities/x/files"),
    ("DELETE", "/opportunities/x/notes/n1"),  # allowlisted path, but no deletes
    ("PUT", "/opportunities/x/addTags"),  # allowlisted path, wrong method
    ("POST", "/opportunities/x/notes/n1"),
    ("PUT", "/opportunities/x/notes"),
    ("POST", "/opportunities/x/../y/addTags"),
    ("POST", "/opportunities/../stage/addTags"),
    ("POST", "/opportunities/x?archived=true/addTags"),
    ("POST", "/opportunities/x%2Fstage/addTags"),
    ("POST", "/opportunities/x#/addTags"),
    ("POST", "opportunities/x/addTags"),
    ("POST", "/opportunities/x/addTags/"),
    ("POST", "/users"),
    ("PUT", "/postings/p1"),
])
async def test_execute_refuses_non_allowlisted_writes_before_any_io(
        client: LeverClient, lever: Any, bucket: Any, method: str, path: str) -> None:
    request = PlannedRequest(method=method, path=path, json_body={"stage": "s"})
    with pytest.raises(LeverWriteNotAllowed):
        await client.execute(request)
    with pytest.raises(LeverWriteNotAllowed):
        await client.request_json(method, path, json_body={"stage": "s"})
    assert lever.requests == []
    assert bucket.acquired == 0


@pytest.mark.parametrize("path", ["/opportunities/../users", "/opportunities/x?y=1",
                                  "opportunities", "/a//b", "/a b"])
async def test_unsafe_get_paths_are_refused(client: LeverClient, lever: Any,
                                            path: str) -> None:
    with pytest.raises(ValueError):
        await client.execute(PlannedRequest(method="GET", path=path))
    assert lever.requests == []


def test_allowlist_covers_only_tags_and_notes() -> None:
    allowed = ["/opportunities/o/addTags", "/opportunities/o/removeTags",
               "/opportunities/o/notes", "/opportunities/o/notes/n"]
    refused = ["/opportunities/o/stage", "/opportunities/o/archived", "/opportunities/o",
               "/opportunities/o/notes/n/extra", "/opportunities/o/addTagsX"]
    assert all(any(p.match(path) for p in WRITE_PATH_ALLOWLIST) for path in allowed)
    assert not any(p.match(path) for p in WRITE_PATH_ALLOWLIST for path in refused)


# ----------------------------------------------------------------------------- rank-only


FORBIDDEN_WORDS = ("stage", "archive", "advance", "hire", "delete")
READ_PREFIXES = ("list_", "get_", "find_", "download_")


def _public_methods() -> dict[str, Any]:
    return {name: member for name, member in inspect.getmembers(LeverClient)
            if not name.startswith("_") and callable(member)}


def test_rank_only_no_public_method_can_change_stage_archive_or_hire() -> None:
    methods = _public_methods()
    mentioning = {name for name in methods
                  if any(word in name.lower() for word in FORBIDDEN_WORDS)}
    # Only read-only listings may mention these words.
    assert mentioning == {"list_stages", "list_archive_reasons", "list_deleted_applications",
                          "list_deleted_opportunities"}
    assert all(name.startswith(READ_PREFIXES) for name in mentioning)


def test_rank_only_the_only_mutating_methods_are_tags_and_notes() -> None:
    plumbing = {"aclose", "from_settings", "paginate", "request_json", "execute"}
    mutating = {name for name in _public_methods()
                if not name.startswith(READ_PREFIXES) and name not in plumbing}
    assert mutating == {"add_tags", "remove_tags", "create_note", "update_note"}


async def test_rank_only_read_methods_issue_only_gets(client: LeverClient, lever: Any) -> None:
    empty = httpx.Response(200, json={"data": []})
    for path in ("/postings", "/opportunities", "/opportunities/o1/resumes", "/stages",
                 "/archive_reasons", "/users", "/applications/deleted",
                 "/opportunities/deleted"):
        lever.on("GET", path, empty)
    lever.on("GET", "/postings/p1", httpx.Response(200, json={"data": {"id": "p1"}}))
    lever.on("GET", "/opportunities/o1", httpx.Response(200, json={"data": {"id": "o1"}}))
    lever.on("GET", "/opportunities/o1/resumes/r1/download", httpx.Response(200, content=b"x"))
    await client.get_posting("p1")
    _ = [p async for p in client.list_postings()]
    _ = [o async for o in client.list_opportunities(posting_id="p1")]
    await client.get_opportunity("o1")
    await client.list_resumes("o1")
    await client.download_resume("o1", "r1", max_bytes=10)
    await client.list_stages()
    await client.list_archive_reasons()
    await client.find_user_by_email("a@example.com")
    _ = [d async for d in client.list_deleted_applications(start_ms=0, end_ms=1)]
    _ = [d async for d in client.list_deleted_opportunities(start_ms=0, end_ms=1)]
    assert len(lever.requests) == 11
    assert {r.method for r in lever.requests} == {"GET"}
