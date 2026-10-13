"""Pagination and the typed read endpoints, against Lever's documented example payloads."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from applicant_ranking.lever.client import (
    DELETED_WINDOW_MS,
    LeverClient,
    LeverError,
    LeverInvalidRequest,
    LeverNotFound,
)
from applicant_ranking.lever.models import Application, Opportunity


def _page(items: list[Any], next_: str | None = None, has_next: bool | None = None) -> Any:
    body: dict[str, Any] = {"data": items}
    if next_ is not None:
        body["next"] = next_
    if has_next is not None:
        body["hasNext"] = has_next
    return httpx.Response(200, json=body)


async def _collect(it: Any) -> list[Any]:
    return [x async for x in it]


# ------------------------------------------------------------------------------ paginate


async def test_paginate_follows_next_while_has_next(client: LeverClient, lever: Any) -> None:
    lever.on("GET", "/opportunities",
             _page([{"id": "1"}, {"id": "2"}], "0.1414895548650.a6070140", True),
             _page([{"id": "3"}], "0.1414895548999.b0000000", True),
             _page([{"id": "4"}], "ignored-cursor", False))
    items = await _collect(client.paginate("/opportunities", {"posting_id": "p1"}))
    assert [i["id"] for i in items] == ["1", "2", "3", "4"]
    offsets = [r.url.params.get("offset") for r in lever.requests]
    assert offsets == [None, "0.1414895548650.a6070140", "0.1414895548999.b0000000"]
    assert all(r.url.params["limit"] == "100" for r in lever.requests)
    assert all(r.url.params["posting_id"] == "p1" for r in lever.requests)


async def test_paginate_bare_array_is_a_single_page(client: LeverClient, lever: Any) -> None:
    lever.on("GET", "/opportunities/o1/files",
             httpx.Response(200, json=[{"id": "f1"}, "junk", {"id": "f2"}]))
    items = await _collect(client.paginate("/opportunities/o1/files"))
    assert items == [{"id": "f1"}, {"id": "f2"}]
    assert len(lever.requests) == 1


@pytest.mark.parametrize("body", [
    {"data": [{"id": "1"}], "next": "abc"},  # hasNext missing
    {"data": [{"id": "1"}], "hasNext": True},  # next missing
    {"data": [{"id": "1"}], "hasNext": True, "next": ""},
    {"data": [{"id": "1"}], "hasNext": "true", "next": "abc"},  # not a real boolean
])
async def test_paginate_missing_has_next_or_next_ends(client: LeverClient, lever: Any,
                                                      body: dict[str, Any]) -> None:
    lever.on("GET", "/tags", httpx.Response(200, json=body))
    assert await _collect(client.paginate("/tags")) == [{"id": "1"}]
    assert len(lever.requests) == 1


@pytest.mark.parametrize("body", [None, {"data": None}, {"data": {"id": "x"}}, {}])
async def test_paginate_tolerates_empty_or_odd_bodies(client: LeverClient, lever: Any,
                                                      body: Any) -> None:
    lever.on("GET", "/tags", httpx.Response(200, json=body))
    assert await _collect(client.paginate("/tags")) == []


async def test_paginate_passes_next_back_exactly_as_received(client: LeverClient,
                                                             lever: Any) -> None:
    # From the file_actions example: the cursor is already percent-encoded.
    cursor = "%5B1711931200000%2C%22a3f7c2d1-0000%22%5D"
    lever.on("GET", "/opportunities/o1/file_actions",
             _page([{"id": "1"}], cursor, True), _page([{"id": "2"}]))
    await _collect(client.paginate("/opportunities/o1/file_actions"))
    raw_query = lever.requests[1].url.query.decode()
    assert f"offset={cursor}" in raw_query
    assert "%25" not in raw_query


async def test_paginate_encodes_a_cursor_that_is_not_url_safe(client: LeverClient,
                                                              lever: Any) -> None:
    cursor = "a+b&limit=1 c/%zz"
    lever.on("GET", "/tags", _page([{"id": "1"}], cursor, True), _page([]))
    await _collect(client.paginate("/tags"))
    second = lever.requests[1]
    assert second.url.params["offset"] == cursor
    assert second.url.params.get_list("limit") == ["100"]


async def test_paginate_detects_a_repeated_cursor(client: LeverClient, lever: Any) -> None:
    lever.on("GET", "/tags", _page([{"id": "1"}], "same", True))
    with pytest.raises(LeverError, match="cursor"):
        await _collect(client.paginate("/tags"))
    assert len(lever.requests) == 2


async def test_paginate_owns_limit_and_offset(client: LeverClient, lever: Any) -> None:
    lever.on("GET", "/tags", _page([]))
    await _collect(client.paginate("/tags", {"limit": 5, "offset": "x", "q": "1"}, limit=50))
    params = lever.requests[0].url.params
    assert params["limit"] == "50"
    assert "offset" not in params
    assert params["q"] == "1"


@pytest.mark.parametrize("limit", [0, 101])
async def test_paginate_limit_bounds(client: LeverClient, limit: int) -> None:
    with pytest.raises(ValueError):
        await _collect(client.paginate("/tags", limit=limit))


async def test_paginate_retries_a_failed_page(client: LeverClient, lever: Any) -> None:
    lever.on("GET", "/tags", _page([{"id": "1"}], "c1", True),
             httpx.Response(503), _page([{"id": "2"}]))
    assert [i["id"] for i in await _collect(client.paginate("/tags"))] == ["1", "2"]
    assert [r.url.params.get("offset") for r in lever.requests] == [None, "c1", "c1"]


# --------------------------------------------------------------------------- typed reads


async def test_list_opportunities_params_and_docs_payload(client: LeverClient, lever: Any,
                                                          payload: Callable[[str], Any]
                                                          ) -> None:
    lever.on("GET", "/opportunities", httpx.Response(200, json=payload("opportunities_list.json")))
    opps = await _collect(client.list_opportunities(posting_id="p1",
                                                    created_at_start_ms=1759900000000))
    assert [o.name for o in opps] == ["Shane Smith", "Chaofan West", "Roberta Easton"]
    assert all(isinstance(o, Opportunity) for o in opps)
    assert opps[0].profile_url == (
        "https://hire.lever.co/candidates/250d8f03-738a-4bba-a671-8a3d73477145")
    params = lever.requests[0].url.params
    assert dict(params) == {"posting_id": "p1", "expand": "applications", "archived": "false",
                            "created_at_start": "1759900000000", "limit": "100"}
    assert "include" not in params  # include= restricts the response; never sent


@pytest.mark.parametrize(("archived", "expected"), [(None, None), (True, "true"),
                                                    (False, "false")])
async def test_list_opportunities_archived_filter(client: LeverClient, lever: Any,
                                                  archived: bool | None,
                                                  expected: str | None) -> None:
    lever.on("GET", "/opportunities", _page([]))
    await _collect(client.list_opportunities(posting_id="p1", archived=archived))
    assert lever.requests[0].url.params.get("archived") == expected
    assert "created_at_start" not in lever.requests[0].url.params


async def test_list_skips_invalid_items_without_logging_their_content(
        client: LeverClient, lever: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    lever.on("GET", "/opportunities",
             _page([{"name": "Secret Person", "emails": ["secret@example.com"]},
                    {"id": "o2", "name": "Valid"}]))
    opps = await _collect(client.list_opportunities(posting_id="p1"))
    assert [o.id for o in opps] == ["o2"]
    assert "Skipping a Opportunity item" in caplog.text
    assert "Secret Person" not in caplog.text
    assert "secret@example.com" not in caplog.text


async def test_get_opportunity_with_expanded_application(client: LeverClient, lever: Any,
                                                         payload: Callable[[str], Any]) -> None:
    body = payload("opportunity.json")
    body["data"]["applications"] = [payload("application.json")]
    lever.on("GET", "/opportunities/250d8f03-738a-4bba-a671-8a3d73477145",
             httpx.Response(200, json=body))
    opp = await client.get_opportunity("250d8f03-738a-4bba-a671-8a3d73477145")
    assert lever.requests[0].url.params["expand"] == "applications"
    assert isinstance(opp.application, Application)
    assert opp.posting_id == "6a1e4b79-75a3-454f-9417-ea79612b9585"
    assert opp.storage_allowed
    assert not opp.is_archived


async def test_get_opportunity_errors(client: LeverClient, lever: Any) -> None:
    lever.on("GET", "/opportunities/missing",
             httpx.Response(404, json={"code": "ResourceNotFound", "message": "not found"}))
    lever.on("GET", "/opportunities/odd", httpx.Response(200, json={"data": []}))
    lever.on("GET", "/opportunities/noid", httpx.Response(200, json={"data": {"name": "x"}}))
    with pytest.raises(LeverNotFound):
        await client.get_opportunity("missing")
    with pytest.raises(LeverError, match="unexpected body"):
        await client.get_opportunity("odd")
    with pytest.raises(LeverError, match="invalid Opportunity"):
        await client.get_opportunity("noid")


@pytest.mark.parametrize("bad_id", ["", "../postings", "a/b", "x?y=1", "a b", ".", "..",
                                    "%2e%2e", "a#b", "x" * 300])
async def test_ids_are_validated_before_any_io(client: LeverClient, lever: Any,
                                               bad_id: str) -> None:
    with pytest.raises(LeverInvalidRequest):
        await client.get_opportunity(bad_id)
    with pytest.raises(ValueError):  # LeverInvalidRequest is also a ValueError
        await client.list_resumes(bad_id)
    assert lever.requests == []


async def test_get_posting(client: LeverClient, lever: Any,
                           payload: Callable[[str], Any]) -> None:
    lever.on("GET", "/postings/f2f01e16-27f8-4711-a728-7d49499795a0",
             httpx.Response(200, json=payload("posting.json")))
    posting = await client.get_posting("f2f01e16-27f8-4711-a728-7d49499795a0")
    assert posting.text == "Infrastructure Engineer"
    assert posting.content.lists[0].text == "Job requirements"
    assert posting.locations == ["San Francisco", "New York City"]
    assert "include" not in lever.requests[0].url.params


async def test_list_postings(client: LeverClient, lever: Any,
                             payload: Callable[[str], Any]) -> None:
    lever.on("GET", "/postings", _page([payload("posting.json")["data"]]))
    postings = await _collect(client.list_postings(state="published"))
    assert [p.id for p in postings] == ["f2f01e16-27f8-4711-a728-7d49499795a0"]
    assert lever.requests[0].url.params["state"] == "published"
    await _collect(client.list_postings())
    assert "state" not in lever.requests[1].url.params


async def test_list_postings_asks_for_both_job_sites(client: LeverClient, lever: Any) -> None:
    """Lever returns only public-site postings for state=published unless both
    distributionChannel values are sent (as a repeated parameter)."""
    lever.on("GET", "/postings", _page([]))
    await _collect(client.list_postings(state="published"))
    first = lever.requests[0].url
    assert "distributionChannel=public&distributionChannel=internal" in first.query.decode()
    assert first.params.get_list("distributionChannel") == ["public", "internal"]
    assert "confidentiality" not in first.params
    await _collect(client.list_postings())  # all states: published ones need it too
    assert lever.requests[1].url.params.get_list("distributionChannel") == ["public", "internal"]
    for state in ("closed", "draft", "internal", "pending", "rejected"):
        await _collect(client.list_postings(state=state))
        assert "distributionChannel" not in lever.requests[-1].url.params, state
    await _collect(client.list_postings(state="published", distribution_channels=None))
    assert "distributionChannel" not in lever.requests[-1].url.params
    await _collect(client.list_postings(state="published", distribution_channels=["internal"]))
    assert lever.requests[-1].url.params.get_list("distributionChannel") == ["internal"]


async def test_list_postings_confidentiality_filter(client: LeverClient, lever: Any) -> None:
    lever.on("GET", "/postings", _page([]))
    await _collect(client.list_postings(confidentiality="all"))
    assert lever.requests[0].url.params["confidentiality"] == "all"
    with pytest.raises(ValueError, match="confidentiality must be one of"):
        await _collect(client.list_postings(confidentiality="secret"))
    assert len(lever.requests) == 1


async def test_list_opportunities_confidentiality_filter(client: LeverClient, lever: Any) -> None:
    lever.on("GET", "/opportunities", _page([{"id": "o1"}]))
    await _collect(client.list_opportunities(posting_id="p1"))
    assert "confidentiality" not in lever.requests[0].url.params  # Lever's default
    for value in ("all", "confidential", "non-confidential"):
        await _collect(client.list_opportunities(posting_id="p1", confidentiality=value))
        assert lever.requests[-1].url.params["confidentiality"] == value
    with pytest.raises(ValueError, match="confidentiality must be one of"):
        await _collect(client.list_opportunities(posting_id="p1", confidentiality="everything"))
    assert len(lever.requests) == 4


async def test_confidential_posting_lists_all_its_opportunities(
        client: LeverClient, lever: Any, payload: Callable[[str], Any]) -> None:
    """Having read a confidential posting proves the key has confidential access, so its
    applicants are listed with confidentiality=all instead of Lever's silent default."""
    body = payload("posting.json")
    body["data"]["confidentiality"] = "confidential"
    posting_id = body["data"]["id"]
    lever.on("GET", f"/postings/{posting_id}", httpx.Response(200, json=body))
    lever.on("GET", "/opportunities", _page([{"id": "o1"}]))
    posting = await client.get_posting(posting_id)
    assert posting.confidentiality == "confidential"
    await _collect(client.list_opportunities(posting_id=posting_id))
    assert lever.requests[-1].url.params["confidentiality"] == "all"
    # An explicit value wins, and other postings are untouched.
    await _collect(client.list_opportunities(posting_id=posting_id,
                                             confidentiality="non-confidential"))
    assert lever.requests[-1].url.params["confidentiality"] == "non-confidential"
    await _collect(client.list_opportunities(posting_id="other-posting"))
    assert "confidentiality" not in lever.requests[-1].url.params


async def test_non_confidential_posting_never_sends_confidentiality(
        client: LeverClient, lever: Any, payload: Callable[[str], Any]) -> None:
    """A key without confidential access would get an access error from the filter."""
    posting_id = payload("posting.json")["data"]["id"]
    lever.on("GET", f"/postings/{posting_id}",
             httpx.Response(200, json=payload("posting.json")))
    lever.on("GET", "/opportunities", _page([{"id": "o1"}]))
    posting = await client.get_posting(posting_id)
    assert posting.confidentiality == "non-confidential"
    await _collect(client.list_opportunities(posting_id=posting_id))
    assert "confidentiality" not in lever.requests[-1].url.params


async def test_list_opportunities_warns_when_lever_lists_nothing(
        client: LeverClient, lever: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    lever.on("GET", "/opportunities", _page([]))
    await _collect(client.list_opportunities(posting_id="p1"))
    assert "no active opportunities for posting p1" in caplog.text
    assert "confidential access" in caplog.text
    caplog.clear()
    await _collect(client.list_opportunities(posting_id="p1", archived=None,
                                             confidentiality="all"))
    assert "no opportunities for posting p1 (confidentiality=all)" in caplog.text
    caplog.clear()
    lever.on("GET", "/opportunities", _page([{"id": "o1"}]))
    await _collect(client.list_opportunities(posting_id="p1"))
    assert "opportunities for posting" not in caplog.text


async def test_list_keeps_an_unreadable_opportunity_as_a_placeholder(
        client: LeverClient, lever: Any, caplog: pytest.LogCaptureFixture) -> None:
    """An applicant whose record does not validate is not dropped: a placeholder with the
    id only (storage not allowed) is yielded; items without a usable id are still skipped."""
    caplog.set_level(logging.WARNING)
    lever.on("GET", "/opportunities", _page([
        {"id": "opp-bad", "name": "Secret Person", "emails": ["secret@example.com"],
         "tags": [1, 2]},
        {"name": "No Id Person", "emails": ["noid@example.com"], "tags": [1]},
        {"id": "../etc/passwd", "tags": [1]},
        {"id": "opp-ok", "name": "Valid"},
    ]))
    opps = await _collect(client.list_opportunities(posting_id="p1"))
    assert [o.id for o in opps] == ["opp-bad", "opp-ok"]
    bad, ok = opps
    assert bad.record_valid is False and bad.storage_allowed is False
    assert bad.name is None and bad.emails == [] and bad.tags == []
    assert ok.record_valid and ok.storage_allowed and ok.name == "Valid"
    assert ("Opportunity opp-bad from GET /opportunities failed validation (2 errors); kept "
            "as an unreadable record") in caplog.text
    assert caplog.text.count("Skipping a Opportunity item") == 2
    for secret in ("Secret Person", "secret@example.com", "No Id Person", "noid@example.com",
                   "passwd"):
        assert secret not in caplog.text


async def test_list_still_skips_invalid_items_of_other_models(
        client: LeverClient, lever: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    lever.on("GET", "/postings", _page([{"id": "p-bad", "tags": [1]}, {"id": "p-ok"}]))
    lever.on("GET", "/opportunities/o1/resumes",
             _page([{"id": "r-bad", "file": "not-an-object"}, {"id": "r-ok"}]))
    assert [p.id for p in await _collect(client.list_postings())] == ["p-ok"]
    assert [r.id for r in await client.list_resumes("o1")] == ["r-ok"]
    assert "Skipping a Posting item" in caplog.text
    assert "Skipping a Resume item" in caplog.text


async def test_list_opportunities_tolerates_nulls(client: LeverClient, lever: Any) -> None:
    lever.on("GET", "/opportunities", _page([
        {"id": "o1", "name": "Shane Smith", "tags": None, "emails": None, "applications": None,
         "urls": None},
    ]))
    [opp] = await _collect(client.list_opportunities(posting_id="p1"))
    assert opp.record_valid and opp.storage_allowed
    assert opp.tags == [] and opp.emails == [] and opp.applications == [] and opp.urls == {}


async def test_list_resumes_tolerates_null_parsed_lists(client: LeverClient, lever: Any) -> None:
    lever.on("GET", "/opportunities/o1/resumes", _page([
        {"id": "r1", "file": {"name": "resume.pdf", "ext": "pdf", "status": "processed"},
         "parsedData": {"positions": None, "schools": None}},
    ]))
    [resume] = await client.list_resumes("o1")
    assert resume.file is not None and resume.file.status == "processed"
    assert resume.parsedData is not None and resume.parsedData.positions == []


async def test_list_resumes(client: LeverClient, lever: Any,
                            payload: Callable[[str], Any]) -> None:
    lever.on("GET", "/opportunities/o1/resumes",
             httpx.Response(200, json=payload("resumes.json")))
    resumes = await client.list_resumes("o1")
    assert resumes[0].id == "68b3e478-ff21-4529-bd29-23ed016042d7"
    assert resumes[0].file is not None and resumes[0].file.status == "processed"
    assert resumes[0].parsedData is not None and resumes[0].parsedData.positions


async def test_list_stages_and_archive_reasons(client: LeverClient, lever: Any,
                                               payload: Callable[[str], Any]) -> None:
    lever.on("GET", "/stages", httpx.Response(200, json=payload("stages.json")))
    lever.on("GET", "/archive_reasons", httpx.Response(200, json=payload("archive_reasons.json")))
    stages = await client.list_stages()
    reasons = await client.list_archive_reasons()
    assert stages[0].text == "New applicant"
    assert {r.text: r.type for r in reasons}["Hired"] == "hired"
    assert all(r.method == "GET" for r in lever.requests)


async def test_find_user_by_email(client: LeverClient, lever: Any,
                                  payload: Callable[[str], Any]) -> None:
    lever.on("GET", "/users", httpx.Response(200, json=payload("users_by_email.json")))
    user = await client.find_user_by_email(" Chandler@Example.com ")
    assert user is not None and user.id == "8d49b010-cc6a-4f40-ace5-e86061c677ed"
    assert lever.requests[0].url.params["email"] == "Chandler@Example.com"


async def test_find_user_by_email_requires_an_exact_match(client: LeverClient, lever: Any,
                                                          payload: Callable[[str], Any]) -> None:
    lever.on("GET", "/users", httpx.Response(200, json=payload("users_by_email.json")))
    assert await client.find_user_by_email("rachel@example.com") is None
    lever.on("GET", "/users", _page([]))
    assert await client.find_user_by_email("nobody@example.com") is None
    with pytest.raises(ValueError):
        await client.find_user_by_email("  ")


async def test_list_deleted_applications_splits_into_30_day_windows(client: LeverClient,
                                                                    lever: Any) -> None:
    start = 1_711_929_600_000
    end = start + 75 * 24 * 3600 * 1000

    def reply(request: httpx.Request) -> httpx.Response:
        s = int(request.url.params["deleted_at_start"])
        return httpx.Response(200, json={"data": [
            {"id": f"app-{s}", "deletedAt": str(s + 1), "opportunityId": "o", "postingId": None}
        ]})

    lever.on("GET", "/applications/deleted", reply)
    deleted = await _collect(client.list_deleted_applications(start_ms=start, end_ms=end))
    windows = [(int(r.url.params["deleted_at_start"]), int(r.url.params["deleted_at_end"]))
               for r in lever.requests]
    assert windows == [(start, start + DELETED_WINDOW_MS),
                       (start + DELETED_WINDOW_MS, start + 2 * DELETED_WINDOW_MS),
                       (start + 2 * DELETED_WINDOW_MS, end)]
    assert all(e - s <= 30 * 24 * 3600 * 1000 for s, e in windows)
    assert [d.deletedAt for d in deleted] == [s + 1 for s, _ in windows]


async def test_list_deleted_applications_short_and_empty_ranges(client: LeverClient,
                                                                lever: Any) -> None:
    lever.on("GET", "/applications/deleted", _page([]))
    await _collect(client.list_deleted_applications(start_ms=10, end_ms=10 + DELETED_WINDOW_MS))
    assert len(lever.requests) == 1
    await _collect(client.list_deleted_applications(start_ms=10, end_ms=10))
    await _collect(client.list_deleted_applications(start_ms=10, end_ms=5))
    assert len(lever.requests) == 1


async def test_list_deleted_opportunities_is_one_inclusive_window(client: LeverClient,
                                                                  lever: Any) -> None:
    start = 1_711_929_600_000
    end = start + 90 * 24 * 3600 * 1000  # longer than 30 days: still one range
    lever.on("GET", "/opportunities/deleted",
             _page([{"id": "opp-1", "deletedAt": "1700000000000", "deletedBy": "user-1"}],
                   "c1", True),
             _page([{"id": "opp-2", "deletedAt": None}, {"deletedAt": 5}]))  # no id: skipped

    deleted = await _collect(client.list_deleted_opportunities(start_ms=start, end_ms=end))

    assert [(d.id, d.deletedAt, d.deletedBy) for d in deleted] == [
        ("opp-1", 1_700_000_000_000, "user-1"), ("opp-2", None, None)]
    assert [r.url.path for r in lever.requests] == ["/v1/opportunities/deleted"] * 2
    first = lever.requests[0].url.params
    assert (first["deleted_at_start"], first["deleted_at_end"]) == (str(start), str(end))
    assert first["limit"] == "100" and "offset" not in first
    assert lever.requests[1].url.params["offset"] == "c1"


async def test_list_deleted_applications_paginates_each_window(client: LeverClient,
                                                               lever: Any) -> None:
    lever.on("GET", "/applications/deleted",
             _page([{"id": "a"}], "c1", True), _page([{"id": "b"}]))
    deleted = await _collect(client.list_deleted_applications(start_ms=0, end_ms=1000))
    assert [d.id for d in deleted] == ["a", "b"]
    assert lever.requests[1].url.params["offset"] == "c1"
    assert lever.requests[1].url.params["deleted_at_end"] == "1000"


async def test_query_strings_are_kept_out_of_request_logs(client: LeverClient, lever: Any,
                                                          caplog: pytest.LogCaptureFixture
                                                          ) -> None:
    """httpx logs each request URL at INFO; emails and cursors must not reach the logs."""
    caplog.set_level(logging.DEBUG)
    lever.on("GET", "/users", _page([]))
    await client.find_user_by_email("private.person@example.com")
    assert "HTTP Request: GET https://api.lever.co/v1/users" in caplog.text
    assert "private.person" not in caplog.text
