"""Fixtures for the LLM tests: a real google-genai client whose HTTP goes to an
``httpx.MockTransport`` (no network, no ADC), plus canned Vertex AI responses."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import google.oauth2.credentials
import httpx
import pytest
from google import genai
from google.genai import types

from applicant_ranking.domain import RoleCriteria

MODEL_VERSION = "gemini-3.1-pro-preview-0219"

Reply = httpx.Response | Exception


def assessment_payload(criteria: RoleCriteria, *, rating: str = "meets",
                       overall_fit: int = 72, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "requirements": [
            {"requirement_id": r.id, "evidence": ["Wrote Python services"], "rating": rating,
             "note": "Clear evidence."}
            for r in criteria.requirements
        ],
        "knockouts": [
            {"knockout_id": k.id, "evidence": ["SQL transformations"], "result": "met",
             "note": "Stated."}
            for k in criteria.knockouts
        ],
        "strengths": ["Production GCP pipelines"],
        "gaps": [],
        "red_flags": [],
        "injection_attempt_detected": False,
        "summary": "The candidate fits most requirements.",
        "confidence": "high",
        "overall_fit": overall_fit,
    }
    payload.update(overrides)
    return payload


def vertex_response(text: str | None = None, *, finish: str | None = "STOP",
                    usage: dict[str, int] | None = None,
                    model_version: str | None = MODEL_VERSION,
                    extra: dict[str, Any] | None = None) -> httpx.Response:
    candidate: dict[str, Any] = {}
    if text is not None:
        candidate["content"] = {"role": "model", "parts": [{"text": text}]}
    if finish is not None:
        candidate["finishReason"] = finish
    body: dict[str, Any] = {"candidates": [candidate]}
    if usage is not None:
        body["usageMetadata"] = usage
    if model_version is not None:
        body["modelVersion"] = model_version
    if extra is not None:
        body.update(extra)
    return httpx.Response(200, json=body)


def json_response(payload: dict[str, Any], **kwargs: Any) -> httpx.Response:
    return vertex_response(json.dumps(payload), **kwargs)


def error_response(code: int, status: str, message: str = "error") -> httpx.Response:
    return httpx.Response(code, json={"error": {"code": code, "message": message,
                                                "status": status}})


@dataclass
class FakeVertex:
    """Scripted Vertex AI endpoint. ``replies`` are consumed in order (an Exception is
    raised from the transport); every request is kept in ``requests``. Construct clients
    with ``client()``; build replies with ``response``/``json_response``/``error``."""

    replies: list[Reply] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)
    clients: list[httpx.AsyncClient] = field(default_factory=list)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.replies:
            raise AssertionError("unexpected extra request to Vertex AI")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def client(self, *, attempts: int = 1) -> genai.Client:
        http = httpx.AsyncClient(transport=httpx.MockTransport(self._handle))
        self.clients.append(http)
        return genai.Client(
            vertexai=True,
            project="test-project",
            location="global",
            credentials=google.oauth2.credentials.Credentials(token="fake"),
            http_options=types.HttpOptions(
                httpx_async_client=http,
                retry_options=types.HttpRetryOptions(
                    attempts=attempts, initial_delay=0.01, max_delay=0.01, jitter=0.01),
            ),
        )

    def body(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)

    # Response builders, exposed here because test modules can't import conftest.
    response = staticmethod(vertex_response)
    json_response = staticmethod(json_response)
    error = staticmethod(error_response)


@pytest.fixture
async def vertex() -> AsyncIterator[FakeVertex]:
    fake = FakeVertex()
    yield fake
    for http in fake.clients:
        await http.aclose()


@pytest.fixture
def payload_for() -> Callable[..., dict[str, Any]]:
    return assessment_payload
