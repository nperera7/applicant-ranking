from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Callable
from typing import Any, ClassVar

import google.oauth2.credentials
import httpx
import pytest
from google import genai

from applicant_ranking.config import ConfigError, Settings
from applicant_ranking.domain import RoleCriteria
from applicant_ranking.llm import gemini, prompts
from applicant_ranking.llm.base import (
    AssessmentRequest,
    Evaluator,
    LLMBlockedError,
    LLMConfigError,
    LLMError,
    LLMOutputError,
    LLMUnavailableError,
)
from applicant_ranking.llm.gemini import GeminiEvaluator, estimate_cost_usd
from applicant_ranking.llm.schema import LLMAssessment

MODEL = "gemini-3.1-pro-preview"
USAGE = {"promptTokenCount": 4000, "candidatesTokenCount": 600, "thoughtsTokenCount": 900,
         "totalTokenCount": 5500}
SECRET_RESUME_LINE = "Wrote Python services and SQL transformations processing 2 TB per day."
PNG_1 = b"\x89PNG\r\n\x1a\npage-one"
PNG_2 = b"\x89PNG\r\n\x1a\npage-two"
_REAL_CLIENT = genai.Client


@pytest.fixture
def request_(sample_criteria: RoleCriteria, sample_resume_text: str) -> AssessmentRequest:
    return AssessmentRequest(criteria=sample_criteria, resume_text=sample_resume_text,
                             evaluation_id="eval-1")


@pytest.fixture
def good(sample_criteria: RoleCriteria, payload_for: Callable[..., dict[str, Any]]
         ) -> dict[str, Any]:
    return payload_for(sample_criteria)


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(key)
            keys |= _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            keys |= _walk_keys(item)
    return keys


def _text_of(content: dict[str, Any]) -> str:
    return "".join(part.get("text", "") for part in content["parts"])


# --- happy path -------------------------------------------------------------------------


async def test_assess_happy_path_request_and_result(vertex, settings: Settings, request_,
                                                    good, sample_resume_text: str) -> None:
    vertex.replies.append(vertex.json_response(good, usage=USAGE))
    evaluator = GeminiEvaluator(settings, client=vertex.client())

    result = await evaluator.assess(request_)

    (sent,) = vertex.requests
    assert sent.method == "POST"
    assert str(sent.url).endswith(
        f"/projects/test-project/locations/global/publishers/google/models/{MODEL}"
        ":generateContent")
    assert sent.headers["authorization"] == "Bearer fake"
    body = vertex.body()

    system = _text_of(body["systemInstruction"])
    assert system == prompts.build_system_instruction(request_.criteria)
    ids = [r.id for r in request_.criteria.requirements] + ["sql"]
    positions = [system.index(f"id: {i} ") for i in ids]
    assert positions == sorted(positions)
    assert SECRET_RESUME_LINE not in system

    (content,) = body["contents"]
    assert content["role"] == "user"
    user_text = _text_of(content)
    fence = re.match(r"<<<(RESUME-[0-9a-f]{16})>>>\n", user_text).group(1)
    assert user_text == prompts.build_user_prompt(sample_resume_text, fence)

    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    schema = config["responseSchema"]
    assert schema["type"] == "OBJECT"
    assert schema["property_ordering"] == list(LLMAssessment.model_fields)
    assert config["thinkingConfig"] == {"thinking_level": "LOW"}
    assert config["maxOutputTokens"] == settings.gemini_max_output_tokens
    assert config["seed"] == 7
    keys = _walk_keys(body)
    for forbidden in ("temperature", "topP", "top_p", "topK", "top_k", "thinkingBudget",
                      "thinking_budget"):
        assert forbidden not in keys
    assert body["labels"] == {"app": "applicant-ranking"}
    assert body["safetySettings"] == [
        {"category": f"HARM_CATEGORY_{c}", "threshold": "BLOCK_ONLY_HIGH"}
        for c in ("HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT")
    ]

    assert result.assessment == LLMAssessment.model_validate(good)
    assert result.raw_text == json.dumps(good)
    assert result.model_id == MODEL
    assert result.model_version == "gemini-3.1-pro-preview-0219"
    assert result.prompt_version == prompts.PROMPT_VERSION == evaluator.prompt_version
    assert result.finish_reason == "STOP"
    assert result.usage.prompt_tokens == 4000
    assert result.usage.output_tokens == 600
    assert result.usage.thinking_tokens == 900
    assert result.usage.total_tokens == 5500
    # 4000 * $2/M + (600 + 900) * $12/M
    assert result.usage.estimated_cost_usd == pytest.approx(0.008 + 0.018)
    assert result.generation_config == {
        "model": MODEL, "location": "global", "thinking_level": "LOW",
        "max_output_tokens": 16384, "seed": 7, "prompt_version": prompts.PROMPT_VERSION,
        "response_mime_type": "application/json", "safety_threshold": "BLOCK_ONLY_HIGH",
    }
    dumped = json.dumps(result.generation_config)
    assert SECRET_RESUME_LINE not in dumped and "fake" not in dumped


async def test_settings_drive_thinking_level_seed_and_model(vertex, make_settings, request_,
                                                            payload_for) -> None:
    settings = make_settings(gemini_thinking_level="high", gemini_seed=None,
                             gemini_model="gemini-9-pro", gemini_max_output_tokens=4096)
    vertex.replies.append(vertex.json_response(payload_for(request_.criteria), usage=USAGE))
    evaluator = GeminiEvaluator(settings, client=vertex.client())

    result = await evaluator.assess(request_)

    assert "/models/gemini-9-pro:generateContent" in str(vertex.requests[0].url)
    config = vertex.body()["generationConfig"]
    assert config["thinkingConfig"] == {"thinking_level": "HIGH"}
    assert "seed" not in config
    assert config["maxOutputTokens"] == 4096
    assert result.model_id == evaluator.model_id == "gemini-9-pro"
    assert result.generation_config["seed"] is None
    assert result.generation_config["thinking_level"] == "HIGH"
    assert result.usage.estimated_cost_usd is None  # no price list for this model


async def test_each_request_gets_a_new_fence(vertex, settings, request_, good) -> None:
    vertex.replies += [vertex.json_response(good), vertex.json_response(good)]
    evaluator = GeminiEvaluator(settings, client=vertex.client())
    await evaluator.assess(request_)
    await evaluator.assess(request_)
    fences = {re.match(r"<<<(RESUME-[0-9a-f]{16})>>>", _text_of(vertex.body(i)["contents"][0]))
              .group(1) for i in (0, 1)}
    assert len(fences) == 2
    assert _text_of(vertex.body(0)["systemInstruction"]) == _text_of(
        vertex.body(1)["systemInstruction"])


async def test_corrective_note_is_sent_in_the_system_instruction(vertex, settings, request_,
                                                                 good, sample_resume_text
                                                                 ) -> None:
    vertex.replies.append(vertex.json_response(good))
    note = "Do not mention age; refer only to the candidate."
    retry = request_.model_copy(update={"corrective_note": note})
    await GeminiEvaluator(settings, client=vertex.client()).assess(retry)
    system = _text_of(vertex.body()["systemInstruction"])
    assert system == prompts.build_system_instruction(retry.criteria, corrective_note=note)
    assert system.rstrip().endswith(f"# Correction\n{note}")
    user_text = _text_of(vertex.body()["contents"][0])
    assert note not in user_text and "Correction" not in user_text
    fence = re.match(r"<<<(RESUME-[0-9a-f]{16})>>>\n", user_text).group(1)
    assert user_text == prompts.build_user_prompt(sample_resume_text, fence)


async def test_assessment_ids_are_returned_as_is(vertex, settings, request_, payload_for
                                                 ) -> None:
    payload = payload_for(request_.criteria)
    payload["requirements"] = [
        {"requirement_id": "made-up", "evidence": [], "rating": "meets", "note": "x"},
        payload["requirements"][1],
    ]
    payload["knockouts"] = []
    vertex.replies.append(vertex.json_response(payload))
    result = await GeminiEvaluator(settings, client=vertex.client()).assess(request_)
    assert [r.requirement_id for r in result.assessment.requirements] == ["made-up", "python"]
    assert result.assessment.knockouts == []


def test_evaluator_satisfies_protocol(vertex, settings) -> None:
    assert isinstance(GeminiEvaluator(settings, client=vertex.client()), Evaluator)


# --- parsing ----------------------------------------------------------------------------


async def test_parsed_none_falls_back_to_text(vertex, settings, request_, good) -> None:
    vertex.replies.append(vertex.json_response(good))
    client = vertex.client()
    real = client.aio.models.generate_content

    async def parsed_none(**kwargs: Any) -> Any:
        response = await real(**kwargs)
        assert response.parsed is not None
        response.parsed = None
        return response

    client.aio.models.generate_content = parsed_none
    result = await GeminiEvaluator(settings, client=client).assess(request_)
    assert result.assessment == LLMAssessment.model_validate(good)


async def test_parsed_dict_is_validated(vertex, settings, request_, good) -> None:
    vertex.replies.append(vertex.json_response(good))
    client = vertex.client()
    real = client.aio.models.generate_content

    async def parsed_dict(**kwargs: Any) -> Any:
        response = await real(**kwargs)
        response.parsed = dict(good)
        return response

    client.aio.models.generate_content = parsed_dict
    result = await GeminiEvaluator(settings, client=client).assess(request_)
    assert result.assessment == LLMAssessment.model_validate(good)


async def test_schema_mismatch_is_output_error_without_leaking_text(vertex, settings,
                                                                     request_, good) -> None:
    bad = dict(good, overall_fit=150, summary=SECRET_RESUME_LINE)
    bad["requirements"] = [dict(good["requirements"][0], rating="superb")]
    vertex.replies.append(vertex.json_response(bad))
    with pytest.raises(LLMOutputError) as info:
        await GeminiEvaluator(settings, client=vertex.client()).assess(request_)
    message = str(info.value)
    assert "overall_fit" in message and "requirements.0.rating" in message
    assert "superb" not in message and SECRET_RESUME_LINE not in message
    assert info.value.__cause__ is None and info.value.__suppress_context__
    assert info.value.retryable is False


@pytest.mark.parametrize("text", ['{"requirements": [', "not json at all", "   "])
async def test_invalid_json_is_output_error(vertex, settings, request_, text: str) -> None:
    vertex.replies.append(vertex.response(text))
    with pytest.raises(LLMOutputError):
        await GeminiEvaluator(settings, client=vertex.client()).assess(request_)


async def test_stop_without_text_is_output_error(vertex, settings, request_) -> None:
    vertex.replies.append(vertex.response(None, finish="STOP"))
    with pytest.raises(LLMOutputError, match="no text"):
        await GeminiEvaluator(settings, client=vertex.client()).assess(request_)


# --- finish reasons and blocks ------------------------------------------------------------


async def test_max_tokens_is_output_error(vertex, settings, request_) -> None:
    vertex.replies.append(vertex.response('{"requirements": [{"requirement_id": "py',
                                          finish="MAX_TOKENS", usage=USAGE))
    with pytest.raises(LLMOutputError, match="MAX_TOKENS") as info:
        await GeminiEvaluator(settings, client=vertex.client()).assess(request_)
    assert not isinstance(info.value, LLMBlockedError)


async def test_prompt_block_reason_is_blocked_error(vertex, settings, request_) -> None:
    vertex.replies.append(httpx.Response(200, json={
        "promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},
        "usageMetadata": {"promptTokenCount": 10}}))
    with pytest.raises(LLMBlockedError, match="PROHIBITED_CONTENT") as info:
        await GeminiEvaluator(settings, client=vertex.client()).assess(request_)
    assert info.value.retryable is False


async def test_block_reason_wins_over_candidates(vertex, settings, request_, good) -> None:
    vertex.replies.append(vertex.json_response(
        good, extra={"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(LLMBlockedError, match="SAFETY"):
        await GeminiEvaluator(settings, client=vertex.client()).assess(request_)


@pytest.mark.parametrize("reason", ["SPII", "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST",
                                    "RECITATION"])
async def test_blocking_finish_reasons(vertex, settings, request_, reason: str) -> None:
    vertex.replies.append(vertex.response(None, finish=reason))
    with pytest.raises(LLMBlockedError, match=reason):
        await GeminiEvaluator(settings, client=vertex.client()).assess(request_)


@pytest.mark.parametrize("reason", ["OTHER", "LANGUAGE", None])
async def test_other_finish_reasons_are_output_errors(vertex, settings, request_, good,
                                                      reason: str | None) -> None:
    vertex.replies.append(vertex.json_response(good, finish=reason))
    with pytest.raises(LLMOutputError) as info:
        await GeminiEvaluator(settings, client=vertex.client()).assess(request_)
    assert not isinstance(info.value, LLMBlockedError)


async def test_no_candidates_is_output_error(vertex, settings, request_) -> None:
    vertex.replies.append(httpx.Response(200, json={"usageMetadata": {"promptTokenCount": 3}}))
    with pytest.raises(LLMOutputError, match="no candidates"):
        await GeminiEvaluator(settings, client=vertex.client()).assess(request_)


# --- HTTP errors and retries ----------------------------------------------------------------


async def test_429_then_success_is_retried_by_the_sdk(vertex, settings, request_, good) -> None:
    vertex.replies += [vertex.error(429, "RESOURCE_EXHAUSTED"), vertex.json_response(good)]
    result = await GeminiEvaluator(settings, client=vertex.client(attempts=3)).assess(request_)
    assert len(vertex.requests) == 2
    assert result.assessment == LLMAssessment.model_validate(good)


async def test_persistent_429_is_retryable_unavailable(vertex, settings, request_) -> None:
    vertex.replies += [vertex.error(429, "RESOURCE_EXHAUSTED", "Quota exceeded")
                       for _ in range(3)]
    with pytest.raises(LLMUnavailableError) as info:
        await GeminiEvaluator(settings, client=vertex.client(attempts=3)).assess(request_)
    assert len(vertex.requests) == 3
    assert info.value.retryable is True
    assert info.value.code == 429
    assert info.value.status == "RESOURCE_EXHAUSTED"
    assert "Quota exceeded" in str(info.value)


@pytest.mark.parametrize(("code", "status"), [(400, "INVALID_ARGUMENT"),
                                              (403, "PERMISSION_DENIED"), (404, "NOT_FOUND")])
async def test_client_errors_are_config_errors(vertex, settings, request_, code: int,
                                               status: str) -> None:
    vertex.replies.append(vertex.error(code, status, "Request contains an invalid argument."))
    with pytest.raises(LLMConfigError) as info:
        await GeminiEvaluator(settings, client=vertex.client(attempts=3)).assess(request_)
    assert len(vertex.requests) == 1  # not retried
    assert info.value.retryable is False
    assert (info.value.code, info.value.status) == (code, status)


async def test_408_is_unavailable_not_config(vertex, settings, request_) -> None:
    vertex.replies.append(vertex.error(408, "DEADLINE_EXCEEDED"))
    with pytest.raises(LLMUnavailableError) as info:
        await GeminiEvaluator(settings, client=vertex.client()).assess(request_)
    assert info.value.retryable is True


@pytest.mark.parametrize(("code", "status"), [(503, "UNAVAILABLE"), (500, "INTERNAL")])
async def test_server_errors_are_unavailable(vertex, settings, request_, code: int,
                                             status: str) -> None:
    vertex.replies += [vertex.error(code, status) for _ in range(2)]
    with pytest.raises(LLMUnavailableError) as info:
        await GeminiEvaluator(settings, client=vertex.client(attempts=2)).assess(request_)
    assert len(vertex.requests) == 2
    assert info.value.retryable is True
    assert info.value.code == code


@pytest.mark.parametrize("exc", [
    httpx.ConnectError("connection refused"),
    httpx.ReadTimeout("timed out"),
    httpx.RemoteProtocolError("peer closed connection"),
])
async def test_transport_errors_are_unavailable(vertex, settings, request_,
                                                exc: Exception) -> None:
    vertex.replies.append(exc)
    with pytest.raises(LLMUnavailableError) as info:
        await GeminiEvaluator(settings, client=vertex.client()).assess(request_)
    assert info.value.retryable is True
    assert type(exc).__name__ in str(info.value)


async def test_connect_error_is_retried_then_succeeds(vertex, settings, request_, good) -> None:
    vertex.replies += [httpx.ConnectError("refused"), vertex.json_response(good)]
    result = await GeminiEvaluator(settings, client=vertex.client(attempts=2)).assess(request_)
    assert len(vertex.requests) == 2
    assert result.finish_reason == "STOP"


# --- transcription ----------------------------------------------------------------------


async def test_transcribe_sends_png_parts_first(vertex, settings) -> None:
    vertex.replies.append(vertex.json_response(
        {"text": "Jane's resume text", "legible": True}, usage=USAGE))
    text = await GeminiEvaluator(settings, client=vertex.client()).transcribe([PNG_1, PNG_2])

    assert text == "Jane's resume text"
    assert str(vertex.requests[0].url).endswith(f"/models/{MODEL}:generateContent")
    body = vertex.body()
    (content,) = body["contents"]
    parts = content["parts"]
    assert [p["inlineData"]["mime_type"] for p in parts[:2]] == ["image/png", "image/png"]
    assert [p["inlineData"]["data"] for p in parts[:2]] == [
        base64.b64encode(png).decode() for png in (PNG_1, PNG_2)]
    assert parts[2] == {"text": prompts.build_transcription_instruction()}
    assert len(parts) == 3
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"]["property_ordering"] == ["text", "legible"]
    assert config["thinkingConfig"] == {"thinking_level": "LOW"}
    assert "temperature" not in config
    assert body["labels"] == {"app": "applicant-ranking"}


async def test_transcribe_illegible_is_output_error(vertex, settings) -> None:
    vertex.replies.append(vertex.json_response({"text": "", "legible": False}))
    with pytest.raises(LLMOutputError, match="not legible"):
        await GeminiEvaluator(settings, client=vertex.client()).transcribe([PNG_1])


async def test_transcribe_blocked_and_truncated(vertex, settings) -> None:
    evaluator = GeminiEvaluator(settings, client=vertex.client())
    vertex.replies.append(vertex.response(None, finish="SPII"))
    with pytest.raises(LLMBlockedError):
        await evaluator.transcribe([PNG_1])
    vertex.replies.append(vertex.response('{"text": "abc', finish="MAX_TOKENS"))
    with pytest.raises(LLMOutputError):
        await evaluator.transcribe([PNG_1])


async def test_transcribe_without_pages_makes_no_call(vertex, settings) -> None:
    with pytest.raises(LLMError):
        await GeminiEvaluator(settings, client=vertex.client()).transcribe([])
    assert vertex.requests == []


async def test_transcribe_caps_pages(vertex, make_settings) -> None:
    settings = make_settings(vision_max_pages=2)
    vertex.replies.append(vertex.json_response({"text": "t", "legible": True}))
    await GeminiEvaluator(settings, client=vertex.client()).transcribe([PNG_1, PNG_2, PNG_1])
    parts = vertex.body()["contents"][0]["parts"]
    assert sum("inlineData" in p for p in parts) == 2


async def test_transcribe_maps_http_errors(vertex, settings) -> None:
    vertex.replies.append(vertex.error(400, "INVALID_ARGUMENT"))
    with pytest.raises(LLMConfigError):
        await GeminiEvaluator(settings, client=vertex.client()).transcribe([PNG_1])


# --- usage and cost -----------------------------------------------------------------------


def test_estimate_cost_standard_and_long_context() -> None:
    assert estimate_cost_usd(100_000, 0, 0) == pytest.approx(0.20)
    assert estimate_cost_usd(0, 1_000_000, 0) == pytest.approx(12.00)
    assert estimate_cost_usd(0, 500_000, 500_000) == pytest.approx(12.00)  # thinking = output
    assert estimate_cost_usd(200_000, 0, 0) == pytest.approx(0.40)  # boundary: standard rate
    # above 200K prompt tokens the whole request is billed at $4 / $18
    assert estimate_cost_usd(250_000, 10_000, 5_000) == pytest.approx(1.00 + 0.27)
    assert estimate_cost_usd(0, 0, 0) == 0


async def test_missing_usage_metadata_gives_zero_usage(vertex, settings, request_, good
                                                       ) -> None:
    vertex.replies.append(vertex.json_response(good, model_version=None))
    result = await GeminiEvaluator(settings, client=vertex.client()).assess(request_)
    assert result.usage.prompt_tokens == result.usage.total_tokens == 0
    assert result.model_version is None


async def test_total_tokens_fallback(vertex, settings, request_, good) -> None:
    usage = {"promptTokenCount": 100, "candidatesTokenCount": 20}
    vertex.replies.append(vertex.json_response(good, usage=usage))
    result = await GeminiEvaluator(settings, client=vertex.client()).assess(request_)
    assert result.usage.total_tokens == 120
    assert result.usage.thinking_tokens == 0


# --- client construction and lifecycle ------------------------------------------------------


class _SpyClient:
    instances: ClassVar[list[_SpyClient]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.real = _REAL_CLIENT(**kwargs)
        self.closed = 0
        self.aclosed = 0
        outer = self

        class _Aio:
            async def aclose(self) -> None:
                outer.aclosed += 1

        self.aio = _Aio()
        _SpyClient.instances.append(self)

    def close(self) -> None:
        self.closed += 1


def test_builds_vertex_client_from_settings(monkeypatch: pytest.MonkeyPatch,
                                            make_settings) -> None:
    _SpyClient.instances.clear()
    monkeypatch.setattr(gemini.genai, "Client", _SpyClient)
    creds = google.oauth2.credentials.Credentials(token="fake")
    settings = make_settings(gcp_project="egen-project", gemini_timeout_seconds=120,
                             gemini_retry_attempts=5)

    GeminiEvaluator(settings, credentials=creds)

    (spy,) = _SpyClient.instances
    assert spy.kwargs["vertexai"] is True
    assert spy.kwargs["project"] == "egen-project"
    assert spy.kwargs["location"] == "global"
    assert spy.kwargs["credentials"] is creds
    options = spy.kwargs["http_options"]
    assert options.timeout == 120_000  # milliseconds
    assert options.retry_options.attempts == 5
    assert options.retry_options.initial_delay == 2.0
    assert options.retry_options.max_delay == 30.0
    api = spy.real._api_client
    assert api.vertexai is True
    assert (api.project, api.location) == ("egen-project", "global")
    assert api.api_key is None


async def test_real_client_construction_needs_no_adc(make_settings) -> None:
    # Environment is scrubbed and HOME is empty: construction must not look up ADC.
    creds = google.oauth2.credentials.Credentials(token="fake")
    evaluator = GeminiEvaluator(make_settings(), credentials=creds)
    assert evaluator.model_id == MODEL
    assert evaluator._client._api_client.project == "test-project"
    await evaluator.aclose()


def test_missing_project_is_config_error(make_settings) -> None:
    with pytest.raises(ConfigError, match="AR_GCP_PROJECT"):
        GeminiEvaluator(make_settings(gcp_project=None))


async def test_aclose_closes_owned_client_once(monkeypatch: pytest.MonkeyPatch,
                                               settings) -> None:
    _SpyClient.instances.clear()
    monkeypatch.setattr(gemini.genai, "Client", _SpyClient)
    evaluator = GeminiEvaluator(settings,
                                credentials=google.oauth2.credentials.Credentials(token="t"))
    await evaluator.aclose()
    await evaluator.aclose()
    (spy,) = _SpyClient.instances
    assert (spy.aclosed, spy.closed) == (1, 1)


async def test_aclose_leaves_injected_client_open(vertex, settings, request_, good) -> None:
    client = vertex.client()
    await GeminiEvaluator(settings, client=client).aclose()
    vertex.replies.append(vertex.json_response(good))
    result = await GeminiEvaluator(settings, client=client).assess(request_)
    assert result.finish_reason == "STOP"


# --- logging ------------------------------------------------------------------------------


async def test_logs_never_contain_resume_or_output(vertex, settings, request_, good,
                                                   caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    good = dict(good, summary="UNIQUE-MODEL-OUTPUT-MARKER")
    vertex.replies += [vertex.error(429, "RESOURCE_EXHAUSTED"), vertex.json_response(good)]
    await GeminiEvaluator(settings, client=vertex.client(attempts=2)).assess(request_)
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "eval-1" in logged  # the evaluation id is logged
    assert SECRET_RESUME_LINE not in logged
    assert "UNIQUE-MODEL-OUTPUT-MARKER" not in logged
    assert "Bearer" not in logged
