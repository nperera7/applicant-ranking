"""Gemini Pro on Vertex AI via google-genai (verified against SDK 2.29.0 source).

Client: ``genai.Client(vertexai=True, project=<required>, location=<"global">,
http_options=types.HttpOptions(timeout=<ms!>, retry_options=types.HttpRetryOptions(
attempts=N, initial_delay=2.0, max_delay=30.0)))``. Always pass project explicitly (the SDK
otherwise looks one up from ADC). HttpOptions.timeout is MILLISECONDS. Retries are off in
the SDK unless retry_options is given; default retryable codes 408/429/500/502/503/504 +
httpx timeout/connect errors.

Request (``client.aio.models.generate_content``, non-streaming):
- ``contents``: the user prompt text (scoring) or PNG Parts + instruction (transcription).
  The corrective note of a retry travels in the system instruction, never in ``contents``.
- ``config=types.GenerateContentConfig(system_instruction=..., response_mime_type=
  "application/json", response_schema=LLMAssessment, thinking_config=types.ThinkingConfig(
  thinking_level=<LOW|MEDIUM|HIGH>), max_output_tokens=..., seed=..., labels={"app":
  "applicant-ranking"}, safety_settings=[explicit BLOCK_ONLY_HIGH for HARASSMENT,
  HATE_SPEECH, SEXUALLY_EXPLICIT, DANGEROUS_CONTENT])``.
- Do NOT set temperature/top_p/top_k (deprecated for Gemini 3) or thinking_budget
  (400 with thinking_level).

Response handling, in order:
1. ``response.prompt_feedback.block_reason`` set -> LLMBlockedError.
2. No candidates, or ``candidates[0].finish_reason`` != STOP: SAFETY/PROHIBITED_CONTENT/
   SPII/BLOCKLIST/RECITATION -> LLMBlockedError; MAX_TOKENS -> LLMOutputError.
3. ``response.parsed`` may be silently None even when text is JSON: fall back to
   ``LLMAssessment.model_validate_json(response.text)``; failure -> LLMOutputError.
4. usage from ``usage_metadata`` (prompt_token_count, candidates_token_count,
   thoughts_token_count, total_token_count); estimated cost at $2/M input and $12/M
   output incl. thinking (<=200K prompt), $4/$18 above (gemini-3.1-pro-preview, global).
5. ``model_version`` from ``response.model_version``.
Errors: ``google.genai.errors.ClientError`` 429 -> LLMUnavailableError (retryable);
other 4xx -> LLMConfigError; ``ServerError`` -> LLMUnavailableError; raw
``httpx.TimeoutException``/``httpx.ConnectError`` escaping the SDK -> LLMUnavailableError.

Tests construct a real client with ``credentials=google.oauth2.credentials.Credentials(
token="fake")`` and ``HttpOptions(httpx_async_client=httpx.AsyncClient(transport=
httpx.MockTransport(handler)))`` so no network or ADC is touched (see tests).

Additional choices made here:
- No candidates and no block reason, or any other non-STOP finish reason (OTHER,
  LANGUAGE, ...) -> LLMOutputError. IMAGE_SAFETY/IMAGE_PROHIBITED_CONTENT count as blocks.
- 408 (request timeout) is mapped like 429: the request was not at fault. Any other
  ``httpx.TransportError`` escaping the SDK (read/write/protocol errors) is a network
  failure -> LLMUnavailableError.
- Error messages never contain prompt, resume or model-output text: schema failures
  report only field locations and error types, and are raised without chaining the
  pydantic error (whose repr includes input values).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

from applicant_ranking.config import Settings
from applicant_ranking.domain import TokenUsage
from applicant_ranking.llm import prompts
from applicant_ranking.llm.base import (
    AssessmentRequest,
    LLMBlockedError,
    LLMCallResult,
    LLMConfigError,
    LLMError,
    LLMOutputError,
    LLMUnavailableError,
)
from applicant_ranking.llm.schema import LLMAssessment, TranscriptionResult

logger = logging.getLogger(__name__)

INPUT_USD_PER_M = (2.00, 4.00)  # (<=200K prompt tokens, >200K)
OUTPUT_USD_PER_M = (12.00, 18.00)
LONG_CONTEXT_THRESHOLD = 200_000
# Models the rates above apply to; other models get no cost estimate rather than a wrong one.
PRICED_MODELS = frozenset({"gemini-3.1-pro-preview"})

LABELS = {"app": "applicant-ranking"}
RETRY_INITIAL_DELAY_S = 2.0
RETRY_MAX_DELAY_S = 30.0
SAFETY_CATEGORIES = (
    types.HarmCategory.HARM_CATEGORY_HARASSMENT,
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
)
SAFETY_THRESHOLD = types.HarmBlockThreshold.BLOCK_ONLY_HIGH

_BLOCKED_FINISH_REASONS = frozenset({
    types.FinishReason.SAFETY,
    types.FinishReason.PROHIBITED_CONTENT,
    types.FinishReason.SPII,
    types.FinishReason.BLOCKLIST,
    types.FinishReason.RECITATION,
    types.FinishReason.IMAGE_SAFETY,
    types.FinishReason.IMAGE_PROHIBITED_CONTENT,
})
_UNAVAILABLE_CLIENT_CODES = frozenset({408, 429})
_MAX_ERROR_MESSAGE_CHARS = 300
_MAX_REPORTED_SCHEMA_ERRORS = 5


def build_http_options(settings: Settings) -> types.HttpOptions:
    """Timeout (milliseconds) and explicit retries; the SDK retries nothing by default."""
    return types.HttpOptions(
        timeout=settings.gemini_timeout_seconds * 1000,
        retry_options=types.HttpRetryOptions(
            attempts=settings.gemini_retry_attempts,
            initial_delay=RETRY_INITIAL_DELAY_S,
            max_delay=RETRY_MAX_DELAY_S,
        ),
    )


def build_client(settings: Settings, *, credentials: Any | None = None) -> genai.Client:
    """A Vertex AI client for the configured project and location, never an API-key client.

    ``credentials``: a ``google.auth.credentials.Credentials``; None means ADC, which the
    SDK loads lazily on the first request (the explicit project avoids an ADC lookup here).
    Raises ``ConfigError`` when ``AR_GCP_PROJECT`` is not set.
    """
    return genai.Client(
        vertexai=True,
        project=settings.require_gcp_project(),
        location=settings.gcp_location,
        credentials=credentials,
        http_options=build_http_options(settings),
    )


class GeminiEvaluator:
    model_id: str
    prompt_version: str

    def __init__(self, settings: Settings, *, client: Any | None = None,
                 credentials: Any | None = None) -> None:
        """``client``: an already-built ``genai.Client`` (tests inject one); otherwise one
        is built from settings (``settings.require_gcp_project()``), with ``credentials``
        if given (ADC otherwise). An injected client is not closed by ``aclose``."""
        self.model_id = settings.gemini_model
        self.prompt_version = prompts.PROMPT_VERSION
        self._settings = settings
        self._owns_client = client is None
        self._client = client if client is not None else build_client(
            settings, credentials=credentials)
        self._thinking_level = types.ThinkingLevel(settings.gemini_thinking_level.upper())
        self._closed = False

    @property
    def generation_config(self) -> dict[str, Any]:
        """JSON-safe record of what is sent with every request (audit log; no secrets)."""
        return {
            "model": self.model_id,
            "location": self._settings.gcp_location,
            "thinking_level": self._thinking_level.value,
            "max_output_tokens": self._settings.gemini_max_output_tokens,
            "seed": self._settings.gemini_seed,
            "prompt_version": self.prompt_version,
            "response_mime_type": "application/json",
            "safety_threshold": SAFETY_THRESHOLD.value,
        }

    async def assess(self, request: AssessmentRequest) -> LLMCallResult:
        fence = prompts.make_fence()
        contents = prompts.build_user_prompt(request.resume_text, fence)
        config = self._config(
            response_schema=LLMAssessment,
            system_instruction=prompts.build_system_instruction(
                request.criteria, corrective_note=request.corrective_note),
        )
        response = await self._generate(contents, config)
        finish_reason = _check_response(response)
        assessment = _parse(response, LLMAssessment)
        usage = self._usage(response)
        logger.info(
            "gemini assessment done evaluation_id=%s model_version=%s finish=%s "
            "prompt_tokens=%d output_tokens=%d thinking_tokens=%d",
            request.evaluation_id, response.model_version, finish_reason,
            usage.prompt_tokens, usage.output_tokens, usage.thinking_tokens,
        )
        return LLMCallResult(
            assessment=assessment,
            raw_text=response.text or "",
            model_id=self.model_id,
            model_version=response.model_version,
            prompt_version=self.prompt_version,
            finish_reason=finish_reason,
            usage=usage,
            generation_config=self.generation_config,
        )

    async def transcribe(self, page_images: list[bytes]) -> str:
        """PNG Parts (``types.Part.from_bytes(data=png, mime_type="image/png")``) first,
        then the transcription instruction; ``response_schema=TranscriptionResult``.
        ``legible=False`` -> LLMOutputError.

        At most ``settings.vision_max_pages`` pages are sent (a warning is logged when
        more are given); an empty list raises LLMError without calling the model."""
        if not page_images:
            raise LLMError("no page images to transcribe")
        max_pages = self._settings.vision_max_pages
        if len(page_images) > max_pages:
            logger.warning("transcribing only the first %d of %d pages",
                           max_pages, len(page_images))
        contents: list[Any] = [
            types.Part.from_bytes(data=png, mime_type="image/png")
            for png in page_images[:max_pages]
        ]
        contents.append(prompts.build_transcription_instruction())
        response = await self._generate(contents, self._config(response_schema=TranscriptionResult))
        finish_reason = _check_response(response)
        result = _parse(response, TranscriptionResult)
        usage = self._usage(response)
        logger.info(
            "gemini transcription done pages=%d model_version=%s finish=%s legible=%s "
            "prompt_tokens=%d output_tokens=%d thinking_tokens=%d",
            len(contents) - 1, response.model_version, finish_reason, result.legible,
            usage.prompt_tokens, usage.output_tokens, usage.thinking_tokens,
        )
        if not result.legible:
            raise LLMOutputError("the model reported the resume pages as not legible")
        return result.text

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aio.aclose()
            self._client.close()

    # ------------------------------------------------------------------------------------

    def _config(self, *, response_schema: type[BaseModel],
                system_instruction: str | None = None) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_schema,
            thinking_config=types.ThinkingConfig(thinking_level=self._thinking_level),
            max_output_tokens=self._settings.gemini_max_output_tokens,
            seed=self._settings.gemini_seed,
            labels=dict(LABELS),
            safety_settings=[
                types.SafetySetting(category=category, threshold=SAFETY_THRESHOLD)
                for category in SAFETY_CATEGORIES
            ],
        )

    async def _generate(self, contents: Any,
                        config: types.GenerateContentConfig) -> types.GenerateContentResponse:
        try:
            return await self._client.aio.models.generate_content(
                model=self.model_id, contents=contents, config=config)
        except genai_errors.ClientError as exc:
            message = _api_error_message(exc)
            if exc.code in _UNAVAILABLE_CLIENT_CODES:
                raise LLMUnavailableError(message, retryable=True, code=exc.code,
                                          status=exc.status) from exc
            raise LLMConfigError(message, retryable=False, code=exc.code,
                                 status=exc.status) from exc
        except genai_errors.APIError as exc:  # ServerError (5xx) or an unexpected status
            raise LLMUnavailableError(_api_error_message(exc), retryable=True, code=exc.code,
                                      status=exc.status) from exc
        except httpx.TransportError as exc:  # TimeoutException, ConnectError, ...
            raise LLMUnavailableError(
                f"Vertex AI request failed after retries: {type(exc).__name__}",
                retryable=True) from exc

    def _usage(self, response: types.GenerateContentResponse) -> TokenUsage:
        meta = response.usage_metadata
        if meta is None:
            return TokenUsage()
        prompt_tokens = meta.prompt_token_count or 0
        output_tokens = meta.candidates_token_count or 0
        thinking_tokens = meta.thoughts_token_count or 0
        cost = (estimate_cost_usd(prompt_tokens, output_tokens, thinking_tokens)
                if self.model_id in PRICED_MODELS else None)
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            total_tokens=meta.total_token_count or prompt_tokens + output_tokens + thinking_tokens,
            estimated_cost_usd=cost,
        )


def estimate_cost_usd(prompt_tokens: int, output_tokens: int, thinking_tokens: int) -> float:
    """Standard pay-as-you-go list price on the global endpoint; thinking is billed as
    output, and a prompt over 200K tokens moves the whole request to the long-context rate."""
    tier = 1 if prompt_tokens > LONG_CONTEXT_THRESHOLD else 0
    cost = (prompt_tokens * INPUT_USD_PER_M[tier]
            + (output_tokens + thinking_tokens) * OUTPUT_USD_PER_M[tier]) / 1_000_000
    return round(cost, 8)


def _enum_name(value: Any) -> str:
    return str(getattr(value, "value", value))


def _api_error_message(exc: genai_errors.APIError) -> str:
    message = " ".join(str(exc.message or "").split())[:_MAX_ERROR_MESSAGE_CHARS]
    return f"Vertex AI error {exc.code} {exc.status or ''}: {message}".strip()


def _check_response(response: types.GenerateContentResponse) -> str:
    """Steps 1-2 of the docstring; returns the finish reason ("STOP") when usable."""
    feedback = response.prompt_feedback
    if feedback is not None and feedback.block_reason is not None:
        raise LLMBlockedError(f"prompt blocked by Vertex AI: {_enum_name(feedback.block_reason)}",
                              retryable=False)
    if not response.candidates:
        raise LLMOutputError("response has no candidates")
    finish = response.candidates[0].finish_reason
    if finish == types.FinishReason.STOP:
        return _enum_name(finish)
    if finish in _BLOCKED_FINISH_REASONS:
        raise LLMBlockedError(f"response blocked by Vertex AI: finish_reason={_enum_name(finish)}",
                              retryable=False)
    if finish == types.FinishReason.MAX_TOKENS:
        raise LLMOutputError("response truncated at max_output_tokens (finish_reason=MAX_TOKENS)")
    raise LLMOutputError(f"response not completed: finish_reason={_enum_name(finish)}")


def _parse[M: BaseModel](response: types.GenerateContentResponse, schema: type[M]) -> M:
    """Step 3: prefer ``response.parsed``; it is silently None when the SDK could not
    validate the text, so re-validate the text to get (and report) the real error."""
    parsed = response.parsed
    try:
        if isinstance(parsed, schema):
            return parsed
        if isinstance(parsed, dict):
            return schema.model_validate(parsed)
        text = response.text
        if not text or not text.strip():
            raise LLMOutputError("response has no text")
        return schema.model_validate_json(text)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['type']}"
            for err in exc.errors(include_url=False, include_input=False)[
                :_MAX_REPORTED_SCHEMA_ERRORS]
        )
        raise LLMOutputError(
            f"model output does not match {schema.__name__} "
            f"({exc.error_count()} errors): {details}"
        ) from None
