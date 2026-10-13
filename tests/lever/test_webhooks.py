from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable
from typing import Any

import pytest

from applicant_ranking.lever import webhooks
from applicant_ranking.lever.webhooks import (
    HANDLED_EVENTS,
    WebhookEnvelope,
    WebhookRejected,
    compute_signature,
    verify_envelope,
)

SIGNING_TOKEN = "test-signing-token"
# Computed outside Python with:
#   printf '%s' "657fd5a0759c87d35660b096dbc388d1d31058869f08ce3f1431458021951" \
#     | openssl dgst -sha256 -hmac "test-signing-token"
PING_SIGNATURE = "58ac391bcf29ef75d606d9791c13bf42edc967d4168dfedd242aa16a5e5862ac"
#   printf '%s' "be58f711f349c7780e4579cd7c63341ac6f881e4dccc31541431454330072" \
#     | openssl dgst -sha256 -hmac "another-webhook-token"
STAGE_CHANGE_SIGNATURE = "846e91c6a805d983bbb6f0f68e78b8856cf910818e973bf20ee53cbd2c91fac3"

MAX_AGE = 600


def _signed(raw: dict[str, Any], signing_token: str = SIGNING_TOKEN) -> WebhookEnvelope:
    body = dict(raw)
    body["signature"] = hmac.new(signing_token.encode(),
                                 f"{body['token']}{body['triggeredAt']}".encode(),
                                 hashlib.sha256).hexdigest()
    return WebhookEnvelope.model_validate(body)


def _verify(envelope: WebhookEnvelope, *, now_ms: int | None = None,
            signing_token: str | None = SIGNING_TOKEN, max_age: int = MAX_AGE) -> None:
    verify_envelope(envelope, signing_token=signing_token,
                    now_ms=envelope.triggeredAt if now_ms is None else now_ms,
                    max_age_seconds=max_age)


# ----------------------------------------------------------------------------- signature


def test_compute_signature_matches_an_independent_vector() -> None:
    assert compute_signature("657fd5a0759c87d35660b096dbc388d1d31058869f08ce3f",
                             1431458021951, SIGNING_TOKEN) == PING_SIGNATURE
    assert compute_signature("be58f711f349c7780e4579cd7c63341ac6f881e4dccc3154",
                             1431454330072, "another-webhook-token") == STAGE_CHANGE_SIGNATURE


def test_compute_signature_is_keyed_and_covers_token_and_timestamp() -> None:
    base = compute_signature("tok", 1, SIGNING_TOKEN)
    assert compute_signature("tok", 1, "other-key") != base
    assert compute_signature("tok", 2, SIGNING_TOKEN) != base
    assert compute_signature("tok2", 1, SIGNING_TOKEN) != base


def test_docs_verify_ping_with_a_real_signature_is_accepted(payload: Callable[[str], Any]
                                                            ) -> None:
    raw = payload("webhook_verify_ping.json")
    raw["signature"] = PING_SIGNATURE
    envelope = WebhookEnvelope.model_validate(raw)
    _verify(envelope, now_ms=raw["triggeredAt"] + 5_000)
    assert envelope.event == "candidateHired"
    assert envelope.data == {}
    assert envelope.opportunity_id is None
    assert envelope.event not in HANDLED_EVENTS


def test_docs_example_signature_does_not_verify_with_our_token(payload: Callable[[str], Any]
                                                               ) -> None:
    # The docs' sample signature isn't even hex; we don't reject on format, only on digest.
    envelope = WebhookEnvelope.model_validate(payload("webhook_verify_ping.json"))
    with pytest.raises(WebhookRejected) as excinfo:
        _verify(envelope)
    assert excinfo.value.reason == "invalid signature"


def test_valid_envelope_passes(payload: Callable[[str], Any]) -> None:
    _verify(_signed(payload("webhook_application_created.json")))


def test_comparison_is_constant_time(monkeypatch: pytest.MonkeyPatch,
                                     payload: Callable[[str], Any]) -> None:
    calls: list[tuple[Any, Any]] = []
    real = hmac.compare_digest

    def spy(a: Any, b: Any) -> bool:
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(webhooks.hmac, "compare_digest", spy)
    _verify(_signed(payload("webhook_application_created.json")))
    assert len(calls) == 1


@pytest.mark.parametrize("signing_token", [None, ""])
def test_missing_signing_token_fails_closed(payload: Callable[[str], Any],
                                            signing_token: str | None) -> None:
    envelope = _signed(payload("webhook_application_created.json"))
    with pytest.raises(WebhookRejected) as excinfo:
        _verify(envelope, signing_token=signing_token)
    assert "signing token" in excinfo.value.reason


@pytest.mark.parametrize("signature", [None, ""])
def test_missing_signature_is_rejected(payload: Callable[[str], Any],
                                       signature: str | None) -> None:
    raw = payload("webhook_application_created.json")
    raw["signature"] = signature
    with pytest.raises(WebhookRejected) as excinfo:
        _verify(WebhookEnvelope.model_validate(raw))
    assert excinfo.value.reason == "missing signature"


def test_empty_token_is_rejected(payload: Callable[[str], Any]) -> None:
    raw = payload("webhook_application_created.json")
    raw["token"] = ""
    with pytest.raises(WebhookRejected) as excinfo:
        _verify(_signed(raw))
    assert excinfo.value.reason == "missing token"


@pytest.mark.parametrize("field", ["token", "triggeredAt", "signature"])
def test_tampering_with_a_signed_field_is_rejected(payload: Callable[[str], Any],
                                                   field: str) -> None:
    envelope = _signed(payload("webhook_application_created.json"))
    tampered = {
        "token": envelope.token + "x",
        "triggeredAt": envelope.triggeredAt + 1,
        "signature": envelope.signature[:-1] + ("0" if envelope.signature[-1] != "0" else "1"),
    }[field]
    forged = envelope.model_copy(update={field: tampered})
    with pytest.raises(WebhookRejected) as excinfo:
        _verify(forged, now_ms=envelope.triggeredAt)
    assert excinfo.value.reason == "invalid signature"


def test_signature_from_another_webhooks_token_is_rejected(payload: Callable[[str], Any]
                                                           ) -> None:
    envelope = _signed(payload("webhook_application_created.json"), "candidate-deleted-token")
    with pytest.raises(WebhookRejected):
        _verify(envelope)


def test_non_ascii_signature_is_a_mismatch_not_a_crash(payload: Callable[[str], Any]) -> None:
    raw = payload("webhook_application_created.json")
    raw["signature"] = "é" * 64
    with pytest.raises(WebhookRejected) as excinfo:
        _verify(WebhookEnvelope.model_validate(raw))
    assert excinfo.value.reason == "invalid signature"


def test_data_is_not_covered_by_the_signature(payload: Callable[[str], Any]) -> None:
    """Documents why ``data`` is only a hint: an altered payload still verifies."""
    envelope = _signed(payload("webhook_application_created.json"))
    altered = envelope.model_copy(update={"data": {"opportunityId": "someone-else"}})
    _verify(altered)
    assert altered.opportunity_id == "someone-else"


# ----------------------------------------------------------------------------- freshness


def test_freshness_window_applies_in_both_directions(payload: Callable[[str], Any]) -> None:
    envelope = _signed(payload("webhook_application_created.json"))
    t = envelope.triggeredAt
    limit = MAX_AGE * 1000
    _verify(envelope, now_ms=t + limit)  # exactly at the edge: accepted
    _verify(envelope, now_ms=t - limit)  # clock skew the other way: accepted
    with pytest.raises(WebhookRejected) as old:
        _verify(envelope, now_ms=t + limit + 1)
    assert old.value.reason.startswith("stale")
    with pytest.raises(WebhookRejected) as future:
        _verify(envelope, now_ms=t - limit - 1)
    assert "future" in future.value.reason


def test_negative_max_age_is_a_programming_error(payload: Callable[[str], Any]) -> None:
    with pytest.raises(ValueError):
        _verify(_signed(payload("webhook_application_created.json")), max_age=-1)


def test_rejection_reasons_never_contain_secrets(payload: Callable[[str], Any]) -> None:
    envelope = _signed(payload("webhook_application_created.json"))
    forged = envelope.model_copy(update={"token": "attacker-token"})
    reasons = []
    for env, now in ((forged, envelope.triggeredAt),
                     (envelope, envelope.triggeredAt + 10**9)):
        with pytest.raises(WebhookRejected) as excinfo:
            _verify(env, now_ms=now)
        reasons.append(excinfo.value.reason)
    for reason in reasons:
        assert SIGNING_TOKEN not in reason
        assert envelope.token not in reason
        assert envelope.signature is not None and envelope.signature not in reason


# ------------------------------------------------------------------------------ envelope


def test_application_created_payload(payload: Callable[[str], Any]) -> None:
    envelope = WebhookEnvelope.model_validate(payload("webhook_application_created.json"))
    assert envelope.event in HANDLED_EVENTS
    assert envelope.opportunity_id == "fc29952a-5b7e-4ce0-bca5-23aab339ec41"
    assert envelope.data["applicationId"] == "6ffe4153-60bb-4e30-bfbe-bd9b9775879c"


def test_opportunity_id_falls_back_to_deprecated_candidate_id() -> None:
    envelope = WebhookEnvelope(event="candidateDeleted", triggeredAt=1, token="t",
                               data={"candidateId": "opp-1", "deletedBy": "u"})
    assert envelope.opportunity_id == "opp-1"


def test_lenient_parsing_of_id_data_and_extra_fields() -> None:
    envelope = WebhookEnvelope.model_validate(
        {"id": 12345, "event": "applicationCreated", "triggeredAt": "1431454330072",
         "token": "t", "signature": "s", "data": None, "futureField": True}
    )
    assert envelope.id == "12345"
    assert envelope.triggeredAt == 1431454330072
    assert envelope.data == {}
    assert envelope.model_extra == {"futureField": True}
    assert WebhookEnvelope(event="e", triggeredAt=1, token="t", id="").id is None


def test_idempotency_key_prefers_the_event_id(payload: Callable[[str], Any]) -> None:
    raw = payload("webhook_application_created.json")
    with_id = WebhookEnvelope.model_validate({**raw, "id": "evt-1"})
    assert with_id.idempotency_key == "id:evt-1"
    retried = WebhookEnvelope.model_validate({**raw, "id": "evt-1", "token": "new-token",
                                              "triggeredAt": raw["triggeredAt"] + 60_000})
    assert retried.idempotency_key == with_id.idempotency_key


def test_idempotency_key_without_id(payload: Callable[[str], Any]) -> None:
    raw = payload("webhook_application_created.json")
    envelope = WebhookEnvelope.model_validate(raw)
    assert envelope.idempotency_key == (
        "ev:applicationCreated:fc29952a-5b7e-4ce0-bca5-23aab339ec41:1431454330072"
    )
    # A redelivery of the same event (new token, same triggeredAt) has the same key ...
    redelivery = WebhookEnvelope.model_validate({**raw, "token": "other"})
    assert redelivery.idempotency_key == envelope.idempotency_key
    # ... a different event or time does not.
    later = WebhookEnvelope.model_validate({**raw, "triggeredAt": raw["triggeredAt"] + 1})
    deleted = WebhookEnvelope.model_validate({**raw, "event": "candidateDeleted"})
    assert len({envelope.idempotency_key, later.idempotency_key,
                deleted.idempotency_key}) == 3
    ping = WebhookEnvelope(event="candidateHired", triggeredAt=5, token="t")
    assert ping.idempotency_key == "ev:candidateHired:-:5"


def test_stage_change_example_from_the_docs(payload: Callable[[str], Any]) -> None:
    raw = payload("webhook_stage_change.json")
    envelope = WebhookEnvelope.model_validate(raw)
    assert envelope.opportunity_id == "3410c8b9-5c31-4bab-b7e9-9f710206d647"
    assert envelope.event not in HANDLED_EVENTS
    envelope = envelope.model_copy(update={"signature": STAGE_CHANGE_SIGNATURE})
    _verify(envelope, signing_token="another-webhook-token")
