"""Lever webhook envelope parsing and verification.

Envelope: ``{"id"?, "event", "triggeredAt": <ms int>, "token": <random str>,
"signature": <hex>, "data": {...}}``. The signature is
``hex(HMAC-SHA256(key=<that webhook's signature token>, msg=token + str(triggeredAt)))``.
It does NOT cover ``data``, so ``data`` is only a hint: take the opportunityId from it and
re-fetch everything else from the API. Defences: constant-time compare, a freshness
window on triggeredAt, and a replay cache of seen tokens (kept by the store).
Don't reject on token length or signature format (Lever's own examples vary); just
compare digests. "Verify connection" pings send event "candidateHired" with data {}.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

HANDLED_EVENTS = frozenset({"applicationCreated", "candidateDeleted"})


class WebhookEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    event: str
    triggeredAt: int
    token: str
    signature: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", mode="before")
    @classmethod
    def _id_as_text(cls, v: Any) -> Any:
        if isinstance(v, int) and not isinstance(v, bool):
            return str(v)
        return v or None

    @field_validator("data", mode="before")
    @classmethod
    def _null_data(cls, v: Any) -> Any:
        return {} if v is None else v

    @property
    def opportunity_id(self) -> str | None:
        v = self.data.get("opportunityId") or self.data.get("candidateId")
        return str(v) if v else None

    @property
    def idempotency_key(self) -> str:
        """Event id when present, else (event, opportunityId, triggeredAt)."""
        if self.id:
            return f"id:{self.id}"
        return f"ev:{self.event}:{self.opportunity_id or '-'}:{self.triggeredAt}"


class WebhookRejected(Exception):
    """Signature missing/invalid, stale, or replayed. ``reason`` is safe to log."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def compute_signature(token: str, triggered_at: int, signing_token: str) -> str:
    """Lever's scheme (its JS sample: ``hmac(SIGNATURE_TOKEN).update(body.token +
    body.triggeredAt).digest('hex')``)."""
    message = f"{token}{triggered_at}".encode()
    return hmac.new(signing_token.encode(), message, hashlib.sha256).hexdigest()


def verify_envelope(envelope: WebhookEnvelope, *, signing_token: str | None, now_ms: int,
                    max_age_seconds: int) -> None:
    """Raise WebhookRejected unless the signature matches (hmac.compare_digest) and
    triggeredAt is within ``max_age_seconds`` of ``now_ms`` (either direction, allowing
    small clock skew). A missing signing token for the event is a rejection (fail closed).
    Replay detection is separate (store.remember_webhook_token)."""
    if max_age_seconds < 0:
        raise ValueError("max_age_seconds must be >= 0")
    if not signing_token:
        raise WebhookRejected("no signing token configured for this event")
    if not envelope.signature:
        raise WebhookRejected("missing signature")
    if not envelope.token:
        # The token is the replay-cache key; an empty one would collide with every other.
        raise WebhookRejected("missing token")
    expected = compute_signature(envelope.token, envelope.triggeredAt, signing_token)
    # Bytes, so a non-ASCII signature is a mismatch rather than a TypeError.
    if not hmac.compare_digest(expected.encode(), envelope.signature.encode()):
        raise WebhookRejected("invalid signature")
    skew_ms = now_ms - envelope.triggeredAt
    limit_ms = max_age_seconds * 1000
    if skew_ms > limit_ms:
        raise WebhookRejected("stale: triggeredAt is older than the freshness window")
    if -skew_ms > limit_ms:
        raise WebhookRejected("triggeredAt is too far in the future")
