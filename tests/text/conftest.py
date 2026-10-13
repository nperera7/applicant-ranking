"""Fixtures for the text area (normalise, security scan, redaction, signals, preparer).

Preparer tests use ``StubExtractor`` in place of the real ``ResumeExtractor``; the stub
only has ``async def extract(self, data) -> ExtractionResult``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from applicant_ranking.domain import ExtractionResult
from applicant_ranking.resumes.prepare import ResumePreparer

HMAC_KEY = b"test-hmac-key"


class StubExtractor:
    def __init__(self, result: ExtractionResult | None = None,
                 error: Exception | None = None) -> None:
        self.result = result or ExtractionResult(kind="txt")
        self.error = error
        self.calls: list[bytes] = []

    async def extract(self, data: bytes) -> ExtractionResult:
        self.calls.append(data)
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def hmac_key() -> bytes:
    return HMAC_KEY


@pytest.fixture
def make_preparer(hmac_key: bytes) -> Callable[..., ResumePreparer]:
    def _make(result: ExtractionResult | None = None, *, error: Exception | None = None,
              **overrides: Any) -> ResumePreparer:
        kwargs: dict[str, Any] = {
            "extractor": StubExtractor(result, error),
            "hmac_key": hmac_key,
            "max_chars": 30_000,
            "redact_education_years": True,
        }
        kwargs.update(overrides)
        return ResumePreparer(**kwargs)

    return _make


@pytest.fixture
def tags() -> Callable[[str], str]:
    """ASCII -> Unicode tag characters (invisible "ASCII smuggling")."""
    return lambda s: "".join(chr(0xE0000 + ord(c)) for c in s)
