"""Shared test fixtures.

Safety: every test runs with Google/Vertex/gcloud and AR_* environment variables removed,
HOME pointed at an empty directory, and outbound non-loopback connections blocked.
Application-default credentials on a developer machine may belong to another project or
organisation; no test may ever reach them or the network. Area-specific fixtures live in
tests/<area>/conftest.py; don't add area fixtures here.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from applicant_ranking.config import Settings
from applicant_ranking.domain import Knockout, Requirement, RoleCriteria

_SCRUB_PREFIXES = ("AR_", "GOOGLE_", "GEMINI_", "CLOUDSDK_", "GCLOUD_", "LEVER_")


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for key in list(os.environ):
        if key.startswith(_SCRUB_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    yield


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        if self.family == socket.AF_UNIX:
            return real_connect(self, address)
        host = address[0] if isinstance(address, tuple) else address
        if host in ("127.0.0.1", "::1", "localhost"):
            return real_connect(self, address)
        raise RuntimeError(f"network access blocked in tests: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[..., Settings]:
    """Settings that never read .env or the process environment's AR_* values."""

    def _make(**overrides: Any) -> Settings:
        base: dict[str, Any] = {
            "env": "test",
            "extraction_in_subprocess": False,
            "state_db": tmp_path / "state.sqlite3",
            "criteria_dir": tmp_path / "criteria",
            "policy_file": tmp_path / "policy.yaml",
            "gcp_project": "test-project",
            "lever_api_key": "test-lever-key",
        }
        base.update(overrides)
        return Settings(_env_file=None, **base)

    return _make


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest.fixture
def sample_criteria() -> RoleCriteria:
    return RoleCriteria(
        role_id="senior-data-engineer",
        title="Senior Data Engineer",
        version=2,
        posting_ids=["posting-1"],
        summary="Builds and runs production data pipelines on Google Cloud for clients.",
        requirements=[
            Requirement(id="gcp-pipelines",
                        description="3+ years building production data pipelines on GCP "
                                    "(BigQuery, Dataflow or Dataproc)",
                        importance="must_have"),
            Requirement(id="python", description="Professional Python development",
                        importance="must_have"),
            Requirement(id="client-facing", description="Client-facing consulting experience",
                        importance="nice_to_have"),
            Requirement(id="streaming", description="Streaming systems such as Kafka or Pub/Sub",
                        importance="nice_to_have", weight=2),
        ],
        knockouts=[
            Knockout(id="sql", description="Hands-on SQL experience",
                     justification="Every client engagement requires writing SQL daily."),
        ],
        guardrails=["Do not credit certifications without hands-on project experience."],
        approved_by="Pat Example",
        approved_at=datetime(2026, 10, 1, tzinfo=UTC),
    )


SAMPLE_RESUME_TEXT = """\
[CANDIDATE]
[EMAIL] | [PHONE] | [LOCATION]

SUMMARY
Data engineer with 6 years of experience building batch and streaming pipelines.

EXPERIENCE
Senior Data Engineer, Acme Analytics (2021 - present)
- Built production data pipelines on GCP using BigQuery and Dataflow for 12 clients.
- Wrote Python services and SQL transformations processing 2 TB per day.
- Led client workshops to gather requirements.

Data Engineer, Beta Corp (2018 - 2021)
- Developed Kafka streaming jobs in Python.

EDUCATION
B.S. Computer Science, State University, [YEAR]
"""


@pytest.fixture
def sample_resume_text() -> str:
    return SAMPLE_RESUME_TEXT
