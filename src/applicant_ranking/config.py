"""Runtime settings, read from the environment (prefix ``AR_``) or a local ``.env`` file.

Nothing here has a secret default. Settings that only some commands need (the Lever key,
the GCP project) are optional at load time and checked by ``require_*`` when used, so
``criteria lint`` works without credentials.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ThinkingLevel = Literal["low", "medium", "high"]


class ConfigError(RuntimeError):
    """A setting needed for the requested operation is missing or invalid."""


class TierThresholds(BaseModel):
    """Minimum score (inclusive) for each tier; anything below ``c`` is tier D."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    a: int = Field(80, ge=1, le=100)
    b: int = Field(65, ge=1, le=100)
    c: int = Field(45, ge=1, le=100)

    @model_validator(mode="after")
    def _ordered(self) -> TierThresholds:
        if not self.a > self.b > self.c:
            raise ValueError("tier thresholds must satisfy a > b > c")
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AR_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    env: Literal["dev", "test", "prod"] = "dev"

    # --- Lever -------------------------------------------------------------------------
    lever_api_key: SecretStr | None = None
    # https://api.eu.lever.co/v1 for accounts on hire.eu.lever.co. Redirects are never
    # followed on API calls (httpx would drop auth and turn POST into GET).
    lever_base_url: str = "https://api.lever.co/v1"
    # Optional Lever user id that write-backs are attributed to; omitted => the API key's name.
    lever_perform_as: str | None = None
    # The API key may read confidential postings and their opportunities (a Lever admin
    # grants it; docs/legal-hr-questions.md). Off by default: ``postings list`` then shows
    # non-confidential postings only. ``rank`` detects a confidential posting on its own.
    lever_confidential_access: bool = False
    lever_rate_per_second: float = Field(8.0, gt=0, le=10)
    lever_burst: int = Field(15, ge=1, le=20)
    lever_timeout_seconds: float = Field(30.0, gt=0)
    lever_max_retries: int = Field(5, ge=0, le=10)
    # Each Lever webhook has its own signature token: map event name -> token,
    # e.g. AR_LEVER_WEBHOOK_TOKENS='{"applicationCreated": "...", "candidateDeleted": "..."}'
    lever_webhook_tokens: dict[str, SecretStr] = Field(default_factory=dict)
    webhook_max_age_seconds: int = Field(600, ge=30)

    # --- Gemini on Vertex AI -------------------------------------------------------------
    gcp_project: str | None = None
    # gemini-3.1-pro-preview is served only on the global endpoint (no data residency).
    gcp_location: str = "global"
    gemini_model: str = "gemini-3.1-pro-preview"
    gemini_thinking_level: ThinkingLevel = "low"
    gemini_timeout_seconds: int = Field(180, ge=10, le=1800)
    gemini_max_output_tokens: int = Field(16384, ge=1024, le=65536)
    gemini_retry_attempts: int = Field(4, ge=1, le=10)
    # Best-effort reproducibility only; Gemini 3 sampling params are deliberately unset.
    gemini_seed: int | None = 7
    llm_concurrency: int = Field(4, ge=1, le=32)
    # Concurrent Lever reads/downloads, resume preparation and write-backs per run.
    io_concurrency: int = Field(4, ge=1, le=16)

    # --- Write-back ----------------------------------------------------------------------
    # Master switch. Live write-back also needs --mode live, approved criteria and a
    # passing jurisdiction policy for the posting.
    write_enabled: bool = False
    tag_prefix: str = "AI Rank"
    write_score_band_tag: bool = False
    note_secret: bool = False
    note_notify_followers: bool = False
    note_max_chars: int = Field(3000, ge=500, le=10000)

    # --- Files and storage -----------------------------------------------------------------
    criteria_dir: Path = Path("criteria")
    policy_file: Path = Path("config/policy.yaml")
    # None => the packaged default list (applicant_ranking/data/protected_terms.yaml).
    protected_terms_file: Path | None = None
    state_db: Path = Path("data/state.sqlite3")
    # Key for HMAC fingerprints of resumes in the audit log. Required outside dev.
    hmac_key: SecretStr | None = None
    store_resume_text: bool = False
    retention_days: int = Field(1461, ge=365)  # 4 years: the strictest verified floor (CA FEHA)

    # --- Resume handling ---------------------------------------------------------------------
    max_resume_bytes: int = Field(10 * 1024 * 1024, ge=1024)
    max_pdf_pages: int = Field(10, ge=1, le=50)
    max_resume_chars: int = Field(30_000, ge=2_000)
    redact_education_years: bool = True
    extraction_timeout_seconds: float = Field(30.0, gt=0)
    # Run PDF/DOCX parsing in worker processes (PDFium is not thread-safe). Tests turn it off.
    extraction_in_subprocess: bool = True
    extraction_max_workers: int = Field(2, ge=1, le=8)
    vision_max_pages: int = Field(5, ge=1, le=10)

    # --- Scoring -----------------------------------------------------------------------------
    tiers: TierThresholds = Field(default_factory=TierThresholds)
    # |computed score - model's holistic score| at or above this adds an info flag.
    disagreement_threshold: int = Field(30, ge=5, le=100)

    @field_validator("lever_base_url")
    @classmethod
    def _https_no_trailing_slash(cls, v: str) -> str:
        v = v.rstrip("/")
        if not v.startswith("https://"):
            raise ValueError("lever_base_url must be https")
        return v

    def require_lever_key(self) -> str:
        if self.lever_api_key is None or not self.lever_api_key.get_secret_value():
            raise ConfigError("AR_LEVER_API_KEY is not set")
        return self.lever_api_key.get_secret_value()

    def require_gcp_project(self) -> str:
        if not self.gcp_project:
            raise ConfigError(
                "AR_GCP_PROJECT is not set (an Egen GCP project with Vertex AI enabled)"
            )
        return self.gcp_project

    def require_hmac_key(self) -> bytes:
        """Key for audit fingerprints. Dev falls back to a fixed, clearly-labelled key."""
        if self.hmac_key is not None and self.hmac_key.get_secret_value():
            return self.hmac_key.get_secret_value().encode()
        if self.env in ("dev", "test"):
            return b"applicant-ranking-dev-only-hmac-key"
        raise ConfigError("AR_HMAC_KEY is required outside dev")

    def webhook_token(self, event: str) -> str | None:
        token = self.lever_webhook_tokens.get(event)
        return token.get_secret_value() if token is not None else None
