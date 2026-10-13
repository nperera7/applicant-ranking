"""Static checks on the Dockerfile, .dockerignore and .env.example."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import BaseModel, SecretStr

from applicant_ranking.config import Settings

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
ENV_EXAMPLE = ROOT / ".env.example"

# Documented with a placeholder rather than the default value.
PLACEHOLDERS = {"lever_webhook_tokens", "lever_perform_as", "protected_terms_file"}


def _instructions(text: str) -> list[str]:
    """Dockerfile instructions with continuation lines joined and comments dropped."""
    joined = re.sub(r"\\\n\s*", " ", text)
    return [line.strip() for line in joined.splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def _env_names(field: str, annotation: object) -> list[str]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [f"AR_{field.upper()}__{sub.upper()}" for sub in annotation.model_fields]
    return [f"AR_{field.upper()}"]


def _example_lines() -> dict[str, str]:
    """AR_NAME -> value for every line, commented or not."""
    lines: dict[str, str] = {}
    for raw in ENV_EXAMPLE.read_text().splitlines():
        match = re.match(r"^#?\s?(AR_[A-Z0-9_]+)=(.*)$", raw)
        if match:
            lines[match.group(1)] = match.group(2)
    return lines


def test_dockerfile_base_uv_and_sync() -> None:
    lines = _instructions(DOCKERFILE.read_text())
    froms = [line for line in lines if line.startswith("FROM ")]
    assert froms and all(line.split()[1] == "python:3.13-slim" for line in froms)
    assert any(re.search(r"COPY --from=ghcr\.io/astral-sh/uv:\d+\.\d+\.\d+ /uv\b", line)
               for line in lines)
    assert any("uv sync --frozen --no-dev" in line for line in lines)


def test_dockerfile_runs_as_non_root() -> None:
    users = [line.split()[1] for line in _instructions(DOCKERFILE.read_text())
             if line.startswith("USER ")]
    assert users, "no USER instruction"
    final = users[-1].split(":")[0]
    assert final not in ("root", "0")


def test_dockerfile_bakes_in_no_secrets() -> None:
    lines = _instructions(DOCKERFILE.read_text())
    for line in lines:
        if line.startswith(("ENV ", "ARG ")):
            assert not re.search(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", line), line
        if line.startswith("COPY "):
            assert ".env" not in line, line
            assert "credentials" not in line.lower(), line


def test_dockerfile_cmd_serves_on_port() -> None:
    [cmd] = [line for line in _instructions(DOCKERFILE.read_text()) if line.startswith("CMD ")]
    assert "applicant-ranking serve --host 0.0.0.0 --port" in cmd
    assert "${PORT:-8080}" in cmd


def test_dockerignore_excludes_local_state_and_secrets() -> None:
    patterns = {line.strip() for line in DOCKERIGNORE.read_text().splitlines()
                if line.strip() and not line.startswith("#")}
    for required in (".venv", "data/", "reports/", ".env*", "tests", "docs/*.pdf", ".git"):
        assert required in patterns, required
    # The packaged lexicon lives in src/applicant_ranking/data and must stay in the context.
    for pattern in patterns:
        assert not pattern.startswith(("src", "**/data", "*/data")), pattern
    for needed in ("pyproject.toml", "uv.lock", "README.md", "criteria", "config"):
        assert needed not in patterns


def test_env_example_lists_every_setting() -> None:
    documented = _example_lines()
    expected = [name for field, info in Settings.model_fields.items()
                for name in _env_names(field, info.annotation)]
    missing = [name for name in expected if name not in documented]
    assert not missing
    unknown = set(documented) - set(expected)
    assert not unknown


def test_env_example_loads_with_no_secrets_set(tmp_path: Path) -> None:
    settings = Settings(_env_file=ENV_EXAMPLE)

    for name in ("lever_api_key", "hmac_key"):
        value = getattr(settings, name)
        assert value is None or (isinstance(value, SecretStr)
                                 and value.get_secret_value() == "")
    assert settings.gcp_project in (None, "")
    assert settings.write_enabled is False
    assert settings.lever_webhook_tokens == {}


def test_env_example_documented_defaults_are_the_real_defaults(tmp_path: Path) -> None:
    lines = []
    for raw in ENV_EXAMPLE.read_text().splitlines():
        match = re.match(r"^#?\s?(AR_[A-Z0-9_]+=.*)$", raw)
        if match:
            lines.append(match.group(1))
    uncommented = tmp_path / "all.env"
    uncommented.write_text("\n".join(lines) + "\n")

    loaded = Settings(_env_file=uncommented)
    defaults = Settings(_env_file=None)

    for field in Settings.model_fields:
        if field in PLACEHOLDERS or field in ("lever_api_key", "gcp_project", "hmac_key"):
            continue
        assert getattr(loaded, field) == getattr(defaults, field), field


@pytest.mark.parametrize("path", [DOCKERFILE, DOCKERIGNORE, ENV_EXAMPLE])
def test_no_real_looking_secrets(path: Path) -> None:
    text = path.read_text()
    assert not re.search(r"AIza[0-9A-Za-z_-]{20,}", text)  # Google API key
    assert not re.search(r"\b[0-9a-f]{40,}\b", text)  # long hex tokens
    assert "egen.ai" not in text
