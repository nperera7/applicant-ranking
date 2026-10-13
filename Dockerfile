# Applicant Ranking webhook service (Cloud Run). Build from the repository root:
#   docker build -t applicant-ranking .
# No secrets or .env are baked in: configuration comes from AR_* environment variables
# (Secret Manager on Cloud Run), and Google credentials from the runtime service account.

FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies first (cached unless the lock changes), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable


FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="applicant-ranking" \
      org.opencontainers.image.description="Advisory LLM ranking of Lever applications (Egen internal)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    AR_ENV=prod

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app \
        --shell /usr/sbin/nologin app

WORKDIR /app

# Code and configuration stay root-owned (read-only for the service user). Approved
# criteria and the Legal-approved config/policy.yaml ship with the image; without a
# policy.yaml the policy engine fails closed (DRY_RUN/LIVE blocked).
COPY --from=builder /app/.venv /app/.venv
COPY criteria ./criteria
COPY config ./config
# SQLite state (ephemeral on Cloud Run; see docs/design.md) is the only writable path.
RUN mkdir -p /app/data && chown app:app /app/data

USER 10001:10001

EXPOSE 8080

# One process: the webhook job queue is in-process. Cloud Run sets PORT.
CMD ["/bin/sh", "-c", "exec applicant-ranking serve --host 0.0.0.0 --port \"${PORT:-8080}\""]
