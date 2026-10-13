# Applicant Ranking (Lever ATS + LLM)

Egen internal project for Talent Acquisition, started 2026-10-08. The requirements are
in [docs/requirements.md](docs/requirements.md).

- **Main contact:** Mike Brinn
- **Users:** Egen Talent Acquisition (recruiters)

## Status: milestone 1 built (2026-10-08)

Core library, CLI and webhook service, with about 2,500 tests run against mocks (2,518 on 2026-10-12). It has
not yet run against the real Lever or Gemini APIs: that needs an Egen GCP project and a
Lever API key (see [Before first real run](#before-first-real-run)).

- [docs/requirements.md](docs/requirements.md): what it must do
- [docs/design.md](docs/design.md): how it works and why
- [docs/legal-hr-questions.md](docs/legal-hr-questions.md): decisions needed before live use
- [docs/research-notes.md](docs/research-notes.md): verified facts and sources

### Quick start

```bash
uv sync
```

```bash
uv run pytest -q
```

```bash
uv run applicant-ranking doctor
```

```bash
uv run applicant-ranking criteria lint
```

Commands: `criteria lint|draft`, `postings list`, `rank`, `evaluate`, `evaluate-file`,
`backtest`, `fairness`, `export-audit`, `purge`, `hold`, `doctor`, `serve`. Run
`uv run applicant-ranking --help` for the full list. Configuration uses `AR_*`
environment variables; every one is documented in `.env.example`.

Safety model: the default mode is `dry_run`, which plans the write-back but never sends
it. Both `dry_run` and `live` need approved criteria and an approved `config/policy.yaml`
that passes for the posting's jurisdictions; `live` also needs `AR_WRITE_ENABLED=true`.
The policy gate reads
both the posting's location tags and Lever's structured address; a US posting with a
location label it cannot map to a state or city is blocked until Lever has an address for
it or Legal/HR list the label under `location_overrides`, and a New York location without a
known non-NYC city counts as New York City. `evaluation` mode is for tuning and backtests:
its output is watermarked and it can never write to Lever. Nothing can reject, archive,
advance or hide a candidate.

### Before first real run

1. **GCP:** pick an Egen project with Vertex AI enabled and set `AR_GCP_PROJECT`. Make
   sure application-default credentials belong to `nelma.perera@egen.ai`:

   ```bash
   gcloud auth login nelma.perera@egen.ai --update-adc
   ```

2. **Lever:** create a dedicated API key with only the permissions listed in
   docs/legal-hr-questions.md. Set `AR_LEVER_API_KEY` in `.env`, which is git-ignored.
3. **Criteria:** draft from a posting with `criteria draft --posting <id>`, have the hiring
   manager refine it, try it on sample resumes with `evaluate-file`, then have TA or HR
   approve it.
4. **Legal/HR:** copy `config/policy.example.yaml` to `config/policy.yaml` and get it
   completed and approved. Until then only `evaluation` mode runs.

## Goal

Rank the applications to a Lever job posting against criteria written for the role, so
recruiters can start with the strongest candidates. An LLM rates each resume against the
criteria, code turns those ratings into a 1–100 score and an A–D tier, and the result goes
back to Lever as tags and one advisory note. Recruiters make every decision. What the
service must do is in [docs/requirements.md](docs/requirements.md); how it does it is in
[docs/design.md](docs/design.md).

## Decisions and open questions

Decided on 2026-10-08:
- **LLM provider:** Gemini Pro on Vertex AI (`gemini-3.1-pro-preview`, the only Pro model
  after `gemini-2.5-pro` retires on 2026-10-20). It is a Public Preview served only on the
  global endpoint, so there is no data-residency guarantee; Legal still needs to approve it.
- **Write-back:** tags that recruiters can filter on in Lever (the tier, and markers for
  applications that need a human check), plus one advisory note per application. Lever
  cannot sort by an integration-written score.
- **Measuring ranking quality:** `applicant-ranking backtest` compares the ranking with past
  recruiter decisions (AUC, precision@k, positive rate per tier) before any live use.
- **Where it runs:** a CLI for batch ranking, plus a webhook service for Cloud Run.

Still open (details in [docs/legal-hr-questions.md](docs/legal-hr-questions.md)):
- What will the per-role criteria contain, and which postings go first? The format is
  `criteria/*.yaml`; see the example file.
- Legal/HR: Illinois notice (in force), NYC LL144 scope, Colorado from 2027-01-01, the
  redaction and protected-term lists, retention, and approval of Vertex AI for resumes.
- Lever: a dedicated API key with the minimal permissions, whether it may read confidential
  postings, webhook signature tokens, and confirmation that the plan includes the Data API
  and Webhook features.

## Environment

This folder is wired to the Egen identity: git commits as `nelma.perera@egen.ai`
and GitHub as `nperera7` (via `~/.gitconfig-egen`). direnv pins gcloud to the
`egen` configuration of the shared `~/.config/gcloud` store and gh to `~/.config/gh-egen`. GitHub pushes refuse to run
until `nperera7` is logged into that gh store.
