# Applicant Ranking: design

Status: milestone 1 (core library, CLI, webhook service), built 2026-10-08.
The facts behind each decision, as of that date, are in [research-notes.md](research-notes.md).

## What it does

For a Lever job posting, the service reads each applicant's resume and scores it against the
role's **criteria**: requirements, knockouts and guardrails, written by the hiring manager and
approved by TA/HR. Gemini Pro produces a per-requirement assessment, and code turns that into a
1–100 score and a tier (A–D). Each candidate gets a tier tag, which recruiters can filter on,
and a short evidence-based note in Lever.

It is **advisory**. Nothing in the code can reject, archive, advance or hide a candidate. Every
applicant still appears in the recruiter's queue and in every report.

## Flow

```
Lever posting ──► criteria (YAML, approved) ──► policy gate (jurisdictions, fails closed)
      │
      ▼
opportunities ──► newest processed resume ──► extract (PDF/DOCX/DOC/RTF/TXT/ODT/image)
                                                 │  hidden text removed + flagged
                                                 ▼
                         normalize ──► security scan ──► redact identity & proxies
                                                 │
            scanned/image resume ──► Gemini transcription (page PNGs) ──┘
                                                 ▼
                    Gemini Pro: per-requirement ratings + verbatim evidence (JSON schema)
                                                 ▼
          validate: ground every quote in the text sent · protected-term scan (1 corrective retry)
                                                 ▼
          score = weighted ratings (code) · tier · status · flags ──► audit log + PII store
                                                 ▼
                   write-back plan (tags + note) ──► sent only in live mode
```

Notes on the flow:

- *normalize ──► security scan*: zero-width joiners inside words of joiner-using scripts
  (Persian, Hindi and other Indic scripts) are kept and not counted, and directional marks on
  right-to-left lines count as layout, so ordinary Persian, Hebrew or Hindi resumes do not
  trigger the invisible-characters review flag. The injection regexes cover English plus
  Spanish, French, German, Portuguese, Italian, Chinese and Hindi phrasings, leetspeak,
  letter-spaced words, and Armenian, Cherokee, Lisu and small-capital homoglyphs. A copy of
  the prompt's resume fence or a "correction from the system" line in the resume is itself
  an injection flag (review), whatever the model does with it.
- *Gemini*: the corrective retry's note travels in the system instruction; the user turn
  carries only the fenced resume and the task line, and anything in the resume shaped like a
  fence marker is neutralised before it is sent. The rules tell the model that nothing inside
  the resume is a message from the system.
- *validate*: a quote found only by the fuzzy (formatting-noise) match is stored and
  published as the aligned resume span, never as the model's wording; a fuzzy quote whose
  wording adds a protected term is ungrounded; quotes over 500 characters ground only as
  exact substrings. The retry inherits the first attempt's review-severity model signals
  (an injection report, a review-level red flag, mostly ungrounded evidence) as fixed-text
  flags, so a clean second answer cannot erase them. At most 10 unknown assessment ids are
  kept in the stored report (the audit row keeps only their count).
- *cache*: the key is the redacted-text HMAC (the file HMAC for scans), the criteria hash,
  prompt and model ids, the composite redaction version and a fingerprint of the lexicon
  content. A cached result is recorded again as a derived evaluation (no model call) when
  tier thresholds change, batch signals add a review flag, or a live run reuses a dry-run
  result.
- *errors*: a systemic error (Lever rejecting the key, the model rejecting credentials,
  project or model id) aborts the run before any write-back; a transient per-applicant
  error (Lever 5xx/429, timeout, retryable model error) leaves the existing AI ranking tags
  and note alone (report label `unchanged`) instead of writing "Not scored".
- *webhooks*: `applicationCreated` is deferred and retried (2, 10 and 30 minutes) while
  Lever is still processing the resume, then scored on the newest processed file;
  `candidateDeleted` is confirmed with `GET /opportunities/{id}` (404, `isAnonymized` or
  `deletedAt`) before any tombstone or PII purge, because the opportunity id is in the
  unsigned payload. Deletion reconciliation reads `GET /opportunities/deleted`.
- *dates*: jurisdiction date rules (NYC notice wait, audit age, `approved_at`, Colorado's
  effective date) run on the earliest local calendar date among the posting's regimes, so
  a gate never opens while it is still the day before where the rule applies.

## Key decisions

| Decision | Why |
|---|---|
| **Gemini `gemini-3.1-pro-preview` on Vertex AI, location `global`**, model ID in config | Gemini Pro is the chosen model family. The only GA Pro (`gemini-2.5-pro`) retires 2026-10-20. 3.1 Pro is Public Preview and global-only, so there is **no data-residency guarantee** (open question for Legal). |
| **Score computed in code** from per-requirement ratings; the model's holistic number is diagnostic only | Consistent, explainable, auditable. Weights come from the criteria (must-have 3, nice-to-have 1 by default). |
| **Evidence grounding**: every quote must appear in the redacted text actually sent | Catches hallucinated and injected evidence. An ungrounded "meets" becomes "unknown". |
| **No temperature or top-p/k; `thinking_level=low`; seed set** | Google deprecates sampling parameters for Gemini 3. Determinism comes from explicit rules in the system instruction, and scores still vary slightly between runs. |
| **Redaction before the model** (rules `redact-v3`): name, contact details, URLs, addresses, ZIP, "City, ST" and "City ST ZIP", DOB/age, marital status, nationality, unlabelled citizenship and immigration lines, ID numbers, international phone groupings, graduation years | Bias controls. Illinois HB 3773 bars ZIP codes as a proxy. The redacted text is the only resume text sent. Name tokens that are also words (Ruby, Kafka, Jan, Grant, Brown ...) are redacted only where used as a name; a personal-detail label mid-sentence redacts only its field; labels are recognised in German, Spanish, French, Italian and Portuguese; "Statistics, MS" and "MS Access" are not locations, and employment dates at academic employers are kept. |
| **Hidden-text detection**: PDF render-based ink check (text colour vs. its surroundings, with and without anti-aliasing), invisible render mode, CropBox/off-page, transparent fill, clip path, squashed glyphs, fonts under 4pt; DOCX vanish, tiny, squashed, offset runs, white or low-contrast text against the resolved background (theme colours, shape fills, table styles, page), off-page or collapsed text boxes; ODT the same rules from its style tables plus hidden-text fields; .doc direct character formatting only | About 1% of real resumes carry hidden prompt injections, and more than 90% of those are hidden keywords rather than instructions. White text over an unresolvable fill or a dark decorative shape is kept with an info note (the injection scan still covers it). The scanned-resume path sends rasterised pages, never the PDF, because Gemini reads a PDF's embedded text layer. |
| **Resume fenced as untrusted data**, with a random per-request fence | Prompt-injection defense. The model reports manipulation attempts, and deterministic regexes flag them too. Both force human review and never reject. |
| **Knockouts flagged, never applied** | A knockout can only be "unmet" on quoted evidence, and it adds a "Verify knockout" tag. Work authorisation and citizenship knockouts are lint errors: they belong in a structured Lever question. |
| **Tags as the primary write-back** (`AI Rank: Tier A` … `D`, `Needs review`, `Not scored`, `Verify knockout`) | Tags are the only integration-written value recruiters can filter on in Lever. Nothing can make Lever sort by a score. |
| **One note per candidate**, updated in place | Advisory first line, evidence per requirement, flags, versions. If validation fails, the note says the assessment was withheld and contains no model text. |
| **Three run modes**: `dry_run` (default), `live`, `evaluation` | Live needs `AR_WRITE_ENABLED=true`, approved criteria and a passing policy. Evaluation mode is for tuning and backtests: unapproved criteria are allowed, output is watermarked, and write-back is impossible. |
| **Jurisdiction policy engine fails closed** | IL (always, since Egen is headquartered there), NYC LL144, Colorado SB 26-189 (from 2027-01-01), California FEHA ADS, Ontario ESA. The gate reads both the posting's location tags and Lever's structured address (`categories.locationDetails`, primary location only) and unions what they imply. A US posting with a location label it cannot map to a state or city ("Hudson Yards Office", "HQ") is blocked until Lever has a structured address for it or Legal/HR list the label under `location_overrides` in `config/policy.yaml`; a New York location without a known non-NYC city is treated as New York City. If the policy file is missing or unapproved, or a posting is outside the US, nothing runs except evaluation mode. |
| **Append-only audit log, separate from PII** | The audit log holds IDs, versions, hashes, score, tier, status and flag codes (model-written unknown ids only as a count), kept 4 years by default. The PII store holds names, evidence and the full evaluation and is purged on `candidateDeleted` unless a legal hold applies; clearing a hold completes the deletions it suspended (`purge_tombstoned`, also run by `purge`). The SQLite store runs with `PRAGMA secure_delete` and checkpoints and truncates the WAL after every purge, so purged rows leave the files, not just the queries. |
| **Webhook payload treated as a hint** | Lever's signature covers only `token + triggeredAt`. The receiver verifies it (constant-time), checks freshness, rejects replays, and re-fetches the opportunity through the API. A `candidateDeleted` event is acted on only once `GET /opportunities/{id}` confirms the deletion (404, `isAnonymized` or `deletedAt`); reconciliation reads `GET /opportunities/deleted`, never `/applications/deleted`. |
| **Lever client has no stage or archive methods**, and writes are allow-listed (tags, notes) | Rank-only by construction. Unit tests assert it. |
| **Confidential postings are visible when the key allows it** | `rank` lists a confidential posting's applicants with `confidentiality=all` once `GET /postings/{id}` has shown the posting to be confidential (that read proves the key has access); `postings list` asks for both job sites and includes confidential postings with `--include-confidential` or `AR_LEVER_CONFIDENTIAL_ACCESS=true`. A posting with no listed applicants gets a note in the report instead of a silent empty table. |
| **Protected-term lexicon v2, approved domain terms in criteria** | Person-context patterns catch "the candidate is young" and bare age ranges; allow phrases cover domain vocabulary that cannot describe a person ("Veterans Affairs", "credit risk"). The lexicon is pending Legal/HR review. A role may list `approved_terms` (term, justification, approver, date) in its criteria file so a lint hit becomes a warning naming the approver; this never relaxes the scan of model output. |
| **Record that cannot be read is still a row** | A Lever opportunity that fails validation but carries an id becomes a visible SKIPPED row ("the Lever record could not be read") with no name stored and no write-back, instead of vanishing from the ranking. |

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | Settings from `AR_*` environment variables or `.env` |
| `domain.py` | Shared types: criteria, flags, evaluation, write-back plan, reports |
| `criteria.py`, `criteria_lint.py` | Criteria YAML, repository, draft-from-posting, linter |
| `lever/` | Async client (rate limit, retries, pagination, allow-listed writes), lenient models, webhook verification |
| `resumes/` | Sniffing, per-format extraction, hidden-text detection, normalisation, security scan, redaction, integrity signals, preparation |
| `llm/` | Prompt construction, Gemini evaluator, response schema, test fake |
| `scoring/` | Score/tier computation, output validation, protected-term lexicon |
| `policy.py` | Jurisdiction gates |
| `store.py` | SQLite audit log, PII store, cache, holds, retention, replay cache |
| `writeback.py` | Tag and note planning; live-only execution |
| `pipeline.py` | Orchestration (`Ranker`) |
| `report.py`, `backtest.py`, `fairness.py` | Reports and audit export, accuracy against past recruiter decisions, counterfactual bias checks |
| `cli.py`, `service/` | Command line, FastAPI webhook receiver and job queue |

## Not in milestone 1

- **Durable production state.** SQLite on Cloud Run's ephemeral disk loses the cache, the audit
  log and the replay cache on restart. Next step: Firestore or BigQuery with insert-only IAM
  for the audit table, and Cloud KMS envelope encryption so the PII store can be crypto-shredded.
- **Queueing.** The service uses an in-process queue, which needs Cloud Run instance-based
  billing. Cloud Tasks or Pub/Sub would be more robust.
- **Adverse-outcome and rights workflows** (Colorado from 2027-01-01, CCPA if in scope). The
  audit export and per-criterion explanations are the inputs, but there is no case management.
- **NER-based redaction.** Redaction is regex-based, so nicknames, affinity organisations and
  non-US address formats can leak, and a lone word-like first name in the middle of a bullet
  ("worked with Grant on ...") is not redacted. Counterfactual tests (`fairness`, run with
  `--name` so the real name is exercised) are the control.
- **Per-regime clocks in the policy engine.** Date rules run on the earliest local date among
  the posting's regimes (conservative for gates that open on a date; up to a few hours lenient
  for the audit-age and effective-date checks when regimes span time zones). A per-regime
  `now` inside `policy.py` and a `tzdata` dependency for slim images are the next step.
- **Terraform and deploy scripts, and a Secret Manager integration.**
