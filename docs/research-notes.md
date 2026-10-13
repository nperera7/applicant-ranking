# Research notes (verified 2026-10-08)

The facts the design depends on, with their sources. Nothing here comes from authenticated
calls: no Lever or Google Cloud credentials were used.

## Lever Data API v1

Source: <https://hire.lever.co/developer/documentation> (anchors below); EU instance:
<https://hire.eu.lever.co/developer/documentation>.

- **Base URL and auth** (`#introduction`, `#basic-auth`): `https://api.lever.co/v1`, or
  `https://api.eu.lever.co/v1` for EU accounts. Authenticate with HTTP Basic: the API key is
  the username and the password is blank. Send it pre-emptively, because a 401 carries no
  challenge.
- **Pagination** (`#pagination`): `{data, next, hasNext}`. Pass `next` back as `offset`.
  `limit` is 1–100.
- **Rate limits** (`#rate-limits`): 10 requests/second per key, with bursts to 20. Overload
  returns 429; back off exponentially. `Retry-After` is not documented.
- **`include`** (`#customizing-the-response`) *restricts* the response to the named fields.
  Use `expand` instead.
- **Applicants for a posting** (`#list-all-opportunities`):
  `GET /opportunities?posting_id=…&expand=applications`. The posting ID is only on
  `applications[0].posting`.
- **Resumes** (`#resumes`, `#download-a-resume-file`):
  `GET /opportunities/{id}/resumes`, then `…/resumes/{rid}/download`. A 422 means Lever could
  not process the file. `file.downloadUrl` is a web-app URL and must not be used.
- **Tags** (`#update-opportunity-tags`): `POST …/addTags` and `…/removeTags` with
  `{"tags": [...]}`. `perform_as` is optional. Tags are the only integration-written value the
  pipeline UI can filter on (<https://help.lever.co/s/article/Filtering-the-candidate-pipeline>),
  and nothing lets an integration sort candidates by a score.
- **Notes** (`#notes`): `POST …/notes` with `value`, `secret`, `notifyFollowers` and `score`
  (score is Lever's 1–4 thumbs scale, which this tool never sets). Notes created through the API
  can be updated with PUT.
- **Profile forms** (`#profile-forms`): create-only, no number field type, not filterable. Not
  used.
- **Webhooks** (`#request-signing`, `#receiving-webhooks`, `#responding-to-a-webhook`): the
  signature is `hex(HMAC-SHA256(signature_token, token + triggeredAt))`. It does **not** cover
  `data`. Any 2xx acknowledges delivery; anything else is retried 5 times. HTTPS only.
- **Plan features** (<https://hire.lever.co/developer/partner>): API keys and manual webhooks
  require the paid "Data API" and "Webhook" features.
- **No customer sandbox:** `api.sandbox.lever.co` is only for partner OAuth apps.

## Gemini on Vertex AI (now "Gemini Enterprise Agent Platform")

- **Model versions** (<https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions>):
  - `gemini-2.5-pro` is the only GA Pro model and retires on **2026-10-20**.
  - `gemini-3.1-pro-preview` is in Public Preview (released 2026-02-19), with no retirement
    date published.
- **3.1 Pro** (<https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-1-pro>):
  - Served on location **`global` only**.
  - 1,048,576-token context window and 65,536 maximum output tokens.
  - Documents must be PDF or plain text (no DOCX).
  - Thinking is set with `thinking_level` (LOW, MEDIUM or HIGH) and cannot be turned off.
- **Sampling** (<https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/migrate>):
  temperature, top_p and top_k are deprecated for Gemini 3; Google says to get determinism from
  explicit system-instruction rules. `thinking_budget` is deprecated, and sending it together
  with `thinking_level` returns 400.
- **Structured output** (<https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/control-generated-output>):
  Vertex enforces only enum, min/max, items/maxItems/minItems, nullable, properties, required
  and propertyOrdering, so string lengths are ignored. Don't repeat the schema in the prompt.
- **SDK** (google-genai 2.29.0 source):
  - `HttpOptions.timeout` is in milliseconds.
  - Retries are off unless `HttpRetryOptions` is passed.
  - `response.parsed` is silently None on invalid JSON, MAX_TOKENS truncation, a validation
    failure or a block.
  - An `IntEnum` field is rejected on the client side.
- **Data** (<https://docs.cloud.google.com/gemini-enterprise-agent-platform/resources/zero-data-retention>,
  `/resources/data-residency`):
  - Google does not train on customer data, preview models included.
  - Abuse-monitoring logs can be kept up to 90 days; Cloud Master Agreement customers are exempt.
  - The 24-hour in-memory cache can be disabled per project.
  - The global endpoint gives **no residency guarantee**.
- **Price** (<https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing>):
  3.1 Pro on global costs $2 input and $12 output per 1M tokens up to 200K context. Thinking
  tokens are billed as output. That works out to roughly $0.02–0.04 per resume at LOW thinking,
  an estimate not yet measured.

## Resume parsing and integrity

- **PDF:** pdfminer.six (MIT; ≥ 20251230 fixes CVE-2025-64512 and CVE-2025-70559) was the only
  extractor that kept multi-column reading order. PyMuPDF is AGPL, so it was rejected.
- **Hidden text:**
  - No extraction library drops hidden text.
  - A pypdfium2 render-based ink check catches white-on-white, tiny, covered and off-page text.
  - Text render mode 3 catches invisible text drawn over existing ink.
  - Map coordinates through the CropBox.
- **Gemini reads a PDF's embedded text layer**
  (<https://ai.google.dev/gemini-api/docs/document-processing>), so scans go to the model as
  rasterised PNGs, never as the PDF.
- **DOCX:** python-docx's high-level API drops text boxes, tracked insertions and content
  controls, so the code walks the XML.
- **Prevalence:** about 1% of about 200K real resumes contain hidden prompt injections, and more
  than 90% of those are hidden keywords rather than instructions
  (<https://arxiv.org/abs/2605.28999>, USENIX Security 2026).
- **Defenses:**
  - Spotlighting (delimiting untrusted text) cut injection success from over 50% to under 2%
    (<https://arxiv.org/abs/2403.14720>).
  - OWASP LLM01:2025 describes a resume payload-splitting attack
    (<https://genai.owasp.org/llmrisk/llm01-prompt-injection/>).
- **Name bias:** retrieval models preferred white-associated names in up to 85.1% of
  comparisons (<https://www.arxiv.org/pdf/2407.20371>). That is the reason for redaction.

## Hiring-AI law (summary; see legal-hr-questions.md)

- **Illinois HB 3773 / PA 103-0804:** <https://ilga.gov/Legislation/publicacts/view/103-0804>.
- **NYC LL144 rules and FAQ:**
  - <https://rules.cityofnewyork.us/wp-content/uploads/2023/04/DCWP-NOA-for-Use-of-Automated-Employment-Decisionmaking-Tools-2.pdf>
  - <https://www.nyc.gov/assets/dca/downloads/pdf/about/DCWP-AEDT-FAQ.pdf>
- **Colorado SB 26-189:** <https://leg.colorado.gov/bills/sb26-189>. Attorney General draft
  rules: <https://coag.gov/ai/>.
- **California FEHA ADS regulations:**
  <https://calcivilrights.ca.gov/wp-content/uploads/sites/32/2025/06/Notice-of-Approval-regulations-automated-employment-decision-systems.pdf>.
  CCPA ADMT: <https://cppa.ca.gov/regulations/ccpa_updates.html>.
- **Federal record retention:**
  <https://www.ecfr.gov/current/title-29/subtitle-B/chapter-XIV/part-1602/section-1602.14>.
  Four-fifths rule:
  <https://www.ecfr.gov/current/title-29/subtitle-B/chapter-XIV/part-1607/section-1607.4>.
- **EU AI Act Annex III:** <https://artificialintelligenceact.eu/annex/3/>. The Digital Omnibus
  moved its obligations to 2027-12-02.
