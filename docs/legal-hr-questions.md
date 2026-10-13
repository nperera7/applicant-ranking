# Questions to settle before live use

None of this is legal advice. It turns the 2026-10-08 research into decisions for Egen
Legal, HR, TA and IT. Until they are answered, the service runs only in `evaluation` mode
(watermarked, no write-back), because the policy gate fails closed. Evaluation mode is not
offline: it still reads applications from Lever and sends their resumes to Gemini (redacted
text, or page images for scanned resumes); see question 13. Legal/HR fill in and approve
`config/policy.yaml`, a copy of `config/policy.example.yaml`.

## Laws that may apply (status on 2026-10-08)

| Regime | Status | What it means here |
|---|---|---|
| Illinois HB 3773 (IHRA 2-102(L)) | In force since 2026-01-01; IDHR notice rules withdrawn 2026-06-02 | Applies to Egen (headquartered in Naperville). Prohibits AI with a discriminatory *effect* and ZIP codes as a proxy. Employees and applicants must be told AI is used, and there is no official notice form yet. |
| NYC Local Law 144 (AEDT) | In force | Applies only to NYC-office roles, or remote roles tied to an NYC office. A score or ranking counts as "simplified output". Requires an independent bias audit within the year before use, a published summary, and notice 10 business days before use. |
| Colorado SB 26-189 (replaced SB 24-205) | Decisions on or after 2027-01-01; Attorney General rules in draft | Covers Colorado residents and Colorado-based opportunities. Requires a notice, an adverse-outcome disclosure within 30 days, correction, meaningful human review, and 3-year records. |
| California FEHA ADS regulations | In force since 2025-10-01 | Resume ranking counts as an ADS. Anti-bias testing is evidence for or against a claim. Records must be kept 4 years. |
| California CCPA ADMT regulations | Compliance by 2027-01-01 | Only if Egen meets the CCPA thresholds and the tool "substantially replaces" human decisions. |
| Ontario ESA (Working for Workers Four Act) | Postings from 2026-01-01 | Employers with 25 or more employees must state AI use in public job postings (Mississauga office). |
| Federal (Title VII) | Disparate impact still in the statute; federal enforcement retreating | Private suits continue (e.g. *Mobley v. Workday*). Application records must be kept 1 year, plus legal holds. |
| EU AI Act / UK / India | High-risk under EU Annex III from 2027-12-02; others not assessed | Dry-run and live runs are blocked by default, because the policy only allows US postings; `evaluation` mode still runs and sends their resumes to the model. |

## Decisions needed

### Scope and jurisdictions
1. Which postings go first, and where are those roles (NYC, IL, CO, CA, remote)? The policy
   gate maps each posting's location tags and Lever's structured address to a state or city;
   a US posting with a label it cannot map ("Hudson Yards Office", "HQ") is blocked until the
   Lever admin sets the posting's structured address or Legal/HR list the label, with the
   regimes that apply, under `location_overrides` in `config/policy.yaml`. A New York
   location without a known non-NYC city is treated as New York City.
2. NYC: commission an independent bias audit, or keep NYC-linked roles out of scope? Who
   documents that decision?
3. Does Egen meet the CCPA business thresholds? Is Egen a federal contractor?
4. Keep EU, UK and India roles out until assessed? This is the default.

### Notices and candidate rights
5. Approve the Illinois AI-use notice text and where it appears: job postings, careers page,
   handbook. `illinois.notice_url` stays empty until then.
6. Who handles accommodation and alternative-process requests and data requests (NYC 30 days;
   the Colorado draft rules say 10 business days and 45 days for human review)?
7. Colorado from 2027-01-01: who sends adverse-outcome disclosures, and who is the
   independent human reviewer?

### What the model may see and say
8. Approve the redaction list: names, contact details, URLs, addresses, ZIP, "City, ST",
   DOB/age, gender, marital status, nationality, citizenship and visa lines, religion,
   disability, graduation years. Should school names or affinity organisations also be
   removed?
9. Approve the protected-term list in `src/applicant_ranking/data/protected_terms.yaml`.
   It blocks those terms in criteria and in model output.
10. Work authorisation and sponsorship: confirm they come from a structured Lever application
    question, never from the resume. The linter already rejects such knockouts.

### Criteria governance
11. Confirm that the hiring manager writes the per-role criteria, and decide who approves
    them (TA or HR). What job-relatedness evidence is required? Should degree and
    years-of-experience requirements get a disparate-impact review?
12. If criteria change mid-posting, are existing applicants re-scored? The tool re-scores
    them on the posting's next `rank` run after the criteria content changes.

### Data handling and the model provider
13. Is Gemini on Vertex AI approved for candidate personal data? `gemini-3.1-pro-preview` is
    a **Public Preview** model served only on the **global** endpoint, with **no data-residency
    guarantee**. Is that acceptable?
14. Egen's Google Cloud agreement type. Customers under a Cloud Master Agreement are exempt
    from abuse-monitoring prompt logging (otherwise up to 90 days). Should a GCP admin disable
    the 24-hour in-memory cache (`cacheConfig disableCache`, needs `roles/aiplatform.admin`)?
15. Retention: the default is 4 years for audit records, with legal-hold override. Approve it.
16. When Lever deletes or anonymises a candidate, PII is purged and a pseudonymous audit row is
    kept. Is that right, or should the audit row also be deleted?

### Monitoring
17. Who runs the adverse-impact analysis (HR analytics or an independent auditor) on
    `export-audit` joined to Lever EEO self-ID data, in a separate project with a separate
    key? How often, with what thresholds, and what happens when one is crossed?
18. What score and tier visibility is allowed: recruiters only, or hiring managers and
    interviewers too? Is Lever's own Talent Fit enabled on the same postings?

## Lever and GCP prerequisites (IT / Lever admin)

- **Lever plan:** API-key access and manually configured webhooks require the paid "Data API"
  and "Webhook" features. Confirm Egen has both.
- **Dedicated API key**, named e.g. "Applicant Ranking (AI)", with:
  - Read: postings, opportunities, resumes (list/read/download), users, stages, archive reasons, deleted opportunities (`GET /opportunities/deleted`, used to reconcile missed `candidateDeleted` webhooks).
  - Write: add/remove tags, create/update notes.
  - No stage, archive or EEO access.
  - **Decision needed: confidential-data access.** Without it, confidential postings are
    invisible to `postings list` and `rank --posting <id>` fails on `GET /postings/{id}`. With it,
    `rank` lists a confidential posting's applicants automatically (the client sends
    `confidentiality=all` once the posting read has shown it to be confidential) and
    `postings list --include-confidential` (or `AR_LEVER_CONFIDENTIAL_ACCESS=true`) shows
    the postings. Record the decision here and in `.env`.
- **Webhooks:** `applicationCreated` and `candidateDeleted`, pointing to the HTTPS service URL.
  Each has its own signature token, set in `AR_LEVER_WEBHOOK_TOKENS`.
- **Region:** confirm Egen's Lever account is on `hire.lever.co`, not the EU instance (which
  uses `api.eu.lever.co`).
- **GCP:** an Egen project with Vertex AI enabled, a service account with
  `roles/aiplatform.user`, and no org policy blocking global endpoints.
