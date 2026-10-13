# Applicant Ranking: requirements

What the service must do and the limits it works within. How it meets them is in
[design.md](design.md). Decisions that Legal, HR, Talent Acquisition (TA) and IT still
have to make are in [legal-hr-questions.md](legal-hr-questions.md).

## Objective

Give Egen's Talent Acquisition team a ranked view of the applications to each Lever job
posting. A small service uses an LLM to compare every application with criteria written
for the role, so recruiters can start with the strongest candidates.

## People

- **Main contact:** Mike Brinn
- **Users:** Egen Talent Acquisition recruiters

## Scope

In scope:
- Lever job postings and the applications they receive.
- For each application: a score, a place in the posting's ranking, and an advisory
  write-back to Lever.

Out of scope:
- Any decision about a candidate. The service never rejects, archives, advances or hides
  anyone; recruiters decide.
- Adding model-generated detail to a job description. A model has no source for what the
  posting leaves out and could misdescribe the role; the role criteria (R5) supply those
  specifics instead.

## Functional requirements

### Ingestion
- **R1.** Pick up each new application to a chosen Lever posting, with its resume, when a
  Lever webhook announces it or when a batch ranking run lists the posting's applications
  through the Lever API.
- **R2.** Through the Lever API, look up postings and the applications they receive, and
  download each applicant's resume.
- **R3.** Turn each resume into plain text before it is evaluated.

### Role criteria
- **R4.** Evaluate each application's resume against the criteria for its posting's role:
  requirements (each a must-have or a nice-to-have), guardrails (extra job-related rules
  for the evaluation) and knockouts (hard requirements).
- **R5.** The hiring manager writes the criteria for each role, and TA or HR approve them
  before live use. The criteria start from the job description and add the role-specific
  detail that a ranking needs and the job description lacks.

### Evaluation
- **R6.** Send the role's criteria and the redacted resume text to the LLM through its API.
  Names, contact details and proxies for protected characteristics, such as ZIP codes and
  graduation years, are removed first; Legal and HR still have to approve that list. A
  scanned resume is the exception: its page images go to the LLM to be transcribed, and
  the transcript is redacted before scoring.
- **R7.** Give the LLM a fixed prompt template that tells it to rate each requirement
  only on evidence quoted from the resume, mark a knockout as unmet only on explicit
  evidence, and treat the resume as data, never as instructions.
- **R8.** Have the LLM answer in JSON that matches a fixed schema; an answer that does not
  match is not used. Each evaluation reports a score, a status, the main strengths and any
  red flags. The service computes the score and the status itself instead of taking them
  from the model.

### Scoring and ranking
- **R9.** Score each application from 1 to 100 and assign it a tier from A to D. Its
  result also lists, in a few bullet points, where the resume meets the criteria and
  where it falls short.
- **R10.** Rank a posting's applications by score, strongest first. Archived applications
  are left out by default; every other application appears in the ranking, including
  those that could not be scored.

### Write-back
- **R11.** Record each application's result in Lever as tags, which recruiters can filter
  on, and one advisory internal note that is updated in place. The tags give the tier and
  mark applications that need a human check; the note gives the tier, the score and the
  evidence behind them.
- **R12.** Through the Lever API, write only tags and notes. The service cannot change an
  application's stage or archive it, and the Lever API key it uses must not be allowed to
  change stages, archive applications or access EEO data.

## Constraints

- **C1. Human oversight.** Results are advisory. A resume that repeats the criteria's
  terms without the experience behind them can still score too high, and the service
  cannot confirm that an applicant is genuine, so a recruiter reviews candidates and makes
  every decision.
- **C2. Talent Fit.** Where Lever's Talent Fit feature does not serve a posting well,
  this service ranks that posting's applications against the role's criteria instead.
- **C3. Approval before live use.** Until the role's criteria are approved, Legal and HR
  have approved the jurisdiction policy (`config/policy.yaml`, which sets where
  applications may be ranked and which legal conditions must hold there), and the posting
  passes that policy, the service ranks the posting only in evaluation mode. That output
  is watermarked and never written to Lever. Writing to Lever is off by default and has
  to be switched on.

## Decisions (2026-10-08)

- **LLM:** Gemini Pro on Vertex AI (`gemini-3.1-pro-preview`, the only Pro model after
  `gemini-2.5-pro` retires on 2026-10-20). It is a Public Preview served only on the
  global endpoint, so there is no data-residency guarantee; Legal still has to approve it
  for candidate data.
- **Write-back:** tags that recruiters can filter on in Lever (the tier, and markers for
  applications that need a human check), plus one advisory note per application. Lever
  cannot sort by an integration-written score. Custom fields are not used: Lever's
  profile forms are create-only, have no number type and cannot be filtered.
- **Measuring ranking quality:** before any live use, a backtest compares the ranking
  with past recruiter decisions. An application counts as a success if recruiters moved
  it to a chosen stage or hired the candidate. The backtest reports how often successes
  rank above the rest (AUC), their share of the top-ranked applications (precision@k)
  and their share in each tier.
- **Where it runs:** a CLI for batch ranking, and a webhook service on Cloud Run.

## Open questions

- What the per-role criteria contain, and which postings go first. The criteria format is
  `criteria/*.yaml`; see the example file.
- Legal, HR, Lever and Google Cloud questions: see
  [legal-hr-questions.md](legal-hr-questions.md).
