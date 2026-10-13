"""Lint per-role criteria before they can be used.

Errors (block every mode, including evaluation):
- Any protected-term lexicon hit in the title, summary, requirement or knockout
  descriptions, or guardrails (e.g. "young", "native English speaker").
- Proxy phrases: "digital native", "recent graduate"/"new grad" (as a requirement),
  "culture fit", "local candidates only", "no (employment )?gaps", "overqualified",
  maximum-experience caps ("no more than N years", "at most N years", "maximum (of )?N
  years", "N years or less", "less than N years of experience").
- Knockouts about citizenship, visa, sponsorship or work authorisation: must come from a
  structured Lever application question, never resume inference.
- Duplicate ids across requirements and knockouts.
Warnings (shown; block LIVE/DRY_RUN only via approval, not here):
- source == "job_description" (uncurated skeleton); not approved, or ``approved_at`` in
  the future (``RoleCriteria.approval_issue``; the pipeline refuses both outside
  evaluation mode).
- Requirement descriptions shorter than 15 characters (too vague to judge).
- More than 10 must-haves.
- Degree requirements ("degree", "bachelor", "BS/BA", "master", "PhD"): review for
  disparate impact; consider "or equivalent experience".

Also (beyond the list above):
- Approved domain terms: a lexicon hit whose term (or matched text) is listed in the
  criteria's ``approved_terms`` (entries with term, justification, approved_by and
  approved_at, recorded by Legal/HR) is reported as a warning that names the approver,
  so a fintech, healthcare or public-sector role can state its domain ("credit history
  data", "Veterans Affairs systems") with a visible record instead of being impossible.
- "fresh graduate" counts with "recent graduate"; "cultural fit" with "culture fit";
  "up to N years of experience" is a maximum-experience cap.
- Where a proxy phrase and a lexicon term overlap, only the proxy error (with its more
  specific advice) is reported; a work-authorisation knockout is reported once by its
  own rule rather than again as a citizenship term.
- Warnings: lexicon hits in a knockout's justification (not sent to the model, but part
  of the documented reasoning); experience ranges such as "3-5 years" (read as a cap);
  the short-description and degree checks also cover knockouts. A degree mention that
  already offers "or equivalent" is not flagged.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

from applicant_ranking.domain import RoleCriteria, utcnow
from applicant_ranking.scoring.lexicon import (
    LexiconHit,
    ProtectedTermLexicon,
    normalize_for_scan,
    phrase_key,
)

MIN_DESCRIPTION_CHARS = 15
MAX_MUST_HAVES = 10
APPROVED_TERMS_FIELD = "approved_terms"

FieldKind = Literal["title", "summary", "requirement", "knockout", "guardrail"]


class LintIssue(BaseModel):
    level: Literal["error", "warning"]
    location: str  # e.g. "requirements[gcp-data].description"
    message: str


@dataclass(frozen=True, slots=True)
class _Field:
    location: str
    text: str  # normalised with lexicon.normalize_for_scan, so offsets match lexicon hits
    kind: FieldKind
    item_id: str | None = None


@dataclass(frozen=True, slots=True)
class _ProxyRule:
    pattern: re.Pattern[str]
    message: str
    kinds: frozenset[FieldKind] | None = None  # None = every field


_N = (r"(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
      r"fifteen|twenty)")
_YEARS = r"\s*\+?\s*(?:years?|yrs?)\b"
_EXPERIENCE = r"\s+(?:of\s+)?(?:[\w/-]+\s+){0,3}?experience\b"
_AS_REQUIREMENT: frozenset[FieldKind] = frozenset({"requirement", "knockout"})

_CAP_MESSAGE = ("caps experience at a maximum, which screens out experienced (typically "
                "older) candidates; state the minimum the job needs instead")

_PROXY_RULES: tuple[_ProxyRule, ...] = (
    _ProxyRule(re.compile(r"\bdigital[\s-]+natives?\b", re.I),
               "'digital native' is an age proxy; describe the specific digital skills "
               "needed instead"),
    _ProxyRule(re.compile(r"\b(?:recent|new|fresh)[\s-]+(?:college[\s-]+|university[\s-]+)?"
                          r"grad(?:uate)?s?\b", re.I),
               "requiring a recent/new graduate is an age proxy; describe the skills or "
               "knowledge needed instead", _AS_REQUIREMENT),
    _ProxyRule(re.compile(r"\bcultur(?:e|al)[\s-]+fit\b", re.I),
               "'culture fit' is subjective and a common proxy for bias; name the "
               "job-related behaviours instead (e.g. 'explains designs to clients')"),
    _ProxyRule(re.compile(r"\b(?:local[\s-]+(?:candidates|applicants)[\s-]+only|"
                          r"locals[\s-]+only|only[\s-]+local[\s-]+(?:candidates|applicants))\b",
                          re.I),
               "restricting to local candidates is a location proxy (zip codes and "
               "neighbourhoods are banned proxies in Illinois); state on-site requirements "
               "in the posting, not the criteria"),
    _ProxyRule(re.compile(r"\bno[\s-]+(?:employment[\s-]+)?gaps?\b", re.I),
               "penalising employment gaps disadvantages caregivers and people with "
               "disabilities; remove it"),
    _ProxyRule(re.compile(r"\bover[\s-]?qualified\b", re.I),
               "'overqualified' is an age proxy; remove it"),
    _ProxyRule(re.compile(rf"\bno\s+more\s+than\s+{_N}{_YEARS}", re.I), _CAP_MESSAGE),
    _ProxyRule(re.compile(rf"\bat\s+most\s+{_N}{_YEARS}", re.I), _CAP_MESSAGE),
    _ProxyRule(re.compile(rf"\bmax(?:imum|\.)?\s+(?:of\s+)?{_N}{_YEARS}", re.I), _CAP_MESSAGE),
    _ProxyRule(re.compile(rf"\b{_N}{_YEARS}(?:{_EXPERIENCE})?\s+or\s+(?:less|fewer)\b", re.I),
               _CAP_MESSAGE),
    _ProxyRule(re.compile(rf"\b(?:less|fewer)\s+than\s+{_N}{_YEARS}{_EXPERIENCE}", re.I),
               _CAP_MESSAGE),
    _ProxyRule(re.compile(rf"\bup\s+to\s+{_N}{_YEARS}{_EXPERIENCE}", re.I), _CAP_MESSAGE),
)

_WORK_AUTH_RE = re.compile(
    r"\b(?:citizen\w*|visas?|sponsor\w*|work(?:ing)?[\s-]+authori[sz]\w*|"
    r"employment[\s-]+authori[sz]\w*|authori[sz]ed[\s-]+to[\s-]+work|right[\s-]+to[\s-]+work|"
    r"eligib\w*[\s-]+to[\s-]+work|work[\s-]+eligib\w*|work[\s-]+permits?|green[\s-]+cards?|"
    r"permanent[\s-]+residen\w*|immigration|naturali[sz]\w*|h-?1b|u\.?s\.?[\s-]+persons?)"
    r"(?!\w)",
    re.I)

_DEGREE_RE = re.compile(
    r"\b(?:degrees?|bachelor\w*|master'?s|masters|master[\s-]+of|ph\.?\s?d\b\.?|doctorate|"
    r"mba|b\.\s?[sa]\.|m\.\s?s\.|bsc|msc|(?:bs|ba|ms|ma)\s*/\s*(?:bs|ba|ms|ma)\b|"
    r"(?:bs|ba|ms)\s+(?:in|degree)\b)",
    re.I)
_EQUIVALENT_RE = re.compile(r"\bor\s+(?:an?\s+)?equivalent\b", re.I)
_RANGE_RE = re.compile(r"\b(\d{1,2})\s*(?:-|to)\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b", re.I)


def _fields(criteria: RoleCriteria) -> list[_Field]:
    fields = [_Field("title", normalize_for_scan(criteria.title), "title")]
    if criteria.summary:
        fields.append(_Field("summary", normalize_for_scan(criteria.summary), "summary"))
    fields.extend(_Field(f"requirements[{r.id}].description", normalize_for_scan(r.description),
                         "requirement", r.id) for r in criteria.requirements)
    fields.extend(_Field(f"knockouts[{k.id}].description", normalize_for_scan(k.description),
                         "knockout", k.id) for k in criteria.knockouts)
    fields.extend(_Field(f"guardrails[{i}]", normalize_for_scan(g), "guardrail")
                  for i, g in enumerate(criteria.guardrails))
    return fields


def _approval_field(entry: Any, name: str) -> Any:
    return entry.get(name) if isinstance(entry, Mapping) else getattr(entry, name, None)


def approved_terms(criteria: RoleCriteria) -> dict[str, Any]:
    """Normalised term -> its approval record (an object or mapping with ``term``,
    ``justification``, ``approved_by`` and ``approved_at``) from the criteria's
    ``approved_terms`` field; empty when none are listed."""
    approved: dict[str, Any] = {}
    for entry in getattr(criteria, APPROVED_TERMS_FIELD, None) or ():
        term = _approval_field(entry, "term")
        if isinstance(term, str) and phrase_key(term):
            approved[phrase_key(term)] = entry
    return approved


def _approved_message(hit: LexiconHit, approval: Any) -> str:
    approver = _approval_field(approval, "approved_by") or "an unnamed approver"
    approved_at = _approval_field(approval, "approved_at")
    justification = _approval_field(approval, "justification")
    message = (f"uses the approved domain term '{hit.term}' ({hit.category}), approved by "
               f"{approver}")
    if approved_at:
        message += f" on {approved_at}"
    if justification:
        message += f": {justification}"
    return message


def _lexicon_message(category: str, term: str, kind: FieldKind) -> str:
    message = (f"mentions a protected characteristic or proxy ({category}: '{term}'); "
               "criteria must describe job-related skills and experience only")
    if kind == "guardrail":
        message += ("; the fairness rules (no protected characteristics) are already built "
                    "into the prompt, so guardrails need not name them")
    return message


def lint_criteria(criteria: RoleCriteria, lexicon: ProtectedTermLexicon, *,
                  now: datetime | None = None) -> list[LintIssue]:
    """``now`` (default: the current UTC time) dates the approval check."""
    errors: list[LintIssue] = []
    warnings: list[LintIssue] = []
    work_auth_knockouts = {
        k.id for k in criteria.knockouts
        if _WORK_AUTH_RE.search(normalize_for_scan(f"{re.sub(r'[-_]', ' ', k.id)} {k.description}"))
    }
    approved = approved_terms(criteria)

    for field in _fields(criteria):
        proxy_spans: list[tuple[int, int]] = []
        for rule in _PROXY_RULES:
            if rule.kinds is not None and field.kind not in rule.kinds:
                continue
            match = rule.pattern.search(field.text)
            if match:
                proxy_spans.append(match.span())
                errors.append(LintIssue(level="error", location=field.location,
                                        message=f"'{match.group()}': {rule.message}"))
        auth_knockout = field.kind == "knockout" and field.item_id in work_auth_knockouts
        for hit in lexicon.scan(field.text):
            if any(hit.start < end and start < hit.end for start, end in proxy_spans):
                continue
            if auth_knockout and hit.category == "citizenship":
                continue
            approval = (approved.get(phrase_key(hit.term))
                        or approved.get(phrase_key(field.text[hit.start:hit.end])))
            if approval is not None:
                warnings.append(LintIssue(level="warning", location=field.location,
                                          message=_approved_message(hit, approval)))
                continue
            errors.append(LintIssue(level="error", location=field.location,
                                    message=_lexicon_message(hit.category, hit.term,
                                                             field.kind)))

    for knockout in criteria.knockouts:
        if knockout.id in work_auth_knockouts:
            errors.append(LintIssue(
                level="error", location=f"knockouts[{knockout.id}]",
                message="knockouts about citizenship, visas, sponsorship or work "
                        "authorisation are not allowed: ask a structured Lever application "
                        "question instead of inferring it from the resume",
            ))

    seen: set[str] = set()
    for section, item_id in [*(("requirements", r.id) for r in criteria.requirements),
                             *(("knockouts", k.id) for k in criteria.knockouts)]:
        if item_id in seen:
            errors.append(LintIssue(
                level="error", location=f"{section}[{item_id}]",
                message=f"duplicate id '{item_id}': ids must be unique across requirements "
                        "and knockouts",
            ))
        seen.add(item_id)

    # Warnings.
    if criteria.source == "job_description":
        warnings.append(LintIssue(
            level="warning", location="source",
            message="drafted from the Lever job description and not yet curated by the "
                    "hiring manager; rewrite the requirements and set source: hiring_manager",
        ))
    approval_issue = criteria.approval_issue(now if now is not None else utcnow())
    if approval_issue:
        warnings.append(LintIssue(
            level="warning", location="approved_by" if not criteria.approved else "approved_at",
            message=f"{approval_issue}; only evaluation mode can use unapproved criteria",
        ))
    described = [(f"requirements[{r.id}].description", r.description)
                 for r in criteria.requirements]
    described += [(f"knockouts[{k.id}].description", k.description) for k in criteria.knockouts]
    for location, description in described:
        if len(description.strip()) < MIN_DESCRIPTION_CHARS:
            warnings.append(LintIssue(
                level="warning", location=location,
                message=f"description is shorter than {MIN_DESCRIPTION_CHARS} characters and "
                        "too vague to judge; say what evidence would show it",
            ))
        text = normalize_for_scan(description)
        degree = _DEGREE_RE.search(text)
        if degree and not _EQUIVALENT_RE.search(text):
            warnings.append(LintIssue(
                level="warning", location=location,
                message=f"degree requirement ('{degree.group()}'): review it for disparate "
                        "impact and consider 'or equivalent experience'",
            ))
    must_haves = sum(1 for r in criteria.requirements if r.importance == "must_have")
    if must_haves > MAX_MUST_HAVES:
        warnings.append(LintIssue(
            level="warning", location="requirements",
            message=f"{must_haves} must-have requirements (more than {MAX_MUST_HAVES}); "
                    "few candidates will meet them all, so the ranking flattens. Move some "
                    "to nice_to_have",
        ))
    range_fields = [("summary", criteria.summary), *described]
    for location, text in range_fields:
        for match in _RANGE_RE.finditer(normalize_for_scan(text)):
            low, high = int(match.group(1)), int(match.group(2))
            if high > low:
                warnings.append(LintIssue(
                    level="warning", location=location,
                    message=f"'{match.group()}' reads as a maximum; state the minimum "
                            f"('{low}+ years') unless the upper bound is truly job-related",
                ))
    for knockout in criteria.knockouts:
        for hit in lexicon.scan(knockout.justification):
            warnings.append(LintIssue(
                level="warning", location=f"knockouts[{knockout.id}].justification",
                message=f"justification mentions a protected characteristic or proxy "
                        f"({hit.category}: '{hit.term}'); check the knockout is job-related",
            ))
    return errors + warnings


def has_errors(issues: list[LintIssue]) -> bool:
    return any(i.level == "error" for i in issues)
