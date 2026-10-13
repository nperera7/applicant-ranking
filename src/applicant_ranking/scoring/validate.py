"""Validate the model's assessment and turn it into a scored outcome.

Steps (all deterministic):
1. Align ids: keep the first assessment per known requirement/knockout id; drop unknown
   ids (``unknown_ids_dropped`` keeps at most MAX_UNKNOWN_IDS_KEPT of them: they are
   model-written text and the report is stored); fill missing ids with rating "unknown" /
   result "unclear" (``missing_ids_filled``).
2. Evidence grounding (``ground_quote``) against the redacted resume text actually sent.
   A rating of exceeds/meets/partial with no grounded quote is downgraded to "unknown"
   (``downgraded_requirements``); a knockout "unmet" with no grounded quote becomes
   "unclear". If >= 4 quotes were given and more than half are ungrounded, or a
   requirement was downgraded although every quote it had was shorter than
   MIN_QUOTE_CHARS -> ``ungrounded_evidence`` flag (review); else any downgrade ->
   ``ungrounded_evidence`` (info). Quotes grounded only by the fuzzy rule are counted in
   ``quotes_grounded_fuzzy`` and kept as the aligned resume span, not the model's
   wording, so ``evidence`` (which the Lever note shows) holds resume text only. A fuzzy
   quote whose model wording carries a protected term the resume span lacks is not
   grounded. Quotes longer than MAX_FUZZY_QUOTE_CHARS ground only as exact substrings.
3. Free text is trimmed: strengths/gaps items 300 chars, notes 300, red-flag detail 300,
   summary 800; empty strings dropped.
4. Protected-term scan (lexicon) over strengths, gaps, red-flag details, summary,
   requirement/knockout notes. Hits on the first pass -> outcome with ``corrective_note``
   set (the caller retries once; the note lists the categories and the model's own words
   that hit them, never the resume text, and asks to refer only to "the candidate" and
   job-related evidence). Hits after the retry -> ``protected_term_in_output`` (review),
   ``rationale_publishable=False``. Evidence quotes are not scanned here: exact quotes
   are resume text, and an affinity-organisation name in the resume must not withhold a
   rationale.
5. Model signals: injection_attempt_detected -> ``model_injection_report`` (review);
   each red flag -> ``model_red_flag`` (info; "possible_prompt_injection" -> review);
   confidence "low" -> ``low_confidence`` (info); half or more of the must-have
   requirements "unknown" -> ``insufficient_information`` (info).
6. Score = compute_score(criteria results); tier = tier_for(score). If
   |score - overall_fit| >= disagreement_threshold -> ``score_disagreement`` (info).
7. Status = worst of: SCORED; KNOCKOUT_FLAGGED if any knockout "unmet"; NEEDS_REVIEW if
   any flag (including ``prior_flags`` from resume preparation) has severity "review".
   ``review_reasons`` = human-readable reason per review flag.

Conventions:
- Ids match after ``strip().lower()``; slugs are lowercase, so this only forgives case
  and stray whitespace from the model. Later duplicates of an id are ignored.
- Ungrounded knockout "unmet" downgrades count as downgrades for the step-2 flag, but
  only requirement ids go into ``downgraded_requirements``.
- Free text is also cleaned of control and zero-width characters before trimming, so the
  lexicon scan sees what a reader would see.
- A first-pass protected-term hit is already fail-safe: besides ``corrective_note`` the
  outcome carries the review flag and ``rationale_publishable=False``, so a caller that
  cannot retry still publishes nothing. The caller carries the review-severity model
  signals of the first attempt (injection report, review-level red flag, mostly
  ungrounded evidence) into the retry as prior flags (``pipeline.carried_review_signals``);
  everything else about the first outcome is replaced by the retry's.
- ``outcome.flags`` is ``prior_flags`` followed by the flags raised here (exact
  duplicates removed); use it as the evaluation's flag list.
- Flag details and ``review_reasons`` are written by this module only; they never contain
  resume text or model-written text, so they are safe to show anywhere. Red-flag details
  written by the model are kept in their ``model_red_flag`` flag (trimmed and scanned).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from applicant_ranking.config import TierThresholds
from applicant_ranking.domain import (
    Confidence,
    CriterionResult,
    Flag,
    FlagCode,
    KnockoutCheck,
    KnockoutResult,
    Rating,
    RoleCriteria,
    Status,
    Tier,
    ValidationReport,
)
from applicant_ranking.llm.schema import (
    KnockoutAssessment,
    LLMAssessment,
    RequirementAssessment,
)
from applicant_ranking.scoring.compute import RATING_VALUES, compute_score, tier_for
from applicant_ranking.scoring.lexicon import ProtectedTermLexicon

logger = logging.getLogger(__name__)

MAX_ITEM_CHARS = 300
MAX_SUMMARY_CHARS = 800
FUZZY_MIN_SCORE = 90  # rapidfuzz partial_ratio, for quotes of 12 to MAX_FUZZY_QUOTE_CHARS
MIN_QUOTE_CHARS = 4  # shorter quotes must match a whole token, case-sensitively
FUZZY_MIN_QUOTE_CHARS = 12
# Longer quotes ground only as exact substrings. partial_ratio alignment grows roughly
# cubically with the quote (an 11K-char non-verbatim quote against a 30K-char resume takes
# about 0.6 s here, a 20K-char one over a minute) and validation runs on the webhook
# service's event loop; the prompt asks for short spans and the Lever note shows 80 chars.
MAX_FUZZY_QUOTE_CHARS = 500
NEGATION_CONTEXT_WORDS = 3  # source words before a match checked for a dropped negation
UNGROUNDED_REVIEW_MIN_QUOTES = 4
# Unknown assessment ids are model-written strings and the report is persisted (audit
# and PII rows): keep a sample for debugging, count the rest.
MAX_UNKNOWN_IDS_KEPT = 10
_MAX_ID_CHARS = 80
_CONTEXT_CHARS = 120  # enough source text before a match to hold NEGATION_CONTEXT_WORDS
_EDGE_TOKEN_MIN_SCORE = 75  # token ratio for a fuzzy window's edge token to count as quoted
# domain.ValidationReport counter for every unknown id, kept or not (see _align).
_UNKNOWN_COUNT_FIELD = "unknown_ids_dropped_count"
_HAS_UNKNOWN_COUNT = _UNKNOWN_COUNT_FIELD in ValidationReport.model_fields

_POSITIVE_RATINGS: frozenset[str] = frozenset({"exceeds", "meets", "partial"})

_WS_RE = re.compile(r"\s+")
_LINE_BREAKS = frozenset("\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")
_WORD_RE = re.compile(r"[\w']+")
# A number with the word right after it, if any ("10 week", "2TB", "15% on"): fuzzy
# matching must neither change "2 years" to "12 years" nor "10 week" to "10 year".
_NUMBER_RE = re.compile(r"(\d+(?:[.,]\d+)*%?)(?:[ -]?([^\W\d_]+))?")
_WRAPPED_WORD_RE = re.compile(r"(?<=[^\W\d_])- (?=[^\W\d_])")  # "pipe- lines"
# What ends a negation's reach before a match: a line break, a bullet or bar, or
# sentence punctuation followed by a space ("No regrets. Built ..." is not negated).
_CLAUSE_BREAK_RE = re.compile(r"[\n|•·]|[.!?;](?= |$)")
_NEGATIONS = frozenset({"no", "not", "never", "without", "non", "none", "nor", "neither",
                        "lack", "lacks", "lacking", "lacked"})
# "not only", "no less than", ... do not negate what follows.
_NEGATION_EXCEPTIONS = frozenset({"only", "just", "merely", "less", "fewer"})
# A short quote must be a whole token: "C++", "C#" and "R&D" are single tokens.
_SHORT_TOKEN_BOUNDARY = r"\w+#&"  # noqa: S105 - a regex class, not a secret
# C0/C1 controls (tabs and newlines become spaces via _WS_RE first), zero-width and bidi
# controls, BOM.
_STRIP_RE = re.compile("[\x00-\x1f\x7f-\x9f\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
_QUOTE_EDGE_CHARS = (
    " \t\"'`.,;:!?()[]{}<>*_-"
    "\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u201f\u00ab\u00bb\u2039\u203a"
    "\u2026\u2013\u2014\u2022\u00b7"
)
# A resume span used as evidence keeps its brackets: "(2023 - present)" stays balanced.
_SPAN_EDGE_CHARS = "".join(c for c in _QUOTE_EDGE_CHARS if c not in "()[]{}<>")
_FOLD = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u2015": "-", "\u2212": "-", "\u00a0": " ",
})

_CATEGORY_NAMES: dict[str, str] = {
    "age": "age",
    "gender": "gender or pronouns",
    "race_ethnicity": "race or ethnicity",
    "national_origin": "national origin, ancestry or native language",
    "citizenship": "citizenship, visa or immigration status",
    "religion": "religion",
    "disability_health": "disability or health",
    "pregnancy": "pregnancy or parental leave",
    "marital_family": "marital or family status",
    "sexual_orientation": "sexual orientation",
    "veteran_military": "veteran or military status",
    "genetic": "genetic information",
    "employment_gap": "employment gaps",
    "location_proxy": "location, address or neighbourhood",
    "appearance": "physical appearance",
    "criminal_record": "arrest or conviction records",
    "financial_history": "credit or salary history",
    "other_status": "other protected status",
}

_REVIEW_REASONS: dict[FlagCode, str] = {
    FlagCode.HIDDEN_TEXT: "The resume file contains hidden text (for example white or "
                          "tiny text).",
    FlagCode.INJECTION_PHRASE: "The resume contains text that looks like instructions to an "
                               "automated evaluator.",
    FlagCode.UNICODE_TAG_PAYLOAD: "The resume contains invisible Unicode tag characters, a "
                                  "known way to hide instructions.",
    FlagCode.INVISIBLE_CHARACTERS: "The resume contains an unusual number of invisible "
                                   "characters.",
    FlagCode.MIXED_SCRIPT: "The resume contains words that mix look-alike letters from "
                           "different alphabets.",
    FlagCode.HIDDEN_TEXT_CHECK_UNAVAILABLE: "Hidden-text checks could not run on this "
                                            "resume file.",
    FlagCode.SCANNED_RESUME: "The resume is a scanned image; its text was transcribed "
                             "automatically.",
    FlagCode.TRUNCATED: "The resume was too long and only the first part was evaluated.",
    FlagCode.UNSUPPORTED_FILE: "The resume file type is not supported.",
    FlagCode.UNREADABLE_FILE: "The resume file could not be read.",
    FlagCode.NO_RESUME: "No resume was available for this application.",
    FlagCode.RESUME_FROM_PARSED_DATA: "The resume file was unavailable; Lever's parsed "
                                      "resume data was evaluated instead.",
    FlagCode.DUPLICATE_RESUME: "Another application for this posting has a near-identical "
                               "resume.",
    FlagCode.CONTACT_REUSE: "The same email address or phone number appears on applications "
                            "under different names.",
    FlagCode.JD_COPY: "Large parts of the resume copy the job description.",
    FlagCode.UNGROUNDED_EVIDENCE: "Evidence the model quoted could not be found in the "
                                  "resume.",
    FlagCode.PROTECTED_TERM_IN_OUTPUT: "The model's rationale referred to protected "
                                       "characteristics; it is withheld from Lever.",
    FlagCode.INVALID_MODEL_OUTPUT: "The model's output was invalid.",
    FlagCode.MODEL_BLOCKED: "The model declined or was blocked from evaluating this resume.",
    FlagCode.SCORE_DISAGREEMENT: "The computed score and the model's overall impression "
                                 "disagree strongly.",
    FlagCode.LOW_CONFIDENCE: "The model reported low confidence in its assessment.",
    FlagCode.INSUFFICIENT_INFORMATION: "The resume gives too little information on the "
                                       "must-have requirements.",
    FlagCode.MODEL_INJECTION_REPORT: "The model reported text in the resume that tries to "
                                     "manipulate the evaluation.",
    FlagCode.MODEL_RED_FLAG: "The model flagged possible prompt injection in the resume.",
}


class ScoringOutcome(BaseModel):
    status: Status
    score: int | None = None
    tier: Tier | None = None
    holistic_score: int | None = None
    confidence: Confidence | None = None
    criteria_results: list[CriterionResult] = Field(default_factory=list)
    knockout_results: list[KnockoutCheck] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    summary: str = ""
    flags: list[Flag] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)
    rationale_publishable: bool = True
    validation: ValidationReport = Field(default_factory=ValidationReport)
    corrective_note: str | None = None  # set => caller should retry once with this note


# --------------------------------------------------------------------------- grounding


class _Source(NamedTuple):
    """The resume text sent, normalised three ways. ``lines`` and ``flat`` share indices."""

    cased: str  # NFKC, quotes/dashes folded, each whitespace run -> "\n" or " "
    lines: str  # cased, casefolded
    flat: str  # lines with "\n" -> " " (substring and fuzzy matching)


def _collapse_ws(match: re.Match[str]) -> str:
    return "\n" if _LINE_BREAKS.intersection(match.group()) else " "


def _prepare_source(text: str) -> _Source:
    folded = unicodedata.normalize("NFKC", text).translate(_FOLD)
    cased = _WS_RE.sub(_collapse_ws, folded).strip()
    lines = cased.casefold()
    return _Source(cased=cased, lines=lines, flat=lines.replace("\n", " "))


def _normalize_quote(quote: str) -> str:
    """NFKC, quotes/dashes folded, whitespace collapsed, edges stripped; case kept."""
    folded = unicodedata.normalize("NFKC", quote).translate(_FOLD)
    return _WS_RE.sub(" ", folded).strip().strip(_QUOTE_EDGE_CHARS)


def _words(text: str) -> list[str]:
    return [w.strip("'") for w in _WORD_RE.findall(text.casefold()) if w.strip("'")]


def _negations(words: Sequence[str], next_word: str = "") -> Counter[str]:
    """Negation words in ``words`` ("isn't", "cannot" count as "not"), except one that is
    followed by "only", "less", ... (``next_word`` follows the last word)."""
    found: Counter[str] = Counter()
    for position, word in enumerate(words):
        canonical = "not" if word.endswith("n't") or word == "cannot" else word
        following = words[position + 1] if position + 1 < len(words) else next_word
        if canonical in _NEGATIONS and following not in _NEGATION_EXCEPTIONS:
            found[canonical] += 1
    return found


def _negated_before(before: str, quote_words: Sequence[str]) -> bool:
    """True if the last few words of ``before`` in the same line and clause hold a negation
    the quote leaves out ("production Kubernetes experience" inside "no production
    Kubernetes experience")."""
    breaks = list(_CLAUSE_BREAK_RE.finditer(before))
    if breaks:
        before = before[breaks[-1].end():]
    words = _words(before)[-NEGATION_CONTEXT_WORDS:]
    next_word = quote_words[0] if quote_words else ""
    return bool(_negations(words, next_word) - _negations(quote_words))


def _numbers(text: str) -> Counter[tuple[str, str]]:
    return Counter(_NUMBER_RE.findall(_WRAPPED_WORD_RE.sub("", text)))


def _widen(text: str, start: int, end: int) -> tuple[int, int]:
    """Extend [start, end) of whitespace-collapsed ``text`` to whole space-separated
    tokens, so a number at the edge of a fuzzy alignment is not cut ("12" vs "2")."""
    if start < len(text) and text[start] != " ":
        while start > 0 and text[start - 1] != " ":
            start -= 1
    if end > 0 and text[end - 1] != " ":
        while end < len(text) and text[end] != " ":
            end += 1
    return start, end


def _match_short(quote: str, quote_words: Sequence[str], source: _Source) -> bool:
    # Case-sensitive: casefolded "go" or "r" would match ordinary prose.
    pattern = re.escape(quote).replace("\\ ", r"\s")
    boundary = _SHORT_TOKEN_BOUNDARY
    for found in re.finditer(rf"(?<![{boundary}]){pattern}(?![{boundary}])", source.cased):
        before = source.cased[max(0, found.start() - _CONTEXT_CHARS):found.start()]
        if not _negated_before(before, quote_words):
            return True
    return False


def _resembles(token: str, quote_words: Sequence[str]) -> bool:
    return any(fuzz.ratio(word, quote_word) >= _EDGE_TOKEN_MIN_SCORE
               for word in _words(token) for quote_word in quote_words)


def _trim_edges(source: _Source, start: int, end: int,
                quote_words: Sequence[str]) -> tuple[int, int]:
    """Shrink a widened fuzzy window [start, end) past leading/trailing tokens that resemble
    no word of the quote: the aligner pairs a word the model added ("young:") with a stray
    neighbouring source word, which must not become evidence. At least one token stays."""
    tokens = source.flat[start:end].split(" ")
    while len(tokens) > 1 and not _resembles(tokens[0], quote_words):
        start += len(tokens.pop(0)) + 1
    while len(tokens) > 1 and not _resembles(tokens[-1], quote_words):
        end -= len(tokens.pop()) + 1
    return start, end


def _span_text(source: _Source, start: int, end: int) -> str:
    """The resume text at [start, end) of ``flat``, in its original case when casefolding
    kept every index (it lengthens a few letters such as "ß"), line breaks as spaces and
    edges stripped like a model quote."""
    text = source.cased if len(source.cased) == len(source.flat) else source.flat
    return text[start:end].replace("\n", " ").strip(_SPAN_EDGE_CHARS)


def _match_fuzzy(cased_quote: str, quote_words: Sequence[str], source: _Source,
                 lexicon: ProtectedTermLexicon | None) -> str | None:
    """Fuzzy matching forgives formatting noise (hyphenation, dropped punctuation, a typo),
    never changed content: the aligned source window must hold the same numbers (each
    with the word after it) as the quote and no negation the quote leaves out, nothing
    just before it may negate it, and (given a lexicon) the model's wording may not carry
    a protected term the window lacks. Returns the resume text of the window, which
    replaces the model's wording as evidence."""
    folded_quote = cased_quote.casefold()
    alignment = fuzz.partial_ratio_alignment(folded_quote, source.flat,
                                             score_cutoff=FUZZY_MIN_SCORE)
    if alignment is None:
        return None
    start, end = _widen(source.flat, alignment.dest_start, alignment.dest_end)
    window = source.flat[start:end]
    if _numbers(folded_quote) != _numbers(window):
        return None
    if _negations(_words(window)) - _negations(quote_words):
        return None
    start, end = _trim_edges(source, start, end, quote_words)
    if _negated_before(source.lines[max(0, start - _CONTEXT_CHARS):start], quote_words):
        return None
    span = _span_text(source, start, end)
    if lexicon is not None:
        added = ({hit.label for hit in lexicon.scan(cased_quote)}
                 - {hit.label for hit in lexicon.scan(span)})
        if added:
            logger.info("rejected a fuzzy evidence quote whose wording adds protected "
                        "terms the resume span lacks (%s)", ", ".join(sorted(added)))
            return None
    return span


class _Match(NamedTuple):
    kind: Literal["exact", "fuzzy"]
    text: str  # what becomes evidence: the model's quote (exact) or the resume span (fuzzy)


def _match(quote: str, source: _Source,
           lexicon: ProtectedTermLexicon | None = None) -> _Match | None:
    stripped = quote.strip()
    cased_quote = _normalize_quote(stripped)
    if not cased_quote or not source.flat:
        return None
    quote_words = _words(cased_quote)
    if len(cased_quote) < MIN_QUOTE_CHARS:
        return _Match("exact", stripped) if _match_short(cased_quote, quote_words,
                                                         source) else None
    folded_quote = cased_quote.casefold()
    start = source.flat.find(folded_quote)
    while start != -1:
        if not _negated_before(source.lines[max(0, start - _CONTEXT_CHARS):start],
                               quote_words):
            return _Match("exact", stripped)
        start = source.flat.find(folded_quote, start + 1)
    # partial_ratio aligns the shorter string inside the longer one: a quote longer than
    # the source would be "found" by containing it, so fuzzy matching needs q <= source.
    if FUZZY_MIN_QUOTE_CHARS <= len(folded_quote) <= min(MAX_FUZZY_QUOTE_CHARS,
                                                         len(source.flat)):
        span = _match_fuzzy(cased_quote, quote_words, source, lexicon)
        if span:
            return _Match("fuzzy", span)
    return None


def ground_quote(quote: str, source_text: str) -> bool:
    """True if the quote appears in the source. Both sides are NFKC-normalised with
    typographic quotes/dashes folded to ASCII and whitespace collapsed; the quote's
    surrounding punctuation/quotes/ellipses are stripped. Then:

    - Quotes of >= 4 chars: casefolded substring match; for quotes of 12 to
      MAX_FUZZY_QUOTE_CHARS chars also rapidfuzz ``fuzz.partial_ratio`` >= FUZZY_MIN_SCORE,
      for formatting noise only: the aligned source window (widened to whole tokens) must
      contain the same numbers, each with the word after it ("10 week"), as a multiset,
      and no negation word the quote lacks. Longer quotes never match fuzzily (the
      alignment cost grows superlinearly with the quote).
    - Quotes shorter than 4 chars ("SQL", "Go", "C++"): case-sensitive whole-token match,
      where + # & are token characters (so "C" is not found in "C++" or "R" in "R&D").
    - Any match is rejected when the 1-3 source words just before it (same line and
      clause) contain a negation the quote omits; another occurrence may still ground."""
    return _match(quote, _prepare_source(source_text)) is not None


# --------------------------------------------------------------------------- free text


def clean_text(text: str, limit: int) -> str:
    """Collapse whitespace, drop control/zero-width characters, trim to ``limit`` chars
    (cut at a word boundary when one is near, ending with an ellipsis)."""
    collapsed = _WS_RE.sub(" ", text)
    cleaned = _STRIP_RE.sub("", collapsed).strip()
    if len(cleaned) <= limit:
        return cleaned
    cut = cleaned[:limit - 1]
    space = cut.rfind(" ")
    if space >= int(limit * 0.6):
        cut = cut[:space]
    return cut.rstrip(" ,;:-") + "\u2026"


def _clean_items(items: Sequence[str], limit: int) -> list[str]:
    cleaned = (clean_text(item, limit) for item in items)
    return [item for item in cleaned if item]


def _safe_id(raw: str) -> str:
    return clean_text(raw, _MAX_ID_CHARS)


def _key(raw: str) -> str:
    return raw.strip().lower()


# --------------------------------------------------------------------------- steps


def _align[T: (RequirementAssessment, KnockoutAssessment)](
    known_ids: Sequence[str], items: Sequence[T], id_of: Callable[[T], str],
    report: ValidationReport,
) -> dict[str, T]:
    known = {_key(i): i for i in known_ids}
    aligned: dict[str, T] = {}
    duplicates = 0
    unknown = 0
    for item in items:
        raw_id = id_of(item)
        canonical = known.get(_key(raw_id))
        if canonical is None:
            unknown += 1
            if len(report.unknown_ids_dropped) < MAX_UNKNOWN_IDS_KEPT:
                report.unknown_ids_dropped.append(_safe_id(raw_id))
        elif canonical in aligned:
            duplicates += 1
        else:
            aligned[canonical] = item
    if unknown:
        logger.info("dropped %d assessment(s) whose ids are not in the criteria", unknown)
        if _HAS_UNKNOWN_COUNT:
            setattr(report, _UNKNOWN_COUNT_FIELD,
                    getattr(report, _UNKNOWN_COUNT_FIELD) + unknown)
    if duplicates:
        logger.info("ignored %d duplicate assessment(s) for already-assessed ids", duplicates)
    for canonical in known_ids:
        if canonical not in aligned:
            report.missing_ids_filled.append(canonical)
    return aligned


class _Quotes(NamedTuple):
    grounded: list[str]
    ungrounded: list[str]
    fuzzy: int  # grounded only by the fuzzy rule
    short_only: bool  # quotes were given and every one is shorter than MIN_QUOTE_CHARS


def _split_quotes(quotes: Sequence[str], source: _Source,
                  lexicon: ProtectedTermLexicon | None = None) -> _Quotes:
    """``grounded`` holds resume text only: the model's quote when it is an exact
    (case/whitespace-insensitive) substring, the aligned resume span when it matched
    fuzzily. ``ungrounded`` keeps the model's wording for the PII record."""
    grounded: list[str] = []
    ungrounded: list[str] = []
    fuzzy = 0
    short_only = True
    for quote in quotes:
        stripped = quote.strip()
        if not stripped:
            continue
        short_only = short_only and 0 < len(_normalize_quote(stripped)) < MIN_QUOTE_CHARS
        match = _match(stripped, source, lexicon)
        if match is None:
            ungrounded.append(stripped)
            continue
        fuzzy += match.kind == "fuzzy"
        grounded.append(match.text)
    return _Quotes(grounded, ungrounded, fuzzy, short_only and bool(grounded or ungrounded))


def _protected_term_note(labels: Sequence[str]) -> str:
    """The corrective note for the retry: each category hit with the words of the model's
    own free text that hit it (quoted so the model can rephrase exactly those), never
    resume text, since evidence quotes are not scanned."""
    terms_by_category: dict[str, list[str]] = {}
    for label in labels:
        category, _, term = label.partition(":")
        terms = terms_by_category.setdefault(category, [])
        if term and term not in terms:
            terms.append(term)
    found = "; ".join(
        _CATEGORY_NAMES.get(category, category.replace("_", " "))
        + (": " + ", ".join(f'"{t}"' for t in terms) if terms else "")
        for category, terms in sorted(terms_by_category.items())
    )
    return (
        "Correction required: your previous answer was rejected by an automated check "
        f"because its free text referred to protected characteristics or proxies ({found}). "
        "Return the complete JSON object again. In every free-text field (strengths, gaps, "
        "red flag details, summary and notes), refer to the person only as \"the "
        "candidate\" (never he, she, his, her or similar), do not mention or allude to any "
        "protected characteristic or a proxy for one, and cite only job-related evidence "
        "from the resume measured against the listed requirements, quoted verbatim."
    )


def _dedupe_flags(flags: Sequence[Flag]) -> list[Flag]:
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[Flag] = []
    for flag in flags:
        key = (flag.code.value, flag.source, flag.severity, flag.detail)
        if key not in seen:
            seen.add(key)
            unique.append(flag)
    return unique


# Flags whose ``detail`` this module writes itself (counts and lexicon category names only).
_DETAILED_CODES = frozenset({
    FlagCode.UNGROUNDED_EVIDENCE,
    FlagCode.PROTECTED_TERM_IN_OUTPUT,
    FlagCode.SCORE_DISAGREEMENT,
    FlagCode.INSUFFICIENT_INFORMATION,
})


def review_reason(flag: Flag) -> str:
    """Human-readable reason for a review flag, written without resume or model text."""
    base = _REVIEW_REASONS.get(flag.code, f"Automated check '{flag.code.value}' requires "
                                          "human review.")
    if flag.source == "system" and flag.code in _DETAILED_CODES and flag.detail:
        return f"{base.rstrip('.')} ({flag.detail})."
    return base


def validate_and_score(
    *,
    criteria: RoleCriteria,
    assessment: LLMAssessment,
    resume_text: str,
    lexicon: ProtectedTermLexicon,
    thresholds: TierThresholds,
    disagreement_threshold: int = 30,
    prior_flags: Sequence[Flag] = (),
    corrective_retries: int = 0,
) -> ScoringOutcome:
    report = ValidationReport(corrective_retries=corrective_retries)
    flags: list[Flag] = []

    # 1. Align ids.
    req_by_id = _align([r.id for r in criteria.requirements], assessment.requirements,
                       lambda a: a.requirement_id, report)
    ko_by_id = _align([k.id for k in criteria.knockouts], assessment.knockouts,
                      lambda a: a.knockout_id, report)

    # 2. Evidence grounding (3. notes trimmed on the way).
    source = _prepare_source(resume_text)
    ungrounded_count = 0
    knockout_downgrades = 0
    short_downgrades = 0
    criteria_results: list[CriterionResult] = []
    for requirement in criteria.requirements:
        item = req_by_id.get(requirement.id)
        model_rating: Rating = item.rating if item is not None else "unknown"
        grounded, ungrounded, fuzzy, short_only = _split_quotes(
            item.evidence if item else (), source, lexicon)
        report.quotes_total += len(grounded) + len(ungrounded)
        report.quotes_grounded += len(grounded)
        report.quotes_grounded_fuzzy += fuzzy
        ungrounded_count += len(ungrounded)
        rating: Rating = model_rating
        if model_rating in _POSITIVE_RATINGS and not grounded:
            rating = "unknown"
            report.downgraded_requirements.append(requirement.id)
            short_downgrades += short_only
        weight = requirement.effective_weight
        criteria_results.append(CriterionResult(
            requirement_id=requirement.id,
            description=requirement.description,
            importance=requirement.importance,
            weight=weight,
            rating=rating,
            model_rating=model_rating,
            evidence=grounded,
            ungrounded_evidence=ungrounded,
            note=clean_text(item.note, MAX_ITEM_CHARS) if item else "",
            points=weight * RATING_VALUES[rating],
        ))

    knockout_results: list[KnockoutCheck] = []
    for knockout in criteria.knockouts:
        ko_item = ko_by_id.get(knockout.id)
        result: KnockoutResult = ko_item.result if ko_item is not None else "unclear"
        grounded, ungrounded, fuzzy, _ = _split_quotes(ko_item.evidence if ko_item else (),
                                                       source, lexicon)
        report.quotes_total += len(grounded) + len(ungrounded)
        report.quotes_grounded += len(grounded)
        report.quotes_grounded_fuzzy += fuzzy
        ungrounded_count += len(ungrounded)
        if result == "unmet" and not grounded:
            result = "unclear"
            knockout_downgrades += 1
        knockout_results.append(KnockoutCheck(
            knockout_id=knockout.id,
            description=knockout.description,
            result=result,
            evidence=grounded,
            note=clean_text(ko_item.note, MAX_ITEM_CHARS) if ko_item else "",
        ))

    downgrades = len(report.downgraded_requirements) + knockout_downgrades
    mostly_ungrounded = (report.quotes_total >= UNGROUNDED_REVIEW_MIN_QUOTES
                         and ungrounded_count * 2 > report.quotes_total)
    if mostly_ungrounded or short_downgrades:
        # A downgrade that rests only on very short quotes ("Sql" for "SQL") may be a
        # near miss rather than invented evidence: never let it sink a candidate silently.
        short_note = (f"; {short_downgrades} of them cited only quotes under "
                      f"{MIN_QUOTE_CHARS} characters" if short_downgrades else "")
        flags.append(Flag(
            code=FlagCode.UNGROUNDED_EVIDENCE, severity="review",
            detail=f"{ungrounded_count} of {report.quotes_total} evidence quotes were not "
                   f"found in the resume text; {downgrades} rating(s) downgraded"
                   + short_note,
        ))
    elif downgrades:
        flags.append(Flag(
            code=FlagCode.UNGROUNDED_EVIDENCE, severity="info",
            detail=f"{downgrades} rating(s) downgraded for lack of evidence found in the "
                   "resume text",
        ))

    # 3. Free text.
    strengths = _clean_items(assessment.strengths, MAX_ITEM_CHARS)
    gaps = _clean_items(assessment.gaps, MAX_ITEM_CHARS)
    summary = clean_text(assessment.summary, MAX_SUMMARY_CHARS)
    red_flags = [(rf.category, clean_text(rf.detail, MAX_ITEM_CHARS))
                 for rf in assessment.red_flags]

    # 4. Protected-term scan.
    texts = [*strengths, *gaps, *(detail for _, detail in red_flags), summary,
             *(r.note for r in criteria_results), *(k.note for k in knockout_results)]
    labels: list[str] = []
    categories: list[str] = []
    for text in texts:
        for hit in lexicon.scan(text):
            if hit.label not in labels:
                labels.append(hit.label)
            if hit.category not in categories:
                categories.append(hit.category)
    report.protected_term_hits = labels
    corrective_note: str | None = None
    rationale_publishable = True
    if categories:
        rationale_publishable = False
        names = ", ".join(sorted(categories))
        flags.append(Flag(
            code=FlagCode.PROTECTED_TERM_IN_OUTPUT, severity="review",
            detail=f"categories: {names}"
                   + ("" if corrective_retries else "; a corrective retry was requested"),
        ))
        if corrective_retries == 0:
            corrective_note = _protected_term_note(labels)
        logger.info("protected-term hits in model output (retry %d): %s",
                    corrective_retries, names)

    # 5. Model signals.
    if assessment.injection_attempt_detected:
        flags.append(Flag(code=FlagCode.MODEL_INJECTION_REPORT, source="model",
                          severity="review",
                          detail="the model reported an attempt to manipulate the evaluation"))
    for category, detail in red_flags:
        flags.append(Flag(
            code=FlagCode.MODEL_RED_FLAG, source="model",
            severity="review" if category == "possible_prompt_injection" else "info",
            detail=f"{category}: {detail}" if detail else category,
        ))
    if assessment.confidence == "low":
        flags.append(Flag(code=FlagCode.LOW_CONFIDENCE, source="model", severity="info",
                          detail="the model reported low confidence"))
    must_haves = [r for r in criteria_results if r.importance == "must_have"]
    unknown_must = sum(1 for r in must_haves if r.rating == "unknown")
    if must_haves and unknown_must * 2 >= len(must_haves):
        flags.append(Flag(
            code=FlagCode.INSUFFICIENT_INFORMATION, severity="info",
            detail=f"{unknown_must} of {len(must_haves)} must-have requirements have no "
                   "evidence either way",
        ))

    # 6. Score.
    score = compute_score(criteria_results)
    tier = tier_for(score, thresholds) if score is not None else None
    if score is not None and abs(score - assessment.overall_fit) >= disagreement_threshold:
        flags.append(Flag(
            code=FlagCode.SCORE_DISAGREEMENT, severity="info",
            detail=f"computed score {score} vs model's holistic {assessment.overall_fit}",
        ))

    # 7. Status.
    all_flags = _dedupe_flags([*prior_flags, *flags])
    statuses = [Status.SCORED]
    if any(k.result == "unmet" for k in knockout_results):
        statuses.append(Status.KNOCKOUT_FLAGGED)
    review_flags = [f for f in all_flags if f.severity == "review"]
    if review_flags:
        statuses.append(Status.NEEDS_REVIEW)
    reasons: list[str] = []
    for flag in review_flags:
        reason = review_reason(flag)
        if reason not in reasons:
            reasons.append(reason)

    return ScoringOutcome(
        status=Status.worst(*statuses),
        score=score,
        tier=tier,
        holistic_score=assessment.overall_fit,
        confidence=assessment.confidence,
        criteria_results=criteria_results,
        knockout_results=knockout_results,
        strengths=strengths,
        gaps=gaps,
        summary=summary,
        flags=all_flags,
        review_reasons=reasons,
        rationale_publishable=rationale_publishable,
        validation=report,
        corrective_note=corrective_note,
    )
