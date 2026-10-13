"""Cheap, defensible fake-candidate / integrity signals. They produce "needs
verification" flags for a human and are NEVER score inputs. Not used, on purpose: IP or
geography, PDF producer metadata, name heuristics, missing LinkedIn.

Normalisation shared by every function here: accents are folded (NFKD, combining marks
dropped), text is case-folded, and every run of non-alphanumeric characters becomes one
space. For ASCII text that is exactly "lower-case, collapse non [a-z0-9]"; letters of other
scripts are kept as letters, so two different all-Cyrillic resumes don't collapse to the
same digits-and-spaces string and collide.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections.abc import Mapping
from itertools import combinations

_NON_ALNUM_RE = re.compile(r"[\W_]+")
_HTML_TAG_RE = re.compile(r"<[^>]{0,200}>")


def _normalise(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()
    return _NON_ALNUM_RE.sub(" ", folded).strip()


def _words(text: str) -> list[str]:
    normalised = _normalise(text)
    return normalised.split(" ") if normalised else []


def fingerprint(redacted_text: str) -> str:
    """sha256 hex of lower-cased text with every run of non [a-z0-9] collapsed to one
    space (identical resumes that differ only in redacted identity collide)."""
    return hashlib.sha256(_normalise(redacted_text).encode()).hexdigest()


def _shingles_of(words: list[str], n: int) -> set[str]:
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def shingles(text: str, n: int = 5) -> set[str]:
    """Word n-grams over the same normalisation as ``fingerprint``. Empty when the text
    has fewer than ``n`` words."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return _shingles_of(_words(text), n)


def jaccard(a: set[str], b: set[str]) -> float:
    """|a & b| / |a | b|; 0.0 when both are empty (nothing to compare is not a match)."""
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def find_duplicates(texts: Mapping[str, str], *, near_threshold: float = 0.8,
                    min_words: int = 80, min_exact_words: int = 1) -> dict[str, list[str]]:
    """Given {opportunity_id: redacted_text} for one posting, return {opportunity_id:
    [other opportunity_ids it duplicates]} for exact fingerprint matches, plus
    near-duplicates (5-gram Jaccard >= near_threshold) among texts with >= min_words
    words. Symmetric. Opportunities with no duplicates are absent.

    Texts with fewer than ``min_exact_words`` words (by default: empty ones, e.g. no
    resume) never match anything. Lists follow the input order. Near-duplicates are
    pairwise (fine for a posting's few hundred applicants), with a size filter: sets
    whose sizes differ by more than the threshold allows can't reach it.
    """
    if not 0.0 < near_threshold <= 1.0:
        raise ValueError("near_threshold must be in (0, 1]")
    order = {opp: i for i, opp in enumerate(texts)}
    words = {opp: _words(text) for opp, text in texts.items()}
    pairs: set[tuple[str, str]] = set()

    by_print: dict[str, list[str]] = {}
    for opp, ws in words.items():
        if len(ws) >= max(1, min_exact_words):
            by_print.setdefault(fingerprint(texts[opp]), []).append(opp)
    for group in by_print.values():
        pairs.update(combinations(group, 2))

    sets = {opp: _shingles_of(ws, 5) for opp, ws in words.items() if len(ws) >= min_words}
    for a, b in combinations(sets, 2):
        if (a, b) in pairs:
            continue
        sa, sb = sets[a], sets[b]
        small, large = sorted((len(sa), len(sb)))
        if not large or small < near_threshold * large:
            continue
        if jaccard(sa, sb) >= near_threshold:
            pairs.add((a, b))

    result: dict[str, list[str]] = {}
    for a, b in pairs:
        result.setdefault(a, []).append(b)
        result.setdefault(b, []).append(a)
    return {opp: sorted(others, key=order.__getitem__)
            for opp, others in sorted(result.items(), key=lambda kv: order[kv[0]])}


def jd_overlap_counts(resume_text: str, jd_text: str, n: int = 8) -> tuple[int, int]:
    """(matched, total): how many of the job description's ``n``-gram shingles appear
    verbatim in the resume, and how many it has. HTML tags and entities in the JD (Lever
    posting content) are stripped first."""
    jd = _HTML_TAG_RE.sub(" ", html.unescape(jd_text))
    jd_shingles = shingles(jd, n)
    if not jd_shingles:
        return 0, 0
    return len(jd_shingles & shingles(resume_text, n)), len(jd_shingles)


def jd_overlap(resume_text: str, jd_text: str, n: int = 8) -> float:
    """Fraction of the job description's 8-gram shingles that appear verbatim in the
    resume (copy-pasted / keyword-stuffed JD text). 0.0 when the JD has no shingles."""
    matched, total = jd_overlap_counts(resume_text, jd_text, n)
    return matched / total if total else 0.0
