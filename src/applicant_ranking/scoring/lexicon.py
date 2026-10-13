"""Protected-characteristic lexicon, used to (a) block criteria that encode protected
traits or proxies (criteria_lint) and (b) reject model output that mentions them
(scoring.validate). The list is data (YAML) so Legal/HR can own it.

File format (packaged default: applicant_ranking/data/protected_terms.yaml):
```
version: 1
allow_phrases: ["white paper", "black-box", "cloud-native", "react native", ...]
categories:
  age: ["young", "older", "years old", "digital native", "recent graduate", ...]
  gender: ["female", "male", "he", "she", "his", "her", "maternity", ...]
  ...
```
Matching: case-insensitive, whole words/phrases (``\\b`` boundaries; phrase whitespace
matches any whitespace/hyphen run). Entries prefixed ``re:`` are regexes. Before
scanning, every ``allow_phrases`` occurrence is blanked out so "white paper" or
"React Native" never hit. Categories (all required in the default file): age, gender,
race_ethnicity, national_origin, citizenship, religion, disability_health, pregnancy,
marital_family, sexual_orientation, veteran_military, genetic, employment_gap, location_proxy.

Details:
- Text is NFKC-normalised and stripped of zero-width/format characters before matching,
  so "h\\u200be" or full-width letters cannot slip past.
- ``re:`` entries are matched exactly as written (with IGNORECASE); add ``\\b`` yourself.
- Allow phrases are blanked with a character that is neither a word character, a space
  nor a hyphen, so a phrase entry can never match across a blanked span.
- ``scan`` reports each (category, term) once, at its first occurrence; a hit wholly
  inside a longer hit of the same category is dropped ("older workers" does not also
  report "older").
- The file may carry a free-form ``metadata`` mapping (owner, review dates); any other
  unknown top-level key is an error, so a typo cannot silently disable a section.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from applicant_ranking.config import ConfigError

logger = logging.getLogger(__name__)

REQUIRED_CATEGORIES: tuple[str, ...] = (
    "age",
    "gender",
    "race_ethnicity",
    "national_origin",
    "citizenship",
    "religion",
    "disability_health",
    "pregnancy",
    "marital_family",
    "sexual_orientation",
    "veteran_military",
    "genetic",
    "employment_gap",
    "location_proxy",
)

DEFAULT_RESOURCE_PACKAGE = "applicant_ranking.data"
DEFAULT_RESOURCE_NAME = "protected_terms.yaml"

_KNOWN_KEYS = frozenset({"version", "allow_phrases", "categories", "metadata"})
_SEPARATOR_CHARS = r"\s\-\u2010-\u2015\u2212"
_SEPARATOR = f"[{_SEPARATOR_CHARS}]+"
_SPLIT_RE = re.compile(_SEPARATOR)
# Zero-width and bidi/format characters that would otherwise break up a word.
_INVISIBLE_RE = re.compile("[\u00ad\u180e\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
_FOLD = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u02bc": "'", "\u2032": "'",
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u2015": "-", "\u2212": "-",
})
_BLANK = "\u2588"  # neither \w, whitespace nor a hyphen
_CONTEXT_CHARS = 60


class LexiconError(ConfigError):
    """The lexicon file or its entries are invalid."""


class LexiconHit(BaseModel):
    category: str
    term: str
    context: str  # up to ~60 chars around the hit, for logs (never sent anywhere)
    # Character offsets of the hit in the NFKC-normalised, cleaned text.
    start: int = 0
    end: int = 0

    @property
    def label(self) -> str:
        return f"{self.category}:{self.term}"


@dataclass(frozen=True, slots=True)
class _Entry:
    category: str
    term: str
    pattern: re.Pattern[str]
    is_regex: bool


def normalize_for_scan(text: str) -> str:
    """The text ``scan`` matches against: NFKC, invisible characters removed, typographic
    apostrophes and dashes folded to ASCII (so ``re:`` entries can use ``'`` and ``-``)."""
    clean = _INVISIBLE_RE.sub("", unicodedata.normalize("NFKC", text))
    return clean.translate(_FOLD)


def phrase_key(phrase: str) -> str:
    """Canonical form of a term for comparisons: ``normalize_for_scan``, lowercased, every
    whitespace/hyphen run collapsed to one space, so "Credit  History" and "credit-history"
    compare equal (the same equivalence phrase entries match with)."""
    return " ".join(p for p in _SPLIT_RE.split(normalize_for_scan(phrase).lower()) if p)


def _phrase_regex(phrase: str) -> str:
    parts = [p for p in _SPLIT_RE.split(phrase.strip()) if p]
    if not parts:
        return ""
    body = _SEPARATOR.join(re.escape(p) for p in parts)
    head = r"(?<!\w)" if re.match(r"\w", parts[0]) else ""
    tail = r"(?!\w)" if re.search(r"\w$", parts[-1]) else ""
    return f"{head}{body}{tail}"


def _compile(entry: str, *, where: str) -> tuple[re.Pattern[str], bool]:
    if not isinstance(entry, str) or not entry.strip():
        raise LexiconError(f"{where}: entries must be non-empty strings (got {entry!r})")
    is_regex = entry.startswith("re:")
    source = entry[3:].strip() if is_regex else _phrase_regex(normalize_for_scan(entry))
    if not source:
        raise LexiconError(f"{where}: {entry!r} has no characters to match")
    try:
        pattern = re.compile(source, re.IGNORECASE)
    except re.error as exc:
        raise LexiconError(f"{where}: invalid regex {entry!r}: {exc}") from exc
    if pattern.search("") is not None:
        raise LexiconError(f"{where}: {entry!r} matches the empty string")
    return pattern, is_regex


def _context(text: str, start: int, end: int) -> str:
    pad = max(0, (_CONTEXT_CHARS - (end - start)) // 2)
    snippet = " ".join(text[max(0, start - pad):end + pad].split())
    return snippet[:_CONTEXT_CHARS]


class ProtectedTermLexicon:
    def __init__(self, categories: dict[str, list[str]], allow_phrases: list[str],
                 version: int = 1, *, source: str = "inline") -> None:
        if not isinstance(categories, dict) or not categories:
            raise LexiconError("categories must be a non-empty mapping of name -> terms")
        if not isinstance(allow_phrases, list):
            raise LexiconError("allow_phrases must be a list of strings")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise LexiconError(f"version must be a positive integer (got {version!r})")
        self.version = version
        self.source = source
        self._categories: dict[str, list[str]] = {}
        self._entries: list[_Entry] = []
        for category, terms in categories.items():
            if not isinstance(category, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", category):
                raise LexiconError(f"invalid category name {category!r} (use snake_case)")
            if not isinstance(terms, list) or not terms:
                raise LexiconError(f"categories.{category}: must be a non-empty list of terms")
            self._categories[category] = list(terms)
            for term in terms:
                pattern, is_regex = _compile(term, where=f"categories.{category}")
                self._entries.append(_Entry(category, term, pattern, is_regex))
        self._allow_phrases = list(allow_phrases)
        allow_sources: list[str] = []
        for phrase in allow_phrases:
            pattern, _ = _compile(phrase, where="allow_phrases")
            allow_sources.append(pattern.pattern)
        # Longest first so the widest allow phrase wins at a given position.
        allow_sources.sort(key=len, reverse=True)
        self._allow = (
            re.compile("|".join(f"(?:{s})" for s in allow_sources), re.IGNORECASE)
            if allow_sources else None
        )

    @classmethod
    def load(cls, path: Path | None = None) -> ProtectedTermLexicon:
        """Load ``path`` or the packaged default (importlib.resources).

        Every category in ``REQUIRED_CATEGORIES`` must be present, in a custom file too:
        a replacement list that silently drops a protected class must not load.
        """
        if path is None:
            source = f"package:{DEFAULT_RESOURCE_PACKAGE}/{DEFAULT_RESOURCE_NAME}"
            try:
                raw = (resources.files(DEFAULT_RESOURCE_PACKAGE)
                       .joinpath(DEFAULT_RESOURCE_NAME).read_text(encoding="utf-8"))
            except (OSError, ModuleNotFoundError) as exc:
                raise LexiconError(f"packaged protected-term list is missing: {exc}") from exc
        else:
            source = str(path)
            try:
                raw = Path(path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise LexiconError(f"{path}: cannot read protected-term list: {exc}") from exc
        return cls._from_yaml(raw, source=source)

    @classmethod
    def _from_yaml(cls, raw: str, *, source: str) -> ProtectedTermLexicon:
        try:
            data: Any = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise LexiconError(f"{source}: invalid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise LexiconError(f"{source}: expected a mapping with 'categories'")
        unknown = sorted(str(k) for k in data if k not in _KNOWN_KEYS)
        if unknown:
            raise LexiconError(f"{source}: unknown top-level keys: {', '.join(unknown)}")
        categories = data.get("categories")
        if not isinstance(categories, dict):
            raise LexiconError(f"{source}: 'categories' must be a mapping")
        missing = [c for c in REQUIRED_CATEGORIES if c not in categories]
        if missing:
            raise LexiconError(f"{source}: missing required categories: {', '.join(missing)}")
        allow = data.get("allow_phrases") or []
        try:
            lexicon = cls(categories, allow, data.get("version", 1), source=source)
        except LexiconError as exc:
            raise LexiconError(f"{source}: {exc}") from exc
        logger.debug("loaded protected-term lexicon v%s from %s (%d terms)",
                     lexicon.version, source, lexicon.term_count)
        return lexicon

    @property
    def categories(self) -> dict[str, list[str]]:
        return {name: list(terms) for name, terms in self._categories.items()}

    @property
    def allow_phrases(self) -> list[str]:
        return list(self._allow_phrases)

    @property
    def term_count(self) -> int:
        return len(self._entries)

    def blank_allowed(self, text: str) -> str:
        """``text`` (already normalised) with every allow-phrase occurrence blanked;
        same length, so offsets still line up."""
        if self._allow is None:
            return text
        return self._allow.sub(lambda m: _BLANK * len(m.group()), text)

    def scan(self, text: str) -> list[LexiconHit]:
        if not text or not text.strip():
            return []
        clean = normalize_for_scan(text)
        target = self.blank_allowed(clean)
        found: list[tuple[int, int, _Entry]] = []
        for entry in self._entries:
            for match in entry.pattern.finditer(target):
                if match.end() > match.start():
                    found.append((match.start(), match.end(), entry))
        if not found:
            return []
        # Widest first at each position, so contained hits of the same category drop out.
        found.sort(key=lambda f: (f[0], -(f[1] - f[0])))
        hits: list[LexiconHit] = []
        covered: list[tuple[str, int, int]] = []
        seen: set[tuple[str, str]] = set()
        for start, end, entry in found:
            if any(cat == entry.category and s <= start and end <= e for cat, s, e in covered):
                continue
            covered.append((entry.category, start, end))
            # A regex entry is reported by what it matched (readable), a phrase by itself.
            term = " ".join(clean[start:end].lower().split())[:60] if entry.is_regex \
                else entry.term
            key = (entry.category, term.lower())
            if key in seen:
                continue
            seen.add(key)
            hits.append(LexiconHit(category=entry.category, term=term,
                                   context=_context(clean, start, end), start=start, end=end))
        return hits

    def __repr__(self) -> str:
        return (f"ProtectedTermLexicon(version={self.version}, source={self.source!r}, "
                f"categories={len(self._categories)}, terms={len(self._entries)})")
