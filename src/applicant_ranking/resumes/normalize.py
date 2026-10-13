"""Normalise extracted resume text before any check or LLM call (same steps for every
file type). Order matters:

1. Decode Unicode tag characters U+E0000-E007F into ``tag_payload`` (ASCII smuggling),
   then strip them — except the RGI subdivision-flag sequences (U+1F3F4 + tag spec for
   gbeng/gbsct/gbwls + U+E007F), which are kept.
2. Strip: Cc controls except \\t \\n; Cf format chars (ZWSP, ZWJ, ZWNJ, soft hyphen,
   bidi controls U+202A-202E, U+2066-2069, U+200E/F, BOM); supplementary variation
   selectors U+E0100-E01EF; private-use glyphs (U+E000-F8FF, planes 15-16); blank-looking
   non-Cf characters U+115F, U+1160, U+3164, U+FFA0, U+2800. Count removals by category
   in ``removed`` (keys: "tag_chars", "bidi_controls", "zero_width", "control",
   "variation_selectors", "private_use", "blank_fillers", "other_format").
   Exceptions for scripts that write joiners inside words: a ZWNJ/ZWJ between two
   letters or marks of a joiner-using script (Arabic, Syriac, Hebrew, the Indic scripts
   from Devanagari to Malayalam, Sinhala, Tibetan, Myanmar, Khmer, Mongolian) is
   orthography (Persian "دادهU+200Cها", Devanagari conjuncts) and is kept without being
   counted; a directional mark U+200E/U+200F/U+061C on a line that contains Hebrew or
   Arabic letters is layout (it keeps Latin terms readable in right-to-left text),
   stripped but counted as "other_format". Next to Latin letters, spaces or punctuation,
   or on a line without right-to-left letters, these characters count as before.
3. NFKC (ligatures, fullwidth, math-bold -> ASCII).
4. Bullets (•, ▪, ◦, ●, ‣, ⁃, -, *) at line start -> "- ".
5. Join lower-case hyphenated line breaks: "experi-\\nence" -> "experience", unless the
   left fragment already contains a hyphen ("end-to-\\nend"), starts with a capital
   ("Data-\\ndriven") or is a compound-forming word common in resumes ("cross-", "self-",
   "on-", "real-", "front-", ...; see ``COMPOUND_PREFIXES``), which keep their hyphen.
6. Collapse runs of spaces/tabs to one space; 3+ newlines to 2; strip line ends.
7. Truncate to ``max_chars``: keep the first 80% and the last 20% with a marker line
   "[... N characters omitted ...]" between; set ``truncated``.

Details beyond the list above:
- Line-break controls (CR, CRLF, VT, FF, NEL, U+2028, U+2029) become "\\n" first; they
  are layout, not hidden content, and are not counted.
- Tag characters separated only by other stripped characters decode as one run (so a
  ZWSP between tag characters can't split the payload); separate runs are joined with a
  space. Only U+E0020-E007E decode (they mirror printable ASCII).
- Also stripped, as known invisible characters: U+2061-2064 (invisible operators),
  U+034F (combining grapheme joiner) and U+180E as "zero_width"; U+17B4/U+17B5 as
  "blank_fillers"; BMP variation selectors U+FE00-FE0F and Mongolian free variation
  selectors as "variation_selectors" (the "emoji smuggling" channel uses both planes);
  U+061C (Arabic letter mark) as "bidi_controls"; lone surrogates and noncharacters as
  "other_format".
- A ZWJ between two emoji (an emoji ZWJ sequence such as a profession emoji) is still
  removed but counted as "other_format", so emoji-heavy resumes don't look like
  zero-width smuggling.
- ``removed`` holds only non-zero counts. ``original_length`` is the length of the input.
- The truncated text, marker included, is never longer than ``max_chars``; the cut
  points move back to a line or word boundary when one is near.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from applicant_ranking.domain import NormalizedText

HEAD_SHARE = 0.8
MIN_MAX_CHARS = 200
_SNAP_WINDOW = 200

_TAG_FIRST = 0xE0000
_TAG_LAST = 0xE007F
_TAG_PRINTABLE_FIRST = 0xE0020
_TAG_PRINTABLE_LAST = 0xE007E
_BLACK_FLAG = "\U0001F3F4"
_CANCEL_TAG = "\U000E007F"
RGI_SUBDIVISION_FLAGS = ("gbeng", "gbsct", "gbwls")


def _as_tags(ascii_text: str) -> str:
    return "".join(chr(_TAG_FIRST + ord(c)) for c in ascii_text)


_RGI_FLAG_PATTERN = "|".join(
    re.escape(_BLACK_FLAG + _as_tags(spec) + _CANCEL_TAG) for spec in RGI_SUBDIVISION_FLAGS
)
# Either a kept flag sequence or any character that might need removal (everything that
# is not tab, newline or printable ASCII). Plain ASCII text never reaches the callback.
_SCAN_RE = re.compile(f"(?P<flag>{_RGI_FLAG_PATTERN})|[^\\t\\n\\x20-\\x7e]")
_LINE_BREAK_RE = re.compile("\r\n?|[\x0b\x0c\x85  ]")

_BIDI = frozenset({0x061C, 0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x2066, 0x206A)})
_ZERO_WIDTH = frozenset({0x200B, 0x200C, 0x200D, 0xFEFF, 0x180E, 0x034F,
                         *range(0x2060, 0x2065)})
_BLANK_FILLERS = frozenset({0x115F, 0x1160, 0x3164, 0xFFA0, 0x2800, 0x17B4, 0x17B5})
_ZWJ = 0x200D
_ZWNJ = 0x200C
_DIRECTIONAL_MARKS = frozenset({0x200E, 0x200F, 0x061C})
_VS16 = 0xFE0F
# Scripts whose orthography puts a ZWNJ/ZWJ inside words (Persian, Indic conjuncts, ...).
_JOINER_SCRIPT_RANGES = (
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),  # Arabic
    (0x0700, 0x074F),  # Syriac
    (0x0900, 0x0D7F),  # Devanagari through Malayalam
    (0x0D80, 0x0DFF),  # Sinhala
    (0x0F00, 0x0FFF),  # Tibetan
    (0x1000, 0x109F),  # Myanmar
    (0x1780, 0x17FF),  # Khmer
    (0x1800, 0x18AF),  # Mongolian
)
_RTL_LETTER_RANGES = ((0x0590, 0x05FF), (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
                      (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
# Words that keep their hyphen when a line break follows it ("cross-\nfunctional").
COMPOUND_PREFIXES = frozenset(
    "cross self end on off real full front back data non co well high low long short large "
    "small open test user time cost state event multi pre post e out in".split())
_WORD_CHARS_RE = re.compile(r"[^\W\d_]")

_BULLET_CHARS = "•▪◦●‣⁃∙■□◆◇○➢➤►▸▹✓✔❖★·‧"
_BULLET_ONLY_LINE_RE = re.compile(
    rf"^[^\S\n]*[{_BULLET_CHARS}]+[^\S\n]*\n(?=[^\S\n]*\S)", re.M
)
_BULLET_RE = re.compile(
    rf"^[^\S\n]*(?:[{_BULLET_CHARS}]+[^\S\n]*|[-*–—][^\S\n]+)(?=\S)", re.M
)
_HYPHEN_BREAK_RE = re.compile(r"(?<=[^\W\d_])-[^\S\n]*\n[^\S\n]*(?=[^\W\d_])")
_HSPACE_RE = re.compile(r"[^\S\n]+")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")


def _is_noncharacter(cp: int) -> bool:
    return 0xFDD0 <= cp <= 0xFDEF or (cp & 0xFFFE) == 0xFFFE


def _is_variation_selector(cp: int) -> bool:
    return (0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF
            or 0x180B <= cp <= 0x180D or cp == 0x180F)


def _is_emoji_like(ch: str) -> bool:
    cp = ord(ch)
    return (0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF or cp == _VS16
            or unicodedata.category(ch) == "So")


def _in_ranges(ch: str, ranges: tuple[tuple[int, int], ...]) -> bool:
    cp = ord(ch)
    return any(low <= cp <= high for low, high in ranges)


def _joiner_script_char(ch: str) -> bool:
    """A letter or combining mark of a script that writes joiners inside words (a
    Devanagari conjunct puts the joiner right after the virama, a mark)."""
    return _in_ranges(ch, _JOINER_SCRIPT_RANGES) and (
        ch.isalpha() or unicodedata.category(ch).startswith("M"))


def _rtl_letter(ch: str) -> bool:
    return _in_ranges(ch, _RTL_LETTER_RANGES) and ch.isalpha()


def _category(ch: str) -> str | None:
    """Removal category for one character, or None to keep it (ZWJ handled by caller)."""
    cp = ord(ch)
    if _TAG_FIRST <= cp <= _TAG_LAST:
        return "tag_chars"
    if cp in _BIDI:
        return "bidi_controls"
    if cp in _ZERO_WIDTH:
        return "zero_width"
    if cp in _BLANK_FILLERS:
        return "blank_fillers"
    if _is_variation_selector(cp):
        return "variation_selectors"
    cat = unicodedata.category(ch)
    if cat == "Cc":
        return "control"
    if cat == "Co":
        return "private_use"
    if cat in ("Cf", "Cs") or _is_noncharacter(cp):
        return "other_format"
    return None


class _Stripper:
    """Step 1 and 2 in a single regex pass (state carries the current tag run)."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.removed: Counter[str] = Counter()
        self.runs: list[str] = []
        self._run: list[str] = []
        self._last_end = 0
        self._rtl_lines: dict[int, bool] = {}  # line start -> holds right-to-left letters

    def _close_run(self) -> None:
        if self._run:
            self.runs.append("".join(self._run))
            self._run = []

    def __call__(self, m: re.Match[str]) -> str:
        if m.start() != self._last_end:  # visible ASCII in between ends a tag run
            self._close_run()
        self._last_end = m.end()
        if m.group("flag"):
            self._close_run()
            return m.group()
        ch = m.group()
        cp = ord(ch)
        category = _category(ch)
        if category is None:
            self._close_run()
            return ch
        if cp in (_ZWJ, _ZWNJ) and self._inside_joiner_word(m.start(), m.end()):
            self._close_run()
            return ch  # orthographic joiner: kept and not counted
        if cp == _ZWJ and self._between_emoji(m.start(), m.end()):
            category = "other_format"
        elif cp in _DIRECTIONAL_MARKS and self._on_rtl_line(m.start()):
            category = "other_format"  # a layout mark in right-to-left text
        self.removed[category] += 1
        if category == "tag_chars" and _TAG_PRINTABLE_FIRST <= cp <= _TAG_PRINTABLE_LAST:
            self._run.append(chr(cp - _TAG_FIRST))
        return ""

    def _between_emoji(self, start: int, end: int) -> bool:
        return (start > 0 and end < len(self.text)
                and _is_emoji_like(self.text[start - 1]) and _is_emoji_like(self.text[end]))

    def _inside_joiner_word(self, start: int, end: int) -> bool:
        return (start > 0 and end < len(self.text)
                and _joiner_script_char(self.text[start - 1])
                and _joiner_script_char(self.text[end]))

    def _on_rtl_line(self, position: int) -> bool:
        line_start = self.text.rfind("\n", 0, position) + 1
        if line_start not in self._rtl_lines:
            line_end = self.text.find("\n", position)
            line = self.text[line_start: len(self.text) if line_end < 0 else line_end]
            self._rtl_lines[line_start] = any(_rtl_letter(ch) for ch in line)
        return self._rtl_lines[line_start]

    def run(self) -> str:
        out = _SCAN_RE.sub(self, self.text)
        self._close_run()
        return out


def _join_hyphenated(m: re.Match[str]) -> str:
    """Drop a line-end hyphen that split a word, keep one that joins a compound."""
    s = m.string
    if not (s[m.start() - 1].islower() and s[m.end()].islower()):
        return m.group()
    start = m.start()
    while start > 0 and (_WORD_CHARS_RE.match(s[start - 1]) or s[start - 1] == "-"):
        start -= 1
    fragment = s[start:m.start()]
    if "-" in fragment or fragment[:1].isupper() or fragment.casefold() in COMPOUND_PREFIXES:
        return "-"
    return ""


def _marker(omitted: int) -> str:
    return f"[... {omitted} characters omitted ...]"


def _truncate(text: str, max_chars: int) -> str:
    # Reserve room for the marker with the largest count it could show.
    budget = max_chars - len(_marker(len(text))) - 2
    head_len = int(budget * HEAD_SHARE)
    tail_len = budget - head_len
    head = text[:head_len]
    tail = text[len(text) - tail_len:]
    window = min(_SNAP_WINDOW, head_len // 10)
    for sep in ("\n", " "):
        cut = head.rfind(sep, head_len - window)
        if cut > 0:
            head = head[:cut]
            break
    window = min(_SNAP_WINDOW, tail_len // 10)
    for sep in ("\n", " "):
        cut = tail.find(sep, 0, window)
        if cut >= 0:
            tail = tail[cut + 1:]
            break
    head, tail = head.rstrip(), tail.lstrip()
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n{_marker(omitted)}\n{tail}"


def normalize_text(text: str, *, max_chars: int = 30_000) -> NormalizedText:
    """Normalise ``text`` (see the module docstring). ``max_chars`` must be >= 200."""
    if max_chars < MIN_MAX_CHARS:
        raise ValueError(f"max_chars must be >= {MIN_MAX_CHARS}")
    original_length = len(text)
    s = _LINE_BREAK_RE.sub("\n", text)
    stripper = _Stripper(s)
    s = stripper.run()
    s = unicodedata.normalize("NFKC", s)
    s = _BULLET_ONLY_LINE_RE.sub("- ", s)
    s = _BULLET_RE.sub("- ", s)
    s = _HYPHEN_BREAK_RE.sub(_join_hyphenated, s)
    s = "\n".join(_HSPACE_RE.sub(" ", line).strip() for line in s.split("\n"))
    s = _MANY_NEWLINES_RE.sub("\n\n", s).strip()
    truncated = len(s) > max_chars
    if truncated:
        s = _truncate(s, max_chars)
    payload = _HSPACE_RE.sub(" ", " ".join(stripper.runs)).strip()[:max_chars]
    return NormalizedText(
        text=s,
        removed={k: v for k, v in stripper.removed.items() if v},
        tag_payload=payload,
        truncated=truncated,
        original_length=original_length,
    )
