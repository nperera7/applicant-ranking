"""RTF via striprtf (pure Python). striprtf also returns \\v hidden text, so detect hidden
formatting with a regex on the raw RTF: ``\\\\v(?![a-z0-9])`` or ``\\\\fs[0-5](?!\\d)``
(font size < 3pt in half-points) -> ``hidden_text`` flag (review). The hidden text cannot
be separated reliably, so it stays in visible_text; the flag forces human review.

A small tokenizer additionally collects best-effort samples and a character count of the
hidden text (\\v, \\fs0-5, and \\cf text whose colour-table entry is white with no dark
\\cb/\\highlight/\\chcbpat background in effect). White text alone raises the flag with the
usual threshold ("review" at >= 20 characters, "info" below).
"""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass, replace

from striprtf.striprtf import rtf_to_text

from applicant_ranking.domain import ExtractionResult
from applicant_ranking.resumes.extract_text import (
    HiddenTextCollector,
    clean_unicode,
    hidden_text_flag,
    unreadable,
)

HIDDEN_FORMAT_RE = re.compile(r"\\v(?![a-z0-9])|\\fs[0-5](?!\d)")
WHITE_LUMINANCE = 0.92
DARK_BACKGROUND = 0.6

_ANSICPG_RE = re.compile(rb"\\ansicpg(\d{3,5})")
_TOKEN_RE = re.compile(
    r"\\([a-zA-Z]{1,32})(-?\d{1,10})? ?|\\'([0-9a-fA-F]{2})|\\([^a-zA-Z])|([{}])|[\r\n]+"
    r"|([^\\{}\r\n]+)"
)
_COLORTBL_RE = re.compile(r"\{\\colortbl([^{}]*)\}")
_COLOR_ENTRY_RE = re.compile(r"\\(red|green|blue)(\d{1,3})")
# Destinations whose content is never displayed as body text.
_SKIPPED_DESTINATIONS = frozenset({
    "fonttbl", "colortbl", "stylesheet", "info", "pict", "object", "fldinst", "header",
    "headerl", "headerr", "headerf", "footer", "footerl", "footerr", "footerf", "footnote",
    "annotation", "listtable", "listoverridetable", "rsidtbl", "themedata", "datastore",
    "latentstyles", "generator", "xmlnstbl", "filetbl", "revtbl", "shpinst", "nonshppict",
})

REASON_LABELS = {
    "vanish": "hidden (\\v) formatting",
    "tiny_font": "font under 3pt",
    "white_text": "white text",
}


@dataclass(frozen=True)
class _State:
    skip: bool = False
    hidden: bool = False
    tiny: bool = False
    color: int = 0
    background_dark: bool = False


def _codepage(data: bytes) -> str:
    match = _ANSICPG_RE.search(data[:4096])
    if match:
        candidate = f"cp{int(match.group(1))}"
        try:
            codecs.lookup(candidate)
        except LookupError:
            return "cp1252"
        return candidate
    return "cp1252"


def _color_table(rtf: str) -> list[float | None]:
    """Luminance of each colour-table entry (index 0 is the 'auto' colour)."""
    match = _COLORTBL_RE.search(rtf)
    if not match:
        return []
    entries: list[float | None] = []
    for entry in match.group(1).split(";")[:-1]:
        values = dict((name, int(v)) for name, v in _COLOR_ENTRY_RE.findall(entry))
        if not values:
            entries.append(None)
            continue
        r, g, b = (min(values.get(k, 0), 255) for k in ("red", "green", "blue"))
        entries.append((0.2126 * r + 0.7152 * g + 0.0722 * b) / 255)
    return entries


def _hidden_runs(rtf: str, colors: list[float | None], codepage: str) -> HiddenTextCollector:
    """Best-effort walk of the RTF group structure collecting hidden character runs."""
    collector = HiddenTextCollector()

    def luminance(index: int) -> float | None:
        return colors[index] if 0 <= index < len(colors) else None

    stack: list[_State] = []
    state = _State()
    uc_skip = 0
    for match in _TOKEN_RE.finditer(rtf):
        word, arg, hex_byte, symbol, brace, text = match.groups()
        if brace == "{":
            stack.append(state)
            continue
        if brace == "}":
            state = stack.pop() if stack else _State()
            collector.end_run()
            continue
        chunk = ""
        if word is not None:
            value = int(arg) if arg is not None else None
            if word in _SKIPPED_DESTINATIONS:
                state = replace(state, skip=True)
            elif word == "v":
                state = replace(state, hidden=value != 0)
            elif word == "fs":
                state = replace(state, tiny=value is not None and value <= 5)
            elif word == "cf":
                state = replace(state, color=value or 0)
            elif word in ("cb", "highlight", "chcbpat"):
                lum = luminance(value or 0)
                state = replace(state, background_dark=lum is not None and lum < DARK_BACKGROUND)
            elif word == "plain":
                state = replace(state, hidden=False, tiny=False, color=0, background_dark=False)
            elif word == "u" and value is not None:
                code = value + 0x10000 if value < 0 else value
                chunk = chr(code) if code < 0x110000 else ""
                uc_skip = 1
            elif word in ("par", "line", "sect", "page", "row", "cell", "tab"):
                collector.end_run()
                continue
            else:
                continue
        elif symbol is not None:
            if symbol == "*":
                state = replace(state, skip=True)
            continue
        elif hex_byte is not None:
            if uc_skip:
                uc_skip -= 1
                continue
            chunk = bytes.fromhex(hex_byte).decode(codepage, errors="replace")
        elif text is not None:
            if uc_skip:
                text = text[uc_skip:]
                uc_skip = 0
            chunk = text
        if not chunk or state.skip:
            continue
        lum = luminance(state.color)
        white = lum is not None and lum > WHITE_LUMINANCE and not state.background_dark
        if state.hidden:
            collector.add(chunk, "vanish")
        elif state.tiny:
            collector.add(chunk, "tiny_font")
        elif white:
            collector.add(chunk, "white_text")
        elif chunk.strip():
            collector.end_run()
    collector.end_run()
    return collector


def extract_rtf(data: bytes) -> ExtractionResult:
    codepage = _codepage(data)
    rtf = data.decode(codepage, errors="replace")
    try:
        text = rtf_to_text(rtf, encoding=codepage, errors="replace")
    except Exception as exc:  # striprtf raises e.g. ValueError on malformed control words
        return unreadable("rtf", f"could not read RTF file ({type(exc).__name__})")
    marked_hidden = HIDDEN_FORMAT_RE.search(rtf) is not None
    try:
        hidden = _hidden_runs(rtf, _color_table(rtf), codepage)
    except Exception:  # best-effort only; the regex flag above still applies
        hidden = HiddenTextCollector()
    detail = hidden.describe(REASON_LABELS) if hidden.char_count else (
        "hidden-text formatting found")
    flag = hidden_text_flag(
        hidden.char_count, force_review=marked_hidden,
        detail=f"{detail}; the RTF text could not be separated and is kept as is",
    )
    return ExtractionResult(
        kind="rtf",
        visible_text=clean_unicode(text),
        hidden_snippets=[clean_unicode(s) for s in hidden.snippets],
        hidden_char_count=hidden.char_count,
        flags=[flag] if flag is not None else [],
    )
