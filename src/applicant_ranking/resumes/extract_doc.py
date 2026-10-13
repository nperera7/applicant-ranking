"""Legacy Word 97-2003 (.doc) text via olefile and the [MS-DOC] piece table, no system
binaries. Algorithm: WordDocument stream FIB magic 0xA5EC; flags at 0x0A (0x0100 =
encrypted -> error; 0x0200 = use the 1Table stream, else 0Table); fcClx/lcbClx at 0x1A2;
skip Prc entries (0x01), read Pcdt (0x02) -> PlcPcd CPs and 8-byte PCDs; if
FcCompressed (0x40000000) decode cp1252 at fc/2, else UTF-16LE at fc. Drop field codes
between 0x13 and 0x14 (keep results up to 0x15); map \\r, \\x07, \\x0b, \\x0c to newline/tab.

Hidden text: the character formatting (CHPX) is read from the formatted-disk-page table
(PlcfBteChpx, FibRgFcLcb97 index 12 -> 512-byte FKPs in the WordDocument stream) and
every run's grpprl is scanned for sprmCFVanish / sprmCFSpecVanish (vanish), sprmCHps
under 6 half-points (tiny), sprmCCharScale under 20% (squashed) and a white colour
(sprmCIco 8 or sprmCCv with luminance > 0.92) without shading or highlight in the same
run (white). Those runs are mapped through the piece table to character positions,
removed from the text and reported like PDF/DOCX hidden text (``hidden_text``, review at
>= 20 characters). Only direct character formatting is covered: hiding inherited from a
character or paragraph style, or white text on paragraph shading, is not detected, which
the ``hidden_text_check_unavailable`` info flag keeps saying.

The FIB offsets are computed from its csw/cslw counts (0x1A2 and 0x4C for every Word 97+
file). Stories read: main text, footnotes, headers, endnotes and text boxes (not comments).
"""

from __future__ import annotations

import io
import logging
import struct
from collections.abc import Iterator

import olefile

from applicant_ranking.domain import ExtractionResult, Flag, FlagCode
from applicant_ranking.resumes.extract_text import HiddenTextCollector, unreadable

logger = logging.getLogger(__name__)

FIB_MAGIC = 0xA5EC
MIN_NFIB = 0x00C1  # Word 97; Word 6/95 files use a different FIB layout
FLAG_ENCRYPTED = 0x0100
FLAG_TABLE1 = 0x0200
FC_COMPRESSED = 0x40000000
MAX_DOC_CHARS = 5_000_000
FKP_SIZE = 512
MAX_FKP_PAGES = 4096  # 2 MB of formatting pages is far beyond any resume

# fibRgLw indexes of the story lengths (ccp*), in CP order after the main text.
_CCP_TEXT, _CCP_FTN, _CCP_HDD, _CCP_MCR, _CCP_ATN, _CCP_EDN, _CCP_TXBX, _CCP_HDR_TXBX = (
    3, 4, 5, 6, 7, 8, 9, 10)
_INCLUDED_STORIES = (_CCP_TEXT, _CCP_FTN, _CCP_HDD, _CCP_EDN, _CCP_TXBX, _CCP_HDR_TXBX)
_FC_CLX_INDEX = 33  # fcClx/lcbClx pair in FibRgFcLcb97
_FC_PLCF_BTE_CHPX_INDEX = 12  # fcPlcfBteChpx/lcbPlcfBteChpx pair in FibRgFcLcb97

# Character property modifiers ([MS-DOC] 2.6.1) that hide or nearly hide a run.
SPRM_CF_VANISH = 0x083C
SPRM_CF_SPEC_VANISH = 0x0818
SPRM_C_ICO = 0x2A42
SPRM_C_CV = 0x6870
SPRM_C_HPS = 0x4A43
SPRM_C_CHAR_SCALE = 0x4852
SPRM_T_DEF_TABLE = 0xD608
_SHADING_SPRMS = frozenset({0x4866, 0xCA71, 0x2A0C})  # sprmCShd80, sprmCShd, sprmCHighlight
ICO_WHITE = 8
MIN_HALF_POINTS = 6  # 3pt
MIN_CHAR_SCALE = 20  # percent
WHITE_LUMINANCE = 0.92
DOC_REASON_LABELS = {
    "vanish": "hidden (vanish) formatting",
    "tiny_font": "font under 3pt",
    "squashed": "character width under 20%",
    "white_text": "white text without shading",
}
CHECK_SCOPE_NOTE = ("Word 97-2003: only direct character formatting (hidden, white, tiny or "
                    "squashed runs) is checked; hiding inherited from styles is not detected")

_CONTROL_MAP = str.maketrans({
    "\r": "\n", "\x07": "\t", "\x0b": "\n", "\x0c": "\n", "\x1e": "-", "\x1f": "",
})


class DocFormatError(ValueError):
    """The stream is not a Word 97-2003 document this parser can read."""


class DocEncryptedError(DocFormatError):
    """The .doc is password-protected."""


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _fib_layout(word_document: bytes) -> tuple[list[int], int]:
    """(fibRgLw values, offset of FibRgFcLcb) located via the FIB's csw/cslw counts."""
    csw = _u16(word_document, 0x20)
    rg_lw = 0x22 + csw * 2 + 2
    cslw = _u16(word_document, rg_lw - 2)
    if cslw <= _CCP_HDR_TXBX:
        raise DocFormatError("FIB too short")
    longs = list(struct.unpack_from(f"<{cslw}i", word_document, rg_lw))
    return longs, rg_lw + cslw * 4 + 2


def _fc_lcb(word_document: bytes, rg_fclcb: int, index: int) -> tuple[int, int]:
    cb_rg_fclcb = _u16(word_document, rg_fclcb - 2)
    if cb_rg_fclcb <= index:
        raise DocFormatError("FIB too short")
    return struct.unpack_from("<II", word_document, rg_fclcb + index * 8)


def _piece_table(table: bytes, fc_clx: int, lcb_clx: int) -> tuple[list[int], list[int]]:
    """(character positions, raw fc values) of the PlcPcd in the Clx."""
    clx = table[fc_clx: fc_clx + lcb_clx]
    i = 0
    while i < len(clx) and clx[i] == 0x01:  # Prc: grpprl of property modifiers
        i += 3 + _u16(clx, i + 1)
    if i >= len(clx) or clx[i] != 0x02:
        raise DocFormatError("no piece table")
    lcb = struct.unpack_from("<I", clx, i + 1)[0]
    plc = clx[i + 5: i + 5 + lcb]
    if lcb < 16 or len(plc) < lcb or (lcb - 4) % 12:
        raise DocFormatError("malformed piece table")
    count = (lcb - 4) // 12
    cps = list(struct.unpack_from(f"<{count + 1}i", plc, 0))
    fcs = [struct.unpack_from("<I", plc, 4 * (count + 1) + 8 * k + 2)[0] for k in range(count)]
    return cps, fcs


# ---- character formatting -------------------------------------------------------------


def _iter_prls(grpprl: bytes) -> Iterator[tuple[int, bytes]]:
    """(sprm, operand) pairs of a property-modifier list; the operand size follows the
    sprm's spra bits."""
    i = 0
    while i + 2 <= len(grpprl):
        sprm = _u16(grpprl, i)
        i += 2
        spra = sprm >> 13
        if spra in (0, 1):
            size = 1
        elif spra in (2, 4, 5):
            size = 2
        elif spra == 7:
            size = 3
        elif spra == 3:
            size = 4
        elif sprm == SPRM_T_DEF_TABLE:
            size = 2 + (_u16(grpprl, i) if i + 2 <= len(grpprl) else 0)
        else:  # variable length: one byte of count, then the bytes
            size = 1 + (grpprl[i] if i < len(grpprl) else 0)
        yield sprm, grpprl[i: i + size]
        i += size


def _grpprl_reason(grpprl: bytes) -> str | None:
    """Why a run with these property modifiers is hidden, or None."""
    reasons: set[str] = set()
    white = False
    shaded = False
    for sprm, operand in _iter_prls(grpprl):
        if sprm in (SPRM_CF_VANISH, SPRM_CF_SPEC_VANISH) and operand[:1] == b"\x01":
            reasons.add("vanish")
        elif sprm == SPRM_C_HPS and len(operand) == 2 and _u16(operand, 0) < MIN_HALF_POINTS:
            reasons.add("tiny_font")
        elif sprm == SPRM_C_CHAR_SCALE and len(operand) == 2 and _u16(operand, 0) < MIN_CHAR_SCALE:
            reasons.add("squashed")
        elif sprm == SPRM_C_ICO and operand[:1] == bytes([ICO_WHITE]):
            white = True
        elif sprm == SPRM_C_CV and len(operand) == 4 and operand[3] == 0:
            r, g, b = operand[0], operand[1], operand[2]  # COLORREF: red, green, blue, fAuto
            white = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255 > WHITE_LUMINANCE
        elif sprm in _SHADING_SPRMS:
            shaded = True
    for reason in ("vanish", "tiny_font", "squashed"):
        if reason in reasons:
            return reason
    return "white_text" if white and not shaded else None


def _formatting_runs(word_document: bytes, table: bytes, rg_fclcb: int
                     ) -> list[tuple[int, int, str]]:
    """(fc_start, fc_end, reason) for every character run with hiding formatting."""
    fc_bte, lcb_bte = _fc_lcb(word_document, rg_fclcb, _FC_PLCF_BTE_CHPX_INDEX)
    if lcb_bte < 12 or (lcb_bte - 4) % 8 or fc_bte + lcb_bte > len(table):
        return []
    count = (lcb_bte - 4) // 8
    pages = struct.unpack_from(f"<{count}I", table, fc_bte + 4 * (count + 1))
    runs: list[tuple[int, int, str]] = []
    for pn in pages[:MAX_FKP_PAGES]:
        start = pn * FKP_SIZE
        page = word_document[start: start + FKP_SIZE]
        if len(page) < FKP_SIZE:
            continue
        crun = page[FKP_SIZE - 1]
        if crun == 0 or 4 * (crun + 1) + crun > FKP_SIZE - 1:
            continue
        rgfc = struct.unpack_from(f"<{crun + 1}I", page, 0)
        rgb_offset = 4 * (crun + 1)
        for i in range(crun):
            offset = page[rgb_offset + i] * 2
            if offset == 0 or offset >= FKP_SIZE - 1:
                continue  # default character properties
            cb = page[offset]
            grpprl = page[offset + 1: offset + 1 + cb]
            reason = _grpprl_reason(grpprl)
            if reason is not None and rgfc[i] < rgfc[i + 1]:
                runs.append((rgfc[i], rgfc[i + 1], reason))
    return runs


def _hidden_cp_spans(runs: list[tuple[int, int, str]], cps: list[int], fcs: list[int]
                     ) -> list[tuple[int, int, str]]:
    """Map hidden FC ranges through the piece table to [cp_start, cp_end) spans."""
    spans: list[tuple[int, int, str]] = []
    for k, fc_raw in enumerate(fcs):
        count = cps[k + 1] - cps[k]
        if count <= 0:
            continue
        if fc_raw & FC_COMPRESSED:
            piece_start, bytes_per_char = (fc_raw & ~FC_COMPRESSED) // 2, 1
        else:
            piece_start, bytes_per_char = fc_raw, 2
        piece_end = piece_start + count * bytes_per_char
        for fc_start, fc_end, reason in runs:
            low, high = max(fc_start, piece_start), min(fc_end, piece_end)
            if low < high:
                cp_low = cps[k] + (low - piece_start) // bytes_per_char
                cp_high = cps[k] + -(-(high - piece_start) // bytes_per_char)
                spans.append((cp_low, cp_high, reason))
    return spans


def _split_hidden(story: str, reasons: list[str | None], collector: HiddenTextCollector) -> str:
    """Visible characters of ``story``; hidden ones (except field/paragraph control
    characters, which keep the structure intact) go to ``collector``."""
    visible: list[str] = []
    fragment: list[str] = []
    fragment_reason: str | None = None

    def flush() -> None:
        nonlocal fragment, fragment_reason
        if fragment:
            collector.add("".join(fragment), fragment_reason or "vanish")
        fragment, fragment_reason = [], None

    for ch, reason in zip(story, reasons, strict=True):
        if reason is None or ch < " ":
            if reason is None:
                flush()
                collector.end_run()
            visible.append(ch)
            continue
        if fragment_reason not in (None, reason):
            flush()
        fragment_reason = reason
        fragment.append(ch)
    flush()
    collector.end_run()
    return "".join(visible)


def _doc_text(word_document: bytes, table: bytes) -> tuple[str, HiddenTextCollector]:
    """(document text, hidden text) from the WordDocument and table streams (raises
    DocFormatError)."""
    try:
        if len(word_document) < 0x22 or _u16(word_document, 0) != FIB_MAGIC:
            raise DocFormatError("bad FIB magic")
        if _u16(word_document, 2) < MIN_NFIB:
            raise DocFormatError("unsupported Word version (before Word 97)")
        if _u16(word_document, 0x0A) & FLAG_ENCRYPTED:
            raise DocEncryptedError("encrypted .doc")
        longs, rg_fclcb = _fib_layout(word_document)
        fc_clx, lcb_clx = _fc_lcb(word_document, rg_fclcb, _FC_CLX_INDEX)
        cps, fcs = _piece_table(table, fc_clx, lcb_clx)
    except struct.error as exc:
        raise DocFormatError("truncated stream") from exc

    wanted = sum(max(0, longs[i]) for i in range(_CCP_TEXT, _CCP_HDR_TXBX + 1))
    chunks: list[str] = []
    total = 0
    for k, fc_raw in enumerate(fcs):
        count = cps[k + 1] - cps[k]
        if count <= 0:
            continue
        count = min(count, MAX_DOC_CHARS - total)
        if fc_raw & FC_COMPRESSED:
            offset = (fc_raw & ~FC_COMPRESSED) // 2
            chunk = word_document[offset: offset + count].decode("cp1252", errors="replace")
        else:
            chunk = word_document[fc_raw: fc_raw + 2 * count].decode("utf-16-le",
                                                                     errors="replace")
        chunks.append(chunk)
        total += len(chunk)
        if total >= min(wanted, MAX_DOC_CHARS):
            break
    full = "".join(chunks)

    reasons: list[str | None] = [None] * len(full)
    try:
        spans = _hidden_cp_spans(_formatting_runs(word_document, table, rg_fclcb), cps, fcs)
    except (struct.error, DocFormatError):
        spans = []  # unreadable formatting table: the text is still worth having
    for cp_low, cp_high, reason in spans:
        for cp in range(max(cp_low, 0), min(cp_high, len(full))):
            reasons[cp] = reason

    collector = HiddenTextCollector()
    stories: list[str] = []
    start = 0
    for index in range(_CCP_TEXT, _CCP_HDR_TXBX + 1):
        length = max(0, longs[index])
        if index in _INCLUDED_STORIES and length:
            stories.append(_split_hidden(full[start: start + length],
                                         reasons[start: start + length], collector))
        start += length
    return _clean("\r".join(stories)), collector


def text_from_streams(word_document: bytes, table: bytes) -> str:
    """Document text (hidden runs removed) from the WordDocument and table streams
    (raises DocFormatError)."""
    return _doc_text(word_document, table)[0]


def _clean(text: str) -> str:
    """Drop field instructions (keep results) and map Word control characters."""
    out: list[str] = []
    in_instruction: list[bool] = []  # one entry per open field: still in its instruction?
    open_instructions = 0  # a nested field's result inside an instruction is still hidden
    for ch in text:
        if ch == "\x13":
            in_instruction.append(True)
            open_instructions += 1
        elif ch == "\x14" and in_instruction:
            if in_instruction[-1]:
                in_instruction[-1] = False
                open_instructions -= 1
        elif ch == "\x15" and in_instruction:
            if in_instruction.pop():
                open_instructions -= 1
        elif not open_instructions:
            out.append(ch)
    cleaned = "".join(out).translate(_CONTROL_MAP)
    return "".join(c for c in cleaned if c in "\n\t" or ord(c) >= 0x20)


def _read_streams(data: bytes) -> tuple[bytes, bytes]:
    with olefile.OleFileIO(io.BytesIO(data)) as ole:
        if not ole.exists("WordDocument"):
            raise DocFormatError("not a Word binary document")
        word_document = ole.openstream("WordDocument").read()
        if len(word_document) < 0x0C:
            raise DocFormatError("FIB too short")
        if _u16(word_document, 0x0A) & FLAG_ENCRYPTED:
            raise DocEncryptedError("encrypted .doc")
        table_name = "1Table" if _u16(word_document, 0x0A) & FLAG_TABLE1 else "0Table"
        if not ole.exists(table_name):
            raise DocFormatError(f"missing {table_name} stream")
        table = ole.openstream(table_name).read()
    return word_document, table


def extract_doc(data: bytes) -> ExtractionResult:
    try:
        text, hidden = _doc_text(*_read_streams(data))
    except DocEncryptedError:
        return unreadable("doc", "Word document is password-protected")
    except Exception as exc:  # olefile and malformed-structure errors of every kind
        logger.info(".doc could not be read (%s)", type(exc).__name__)
        return unreadable("doc", f"could not read Word 97-2003 file ({type(exc).__name__})")
    flags = [Flag(code=FlagCode.HIDDEN_TEXT_CHECK_UNAVAILABLE, severity="info",
                  detail=CHECK_SCOPE_NOTE)]
    hidden_flag = hidden.flag(DOC_REASON_LABELS)
    if hidden_flag is not None:
        flags.append(hidden_flag)
    return ExtractionResult(
        kind="doc", visible_text=text,
        hidden_snippets=hidden.snippets,
        hidden_char_count=hidden.char_count,
        flags=flags,
    )
