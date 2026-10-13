"""Plain-text resumes and ODT, plus small helpers shared by every extractor.

Decoding order (verified): BOM -> UTF-8 strict -> charset-normalizer's best guess,
accepted unless it is a Latin-1-family guess for mostly-ASCII input -> cp1252 (errors
"replace"). Trying cp1252 strict before charset-normalizer would turn cp1251 etc. into
mojibake.
ODT: read ``content.xml`` (and ``styles.xml`` when present) from the zip with lxml
``XMLParser(resolve_entities=False, no_network=True)``, each capped at
``ODT_MAX_PART_BYTES``, and collect text:p / text:h in order.

ODT hidden text: ``office:automatic-styles`` and ``office:styles`` are resolved into text
properties per style (following ``style:parent-style-name``, at most ``_MAX_STYLE_DEPTH``
levels), and every text fragment is judged by the effective properties of its paragraph
and enclosing ``text:span`` elements (``text:style-name`` and ``text:class-names``):
``text:display="none"`` -> "vanish"; ``fo:font-size`` under 3pt (percentages resolve
against the parent) -> "tiny_font"; ``style:text-scale`` under 20% -> "squashed";
``fo:color`` with luminance over 0.92 and no dark background -> "white_text"; a colour
within 0.10 luminance of the innermost known background (span, paragraph or table-cell
``fo:background-color``) -> "low_contrast". The bodies of ``text:hidden-text`` and of a
paragraph holding a ``text:hidden-paragraph`` field with ``text:is-hidden="true"`` are
hidden too, and ``text:string-value`` attributes are never read. Hidden fragments go to
``hidden_snippets``/``hidden_char_count`` and the ``hidden_text`` flag ("review" at >= 20
characters), exactly as for PDF and DOCX.
"""

from __future__ import annotations

import codecs
import io
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, replace

from charset_normalizer import from_bytes
from lxml import etree

from applicant_ranking.domain import ExtractionResult, FileKind, Flag, FlagCode

MAX_SNIPPETS = 5
MAX_SNIPPET_CHARS = 200
HIDDEN_REVIEW_THRESHOLD = 20  # hidden characters at which the flag forces human review

# Single-byte Western/Central European code pages. charset-normalizer confuses these with
# each other on short, mostly-ASCII text (cp1252 "naïve" read as cp1250 "naďve").
_LATIN_FAMILY = frozenset({
    "ascii", "latin_1", "iso8859_2", "iso8859_3", "iso8859_4", "iso8859_9", "iso8859_10",
    "iso8859_13", "iso8859_14", "iso8859_15", "iso8859_16",
    "cp1250", "cp1252", "cp1254", "cp1257", "cp1258",
    "cp437", "cp775", "cp850", "cp852", "cp857", "cp858", "cp860", "cp861", "cp863", "cp865",
    "cp037", "cp273", "cp500", "cp1026", "cp1140",
    "mac_roman", "mac_latin2", "mac_iceland", "mac_turkish", "hp_roman8",
})
# Every Latin-script language stays above this share of ASCII bytes, so a Latin-family
# guess on such input is no better than cp1252 (the most common legacy encoding).
_MOSTLY_ASCII = 0.75
_HIGH_BYTES = bytes(range(0x80, 0x100))

# Byte-order marks, longest first (the UTF-32 LE mark starts with the UTF-16 LE one).
_BOMS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

_ODF_TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_ODF_OFFICE_NS = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
_ODF_STYLE_NS = "urn:oasis:names:tc:opendocument:xmlns:style:1.0"
_ODF_FO_NS = "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
_ODF_TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
ODT_MAX_PART_BYTES = 8 * 1024 * 1024  # same bound as the sniffer's XML cap

ODT_MIN_FONT_PT = 3.0
ODT_MIN_TEXT_SCALE = 20.0  # percent
ODT_WHITE_LUMINANCE = 0.92
ODT_DARK_BACKGROUND = 0.6
ODT_LOW_CONTRAST = 0.10
_DEFAULT_FONT_PT = 12.0
_MAX_STYLE_DEPTH = 20
ODT_REASON_LABELS = {
    "vanish": "hidden (display: none) formatting",
    "tiny_font": "font under 3pt",
    "squashed": "character width under 20%",
    "white_text": "white text on a light background",
    "low_contrast": "text colour matching its background",
    "hidden_field": "hidden text or hidden paragraph field",
}


# ----------------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------------


def clean_unicode(text: str) -> str:
    """Join UTF-16 surrogate pairs and replace lone surrogates with U+FFFD. Some sources
    (RTF ``\\u`` escapes for emoji, for one) yield surrogates, which can't be encoded to
    UTF-8 and would break hashing and JSON serialisation downstream."""
    if not _SURROGATE_RE.search(text):
        return text
    return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")


def unreadable(kind: FileKind, error: str) -> ExtractionResult:
    """Result for a file that could not be read at all (forces human review)."""
    return ExtractionResult(
        kind=kind, error=error,
        flags=[Flag(code=FlagCode.UNREADABLE_FILE, severity="review", detail=error)],
    )


def hidden_text_flag(char_count: int, *, detail: str, force_review: bool = False) -> Flag | None:
    """``hidden_text`` flag: "review" at >= HIDDEN_REVIEW_THRESHOLD characters (or when
    forced), "info" below it, nothing when no hidden text was found."""
    if char_count <= 0 and not force_review:
        return None
    review = force_review or char_count >= HIDDEN_REVIEW_THRESHOLD
    return Flag(code=FlagCode.HIDDEN_TEXT, severity="review" if review else "info",
                detail=detail)


def hex_luminance(value: str | None) -> float | None:
    """Relative luminance (0-1) of a ``#RRGGBB`` / ``RRGGBB`` colour, else None."""
    if not value:
        return None
    value = value.strip().lstrip("#")
    if len(value) != 6:
        return None
    try:
        r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255


class HiddenTextCollector:
    """Accumulates hidden fragments into runs: consecutive hidden fragments form one run
    until ``end_run`` (called when visible text or a block boundary intervenes). Keeps at
    most MAX_SNIPPETS short samples, but counts every hidden non-whitespace character."""

    def __init__(self) -> None:
        self.char_count = 0
        self.reasons: Counter[str] = Counter()
        self._snippets: list[str] = []
        self._current: list[str] = []

    def add(self, text: str, reason: str) -> None:
        chars = sum(1 for ch in text if not ch.isspace())
        if not chars:
            return
        self.char_count += chars
        self.reasons[reason] += chars
        if len(self._snippets) < MAX_SNIPPETS:
            self._current.append(text)

    def end_run(self) -> None:
        if self._current and len(self._snippets) < MAX_SNIPPETS:
            snippet = " ".join(" ".join(self._current).split())
            if len(snippet) > MAX_SNIPPET_CHARS:
                snippet = snippet[: MAX_SNIPPET_CHARS - 3].rstrip() + "..."
            if snippet:
                self._snippets.append(snippet)
        self._current = []

    @property
    def snippets(self) -> list[str]:
        self.end_run()
        return list(self._snippets)

    def describe(self, labels: dict[str, str]) -> str:
        parts = [f"{labels.get(reason, reason)}: {count}"
                 for reason, count in self.reasons.most_common()]
        return f"{self.char_count} hidden characters removed ({'; '.join(parts)})"

    def flag(self, labels: dict[str, str]) -> Flag | None:
        return hidden_text_flag(self.char_count, detail=self.describe(labels))


# ----------------------------------------------------------------------------------------
# Plain text
# ----------------------------------------------------------------------------------------


def _strip_bom(data: bytes) -> tuple[str, bytes] | None:
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            return encoding, data[len(bom):]
    return None


def _codec_name(encoding: str) -> str:
    try:
        return codecs.lookup(encoding).name.replace("-", "_")
    except LookupError:
        return encoding.lower().replace("-", "_")


def decode_text(data: bytes) -> str:
    bom = _strip_bom(data)
    if bom is not None:
        encoding, body = bom
        return body.decode(encoding, errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    best = from_bytes(data).best()
    if best is not None and best.encoding:
        encoding = _codec_name(best.encoding)
        ascii_share = len(data.translate(None, _HIGH_BYTES)) / max(len(data), 1)
        if encoding in _LATIN_FAMILY and ascii_share >= _MOSTLY_ASCII:
            try:
                return data.decode("cp1252")
            except UnicodeDecodeError:
                pass  # bytes cp1252 leaves undefined: trust the detector after all
        try:
            return data.decode(best.encoding, errors="replace")
        except LookupError:
            pass
    return data.decode("cp1252", errors="replace")


def extract_txt(data: bytes) -> ExtractionResult:
    return ExtractionResult(kind="txt", visible_text=decode_text(data), pages=None)


# ----------------------------------------------------------------------------------------
# ODT
# ----------------------------------------------------------------------------------------


def _text(tag: str) -> str:
    return f"{{{_ODF_TEXT_NS}}}{tag}"


def _style(tag: str) -> str:
    return f"{{{_ODF_STYLE_NS}}}{tag}"


def _fo(tag: str) -> str:
    return f"{{{_ODF_FO_NS}}}{tag}"


_TEXT_P = _text("p")
_TEXT_H = _text("h")
_TEXT_S = _text("s")
_TEXT_TAB = _text("tab")
_TEXT_BREAK = _text("line-break")
_TEXT_SPAN = _text("span")
_TEXT_HIDDEN_TEXT = _text("hidden-text")
_TEXT_HIDDEN_PARAGRAPH = _text("hidden-paragraph")
_TABLE_CELL = f"{{{_ODF_TABLE_NS}}}table-cell"
_STYLE_CONTAINERS = (f"{{{_ODF_OFFICE_NS}}}automatic-styles", f"{{{_ODF_OFFICE_NS}}}styles")
# Not displayed in the document body: comments and tracked-change bookkeeping.
_ODT_SKIP = frozenset({
    f"{{{_ODF_OFFICE_NS}}}annotation",
    f"{{{_ODF_OFFICE_NS}}}annotation-end",
    _text("tracked-changes"),
})


@dataclass(frozen=True)
class _OdtTextProps:
    """Resolved text properties of a style (None = not set)."""

    display: str | None = None
    font_size_pt: float | None = None
    text_scale: float | None = None  # percent
    color: str | None = None
    background: str | None = None  # "#RRGGBB" or "transparent"

    def over(self, child: _OdtTextProps) -> _OdtTextProps:
        """``child`` overriding ``self`` where it sets a value."""
        return replace(self, **{k: v for k, v in child.__dict__.items() if v is not None})


def _length_pt(value: str | None, parent_pt: float | None) -> float | None:
    if not value:
        return None
    value = value.strip().lower()
    try:
        if value.endswith("%"):
            return (parent_pt or _DEFAULT_FONT_PT) * float(value[:-1]) / 100
        for unit, factor in (("pt", 1.0), ("in", 72.0), ("cm", 72 / 2.54), ("mm", 72 / 25.4),
                             ("px", 0.75), ("pc", 12.0), ("em", parent_pt or _DEFAULT_FONT_PT)):
            if value.endswith(unit):
                return float(value[:-len(unit)]) * factor
        return float(value)
    except ValueError:
        return None


def _percent(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value.strip().rstrip("%"))
    except ValueError:
        return None


class _OdtStyles:
    """Style lookup by (family, name) across content.xml and styles.xml."""

    def __init__(self, roots: list[etree._Element]) -> None:
        self._styles: dict[tuple[str, str], etree._Element] = {}
        self._cache: dict[tuple[str, str], _OdtTextProps] = {}
        for root in roots:
            for container in root.iter(*_STYLE_CONTAINERS):
                for style in container.iterfind(_style("style")):
                    family = style.get(_style("family"))
                    name = style.get(_style("name"))
                    if family and name:
                        self._styles.setdefault((family, name), style)

    def props(self, family: str, name: str | None) -> _OdtTextProps:
        if not name:
            return _OdtTextProps()
        key = (family, name)
        if key in self._cache:
            return self._cache[key]
        chain: list[etree._Element] = []
        seen: set[str] = set()
        current: str | None = name
        while current and current not in seen and len(chain) < _MAX_STYLE_DEPTH:
            style = self._styles.get((family, current))
            if style is None:
                break
            seen.add(current)
            chain.append(style)
            current = style.get(_style("parent-style-name"))
        props = _OdtTextProps()
        for style in reversed(chain):  # parents first, children override
            props = props.over(self._own_props(style, props.font_size_pt))
        self._cache[key] = props
        return props

    @staticmethod
    def _own_props(style: etree._Element, parent_pt: float | None) -> _OdtTextProps:
        text_props = style.find(_style("text-properties"))
        background = None
        for section in ("paragraph-properties", "table-cell-properties", "text-properties"):
            element = style.find(_style(section))
            if element is not None and element.get(_fo("background-color")):
                background = element.get(_fo("background-color")).strip().lower()
        if text_props is None:
            return _OdtTextProps(background=background)
        return _OdtTextProps(
            display=text_props.get(_text("display")),
            font_size_pt=_length_pt(text_props.get(_fo("font-size")), parent_pt),
            text_scale=_percent(text_props.get(_style("text-scale"))),
            color=(text_props.get(_fo("color")) or "").strip().lower() or None,
            background=background,
        )


def _hidden_reason(props: _OdtTextProps, backgrounds: list[float]) -> str | None:
    if props.display == "none":
        return "vanish"
    if props.font_size_pt is not None and props.font_size_pt < ODT_MIN_FONT_PT:
        return "tiny_font"
    if props.text_scale is not None and props.text_scale < ODT_MIN_TEXT_SCALE:
        return "squashed"
    luminance = hex_luminance(props.color)
    if luminance is None:
        return None
    if any(bg < ODT_DARK_BACKGROUND for bg in backgrounds):
        return "low_contrast" if abs(luminance - backgrounds[0]) < ODT_LOW_CONTRAST else None
    if luminance > ODT_WHITE_LUMINANCE:
        return "white_text"
    background = backgrounds[0] if backgrounds else 1.0
    return "low_contrast" if abs(luminance - background) < ODT_LOW_CONTRAST else None


class _OdtWalker:
    def __init__(self, styles: _OdtStyles) -> None:
        self._styles = styles
        self.hidden = HiddenTextCollector()

    def _span_props(self, element: etree._Element, base: _OdtTextProps) -> _OdtTextProps:
        family = "paragraph" if element.tag in (_TEXT_P, _TEXT_H) else "text"
        props = base
        names = [element.get(_text("style-name"))]
        names += (element.get(_text("class-names")) or "").split()
        for name in names:
            if name:
                props = props.over(self._styles.props(family, name))
        return props

    @staticmethod
    def _cell_background(paragraph: etree._Element, styles: _OdtStyles) -> float | None:
        for ancestor in paragraph.iterancestors():
            if ancestor.tag == _TABLE_CELL:
                props = styles.props("table-cell", ancestor.get(f"{{{_ODF_TABLE_NS}}}style-name"))
                return hex_luminance(props.background)
        return None

    def paragraph_text(self, element: etree._Element) -> str:
        """Visible text of one text:p / text:h, excluding nested paragraphs (visited on
        their own); hidden fragments go to the collector."""
        props = self._span_props(element, _OdtTextProps())
        cell_bg = self._cell_background(element, self._styles)
        forced = None
        if any(child.tag == _TEXT_HIDDEN_PARAGRAPH and child.get(_text("is-hidden")) == "true"
               for child in element.iter(_TEXT_HIDDEN_PARAGRAPH)):
            forced = "hidden_field"
        out: list[str] = []

        def emit(fragment: str, current: _OdtTextProps, reason: str | None) -> None:
            if not fragment:
                return
            backgrounds = [lum for lum in (hex_luminance(current.background), cell_bg)
                           if lum is not None]
            reason = reason or _hidden_reason(current, backgrounds)
            if reason is None:
                self.hidden.end_run()
                out.append(fragment)
            else:
                self.hidden.add(fragment, reason)

        def walk(node: etree._Element, current: _OdtTextProps, reason: str | None) -> None:
            for child in node:
                tag = child.tag
                if not isinstance(tag, str) or tag in (_TEXT_P, _TEXT_H) or tag in _ODT_SKIP:
                    pass  # comments, processing instructions, entities, nested/skipped parts
                elif tag == _TEXT_S:
                    try:
                        count = int(child.get(_text("c"), "1"))
                    except ValueError:
                        count = 1
                    emit(" " * max(1, min(count, 100)), current, reason)
                elif tag == _TEXT_TAB:
                    emit("\t", current, reason)
                elif tag == _TEXT_BREAK:
                    emit("\n", current, reason)
                elif tag == _TEXT_HIDDEN_TEXT:
                    hidden = reason or ("hidden_field" if child.get(_text("is-hidden")) != "false"
                                        else None)
                    emit(child.text or "", current, hidden)
                    walk(child, current, hidden)
                else:
                    inner = self._span_props(child, current) if tag == _TEXT_SPAN else current
                    emit(child.text or "", inner, reason)
                    walk(child, inner, reason)
                emit(child.tail or "", current, reason)

        emit(element.text or "", props, forced)
        walk(element, props, forced)
        self.hidden.end_run()
        return "".join(out)


def _inside_skipped(element: etree._Element) -> bool:
    return any(ancestor.tag in _ODT_SKIP for ancestor in element.iterancestors())


def _read_odt_part(archive: zipfile.ZipFile, name: str, parser: etree.XMLParser,
                   required: bool) -> etree._Element | None:
    try:
        info = archive.getinfo(name)
    except KeyError:
        if required:
            raise
        return None
    if info.file_size > ODT_MAX_PART_BYTES:
        raise ValueError(f"{name} is too large")
    return etree.fromstring(archive.read(info), parser)


def extract_odt(data: bytes) -> ExtractionResult:
    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            root = _read_odt_part(archive, "content.xml", parser, required=True)
            styles_root = _read_odt_part(archive, "styles.xml", parser, required=False)
    except (KeyError, zipfile.BadZipFile, etree.XMLSyntaxError, ValueError, OSError,
            RuntimeError, NotImplementedError) as exc:
        return unreadable("odt", f"could not read ODT file ({type(exc).__name__})")
    assert root is not None
    styles = _OdtStyles([r for r in (root, styles_root) if r is not None])
    walker = _OdtWalker(styles)
    lines = [
        walker.paragraph_text(element)
        for element in root.iter(_TEXT_P, _TEXT_H)
        if not _inside_skipped(element)
    ]
    text = "\n".join(line for line in lines if line.strip())
    flag = walker.hidden.flag(ODT_REASON_LABELS)
    return ExtractionResult(
        kind="odt", visible_text=text,
        hidden_snippets=walker.hidden.snippets,
        hidden_char_count=walker.hidden.char_count,
        flags=[flag] if flag is not None else [],
    )
