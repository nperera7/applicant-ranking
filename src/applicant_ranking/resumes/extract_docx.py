"""DOCX text extraction with hidden-run detection.

python-docx's high-level API drops text boxes, tracked insertions (w:ins), and content
controls (w:sdt), so walk its lxml tree (``docx.Document(io.BytesIO(data)).element``)
directly:
- Parts: body, plus every header/footer of every section (default, first-page, even-page).
- Visit ``.//w:r`` under each ``w:p`` in document order; include w:sdt content and text
  boxes (w:txbxContent).
- Tables: a data table (several rows, one line per cell) reads row by row with the cells
  joined by " | "; a layout table (a single row, or a cell holding several paragraphs:
  the two-column resume template) reads cell after cell, one output line per paragraph,
  so a whole sidebar never collapses into one line.
- Skip anything inside ``w:del``, ``w:moveFrom``, ``mc:Fallback`` (Word stores text boxes
  twice) and field instructions (``w:instrText``). Text: w:t, w:tab -> "\\t",
  w:br/w:cr -> "\\n".
- Hidden run if any: ``w:vanish``/``w:specVanish`` on the run, its rStyle, the
  paragraph's pStyle, the table style (base and conditional ``w:tblStylePr`` formatting
  for the cell's position), or w:docDefaults (follow basedOn chains); size ``w:sz`` < 6
  half-points (3pt); character scale ``w:w`` < 20 (%); ``w:position`` beyond
  ``MAX_RUN_OFFSET_HALF_POINTS`` (text moved off its line); colour luminance > 0.92 on a
  background that is known to be light ("white_text"); colour within ``LOW_CONTRAST``
  luminance of the innermost known background, light or dark ("low_contrast"; the page
  is white unless ``w:background`` says otherwise); a text box whose ``wp:extent`` is
  under a point ("collapsed_box") or whose anchor lies beyond a page's distance from the
  page ("off_page").
- Colours: ``w:color/@w:val`` is the resolved RGB that Word always writes, so it wins over
  ``w:themeColor``; the theme is consulted only when w:val is missing (theme1.xml through
  the settings' clrSchemeMapping, then w:themeShade/w:themeTint). "auto" is a colour
  Word picks to contrast with the background and is never hidden. Shading (``w:shd``)
  follows the same rule with w:fill / w:themeFill / w:themeFillShade / w:themeFillTint.
- Backgrounds: highlight, run/paragraph/cell/table shading (direct, paragraph style or
  table style with conditional formatting), the fill of an enclosing text-box shape
  (a:solidFill with srgbClr or schemeClr plus lumMod/lumOff/shade/tint modifiers,
  a:gradFill judged by its darkest stop, a:pattFill by its foreground, or the shape
  style's a:fillRef), VML fill colours and the page background. White text is removed
  only when every background behind it is positively known to be light: an
  unresolvable fill (a picture fill, a placeholder colour) or a dark decorative shape
  anchored behind the text in the same part (``behindDoc``/``wrapNone``, at least
  ``MIN_DECORATIVE_SHAPE_EMU2`` in area) means the text is kept and only counted in an
  info-level note, never promoted to review.
Hidden runs go to ``hidden_snippets``/``hidden_char_count`` (flag ``hidden_text``,
"review", when >= 20 chars), not to ``visible_text``. Parse with python-docx (its parser
has resolve_entities=False). Corrupt file -> ``error`` + ``unreadable_file`` flag.

Also handled: footnotes and endnotes (visible content), and a first-page or even-page
header/footer that Word never displays (no ``w:titlePg`` / ``w:evenAndOddHeaders``) is
reported as hidden rather than visible text.
"""

from __future__ import annotations

import colorsys
import io
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import docx
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml.parser import parse_xml
from lxml import etree

from applicant_ranking.domain import ExtractionResult, Flag, FlagCode
from applicant_ranking.resumes.extract_text import HiddenTextCollector, unreadable

if TYPE_CHECKING:
    from docx.document import Document as DocxDocument

logger = logging.getLogger(__name__)

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
WPS_NS = "http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
V_NS = "urn:schemas-microsoft-com:vml"
NS = {"w": W_NS, "mc": MC_NS, "a": A_NS, "wp": WP_NS, "wps": WPS_NS, "v": V_NS}

MIN_HALF_POINTS = 6  # w:sz is in half-points: < 3pt
MIN_CHAR_SCALE = 20  # w:w, percent
MAX_RUN_OFFSET_HALF_POINTS = 144  # w:position beyond +/- 72pt is not a typographic shift
WHITE_LUMINANCE = 0.92
DARK_BACKGROUND = 0.6
LOW_CONTRAST = 0.10  # catches #E6E6E6 on white; #BFBFBF (0.75) stays visible
EMU_PER_TWIP = 635
EMU_PER_POINT = 12700
DEFAULT_PAGE_TWIPS = (12240, 15840)  # US Letter
MIN_BOX_EMU = EMU_PER_POINT  # a text box narrower or shorter than 1pt shows nothing
MIN_DECORATIVE_SHAPE_EMU2 = 2 * 914400 * 914400  # two square inches

REASON_LABELS = {
    "vanish": "hidden (vanish) formatting",
    "tiny_font": "font under 3pt",
    "squashed": "character width under 20%",
    "offset": "text moved off its line",
    "white_text": "white text on a light background",
    "low_contrast": "text colour matching its background",
    "collapsed_box": "text box smaller than a point",
    "off_page": "text box anchored outside the page",
    "undisplayed_header": "header/footer that Word never displays",
}
UNVERIFIED_WHITE = "white_text_unverified"

_HIGHLIGHT_HEX = {
    "black": "000000", "blue": "0000FF", "cyan": "00FFFF", "green": "00FF00",
    "magenta": "FF00FF", "red": "FF0000", "yellow": "FFFF00", "white": "FFFFFF",
    "darkBlue": "000080", "darkCyan": "008080", "darkGreen": "008000",
    "darkMagenta": "800080", "darkRed": "800000", "darkYellow": "808000",
    "darkGray": "808080", "lightGray": "C0C0C0",
}
_THEME_LUMINANCE = {"background1": 1.0, "light1": 1.0, "lt1": 1.0, "bg1": 1.0,
                    "text1": 0.0, "dark1": 0.0, "dk1": 0.0, "tx1": 0.0}
# WordprocessingML theme colour names -> DrawingML scheme slots (overridden by the
# settings' w:clrSchemeMapping).
_DEFAULT_SCHEME_MAPPING = {
    "background1": "lt1", "text1": "dk1", "background2": "lt2", "text2": "dk2",
    "bg1": "lt1", "tx1": "dk1", "bg2": "lt2", "tx2": "dk2",
    "light1": "lt1", "dark1": "dk1", "light2": "lt2", "dark2": "dk2",
    "hyperlink": "hlink", "followedHyperlink": "folHlink",
}
_PRESET_COLORS = {"black": (0, 0, 0), "white": (255, 255, 255), "navy": (0, 0, 128),
                  "gray": (128, 128, 128), "grey": (128, 128, 128), "silver": (192, 192, 192),
                  "red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 128, 0),
                  "yellow": (255, 255, 0), "maroon": (128, 0, 0), "purple": (128, 0, 128),
                  "teal": (0, 128, 128), "olive": (128, 128, 0)}
_NOTE_SKIP_TYPES = frozenset({"separator", "continuationSeparator", "continuationNotice"})
_MAX_STYLE_DEPTH = 20
_VML_UNITS_PT = {"pt": 1.0, "in": 72.0, "cm": 72.0 / 2.54, "mm": 72.0 / 25.4, "px": 0.75,
                 "pc": 12.0, "": 1.0}

Rgb = tuple[int, int, int]


def _w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def _a(tag: str) -> str:
    return f"{{{A_NS}}}{tag}"


_SKIP_ANCESTORS = frozenset({_w("del"), _w("moveFrom"), f"{{{MC_NS}}}Fallback"})
_CONTAINERS = frozenset({_w("customXml"), _w("ins"), _w("moveTo"), _w("smartTag")})
_RUN_TEXT = {
    _w("tab"): "\t", _w("ptab"): "\t", _w("br"): "\n", _w("cr"): "\n",
    _w("noBreakHyphen"): "-",
}
_WP_ANCHOR, _WP_INLINE = f"{{{WP_NS}}}anchor", f"{{{WP_NS}}}inline"
_VML_SHAPES = frozenset({f"{{{V_NS}}}shape", f"{{{V_NS}}}rect", f"{{{V_NS}}}roundrect",
                         f"{{{V_NS}}}oval"})


def _on(element: etree._Element | None) -> bool:
    """OOXML toggle property: present and not val=false/0/off."""
    if element is None:
        return False
    return element.get(_w("val"), "true").lower() not in ("false", "0", "off")


def _int_val(element: etree._Element | None, attr: str = "val") -> int | None:
    if element is None:
        return None
    raw = (element.get(_w(attr)) or "").strip().rstrip("%")
    try:
        return int(float(raw))
    except ValueError:
        return None


def _int_attr(element: etree._Element | None, attr: str) -> int | None:
    if element is None:
        return None
    try:
        return int(element.get(attr, ""))
    except ValueError:
        return None


# ---- colour arithmetic ----------------------------------------------------------------------


def _parse_hex(value: str | None) -> Rgb | None:
    if not value:
        return None
    value = value.strip().lstrip("#")
    if len(value) != 6:
        return None
    try:
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return None


def _luminance(rgb: Rgb) -> float:
    r, g, b = rgb
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255


def _hex_luminance(value: str | None) -> float | None:
    rgb = _parse_hex(value)
    return _luminance(rgb) if rgb is not None else None


def _clamp(channel: float) -> int:
    return max(0, min(255, round(channel)))


def _shade(rgb: Rgb, factor: float) -> Rgb:
    """Toward black: factor 1 keeps the colour, 0 is black."""
    return tuple(_clamp(c * factor) for c in rgb)  # type: ignore[return-value]


def _tint(rgb: Rgb, factor: float) -> Rgb:
    """Toward white: factor 1 keeps the colour, 0 is white."""
    return tuple(_clamp(255 - (255 - c) * factor) for c in rgb)  # type: ignore[return-value]


def _hex_byte_factor(value: str | None) -> float | None:
    """w:themeShade / w:themeTint: a hex byte, 00-FF -> 0.0-1.0."""
    if not value:
        return None
    try:
        return int(value, 16) / 255
    except ValueError:
        return None


def _theme_adjust(rgb: Rgb, shade: str | None, tint: str | None) -> Rgb:
    factor = _hex_byte_factor(shade)
    if factor is not None:
        rgb = _shade(rgb, factor)
    factor = _hex_byte_factor(tint)
    if factor is not None:
        rgb = _tint(rgb, factor)
    return rgb


def _drawingml_modifiers(rgb: Rgb, color_element: etree._Element) -> Rgb:
    """Apply a:lumMod, a:lumOff (HSL lightness), a:shade and a:tint children."""
    for mod in color_element:
        tag = etree.QName(mod).localname if isinstance(mod.tag, str) else ""
        value = _int_attr(mod, "val")
        if value is None:
            continue
        share = value / 100_000
        if tag in ("lumMod", "lumOff"):
            h, lightness, s = colorsys.rgb_to_hls(*(c / 255 for c in rgb))
            lightness = lightness * share if tag == "lumMod" else lightness + share
            rgb = tuple(_clamp(c * 255) for c in  # type: ignore[assignment]
                        colorsys.hls_to_rgb(h, max(0.0, min(1.0, lightness)), s))
        elif tag == "shade":
            rgb = _shade(rgb, share)
        elif tag == "tint":
            rgb = _tint(rgb, share)
    return rgb


@dataclass(frozen=True)
class _Backdrop:
    """Background luminances behind a run, innermost first, and whether one of the
    backgrounds could not be resolved (so the text's visibility is unknown)."""

    luminances: tuple[float, ...]
    unknown: bool

    @property
    def dark(self) -> bool:
        return any(lum < DARK_BACKGROUND for lum in self.luminances)


class _Theme:
    """The document theme's colour scheme, through the settings' clrSchemeMapping."""

    def __init__(self, document: DocxDocument) -> None:
        self._colors: dict[str, Rgb] = {}
        self._mapping = dict(_DEFAULT_SCHEME_MAPPING)
        for rel in document.part.rels.values():
            if rel.is_external or rel.reltype != RT.THEME:
                continue
            try:
                root = parse_xml(rel.target_part.blob)
            except Exception as exc:  # a malformed theme part only disables the lookup
                logger.info("DOCX theme part could not be read (%s)", type(exc).__name__)
                continue
            scheme = root.find("a:themeElements/a:clrScheme", NS)
            if scheme is None:
                continue
            for slot in scheme:
                if not isinstance(slot.tag, str):
                    continue
                rgb = _scheme_slot_rgb(slot)
                if rgb is not None:
                    self._colors[etree.QName(slot).localname] = rgb
            break
        mapping = document.settings.element.find("w:clrSchemeMapping", NS)
        if mapping is not None:
            for key, value in mapping.attrib.items():
                name = etree.QName(key).localname
                self._mapping[name] = _DEFAULT_SCHEME_MAPPING.get(value, value)

    def rgb(self, name: str | None) -> Rgb | None:
        if not name:
            return None
        return self._colors.get(self._mapping.get(name, name))


def _scheme_slot_rgb(slot: etree._Element) -> Rgb | None:
    srgb = slot.find("a:srgbClr", NS)
    if srgb is not None:
        return _parse_hex(srgb.get("val"))
    sys_color = slot.find("a:sysClr", NS)
    if sys_color is not None:
        return _parse_hex(sys_color.get("lastClr"))
    return None


@dataclass(frozen=True)
class _TableLook:
    first_row: bool = True
    last_row: bool = True
    first_col: bool = True
    last_col: bool = True
    no_h_band: bool = False
    no_v_band: bool = False


@dataclass(frozen=True)
class _CellContext:
    """Where a cell sits in its table, for conditional table-style formatting."""

    style_id: str | None
    look: _TableLook
    row: int
    rows: int
    col: int
    cols: int

    def conditional_types(self) -> list[str]:
        """Applicable w:tblStylePr types, lowest precedence first."""
        first_row = self.look.first_row and self.row == 0
        last_row = self.look.last_row and self.rows > 1 and self.row == self.rows - 1
        first_col = self.look.first_col and self.col == 0
        last_col = self.look.last_col and self.cols > 1 and self.col == self.cols - 1
        types = ["wholeTable"]
        if not self.look.no_v_band and not (first_col or last_col):
            body_col = self.col - (1 if self.look.first_col else 0)
            types.append("band1Vert" if body_col % 2 == 0 else "band2Vert")
        if not self.look.no_h_band and not (first_row or last_row):
            body_row = self.row - (1 if self.look.first_row else 0)
            types.append("band1Horz" if body_row % 2 == 0 else "band2Horz")
        if last_col:
            types.append("lastCol")
        if first_col:
            types.append("firstCol")
        if last_row:
            types.append("lastRow")
        if first_row:
            types.append("firstRow")
        if first_row and first_col:
            types.append("nwCell")
        if first_row and last_col:
            types.append("neCell")
        if last_row and first_col:
            types.append("swCell")
        if last_row and last_col:
            types.append("seCell")
        return types


def _table_look(tblpr: etree._Element | None) -> _TableLook:
    look = tblpr.find("w:tblLook", NS) if tblpr is not None else None
    if look is None:
        return _TableLook()
    if look.get(_w("firstRow")) is not None:
        def flag(name: str) -> bool:
            return look.get(_w(name), "0").lower() in ("1", "true", "on")

        return _TableLook(flag("firstRow"), flag("lastRow"), flag("firstColumn"),
                          flag("lastColumn"), flag("noHBand"), flag("noVBand"))
    try:
        bits = int(look.get(_w("val"), "0"), 16)
    except ValueError:
        return _TableLook()
    return _TableLook(bool(bits & 0x0020), bool(bits & 0x0040), bool(bits & 0x0080),
                      bool(bits & 0x0100), bool(bits & 0x0200), bool(bits & 0x0400))


def _run_text(run: etree._Element) -> str:
    out: list[str] = []
    for child in run:
        if child.tag == _w("t"):
            out.append(child.text or "")
        else:
            out.append(_RUN_TEXT.get(child.tag, ""))
    return "".join(out)


def _vml_style(element: etree._Element) -> dict[str, float]:
    """Numeric CSS-like properties of a VML ``style`` attribute, in points."""
    out: dict[str, float] = {}
    for part in (element.get("style") or "").split(";"):
        name, _, raw = part.partition(":")
        raw = raw.strip().lower()
        unit = raw.lstrip("-0123456789.")
        number = raw[:len(raw) - len(unit)]
        if unit in _VML_UNITS_PT and number:
            try:
                out[name.strip().lower()] = float(number) * _VML_UNITS_PT[unit]
            except ValueError:
                continue
    return out


class _DocxWalker:
    def __init__(self, document: DocxDocument) -> None:
        self._document = document
        styles_root = document.styles.element
        self._styles: dict[str, etree._Element] = {}
        self._default_pstyle: str | None = None
        for style in styles_root.iterfind("w:style", NS):
            style_id = style.get(_w("styleId"))
            if style_id:
                self._styles[style_id] = style
            if (style.get(_w("type")) == "paragraph" and self._default_pstyle is None
                    and style.get(_w("default")) in ("1", "true", "on")):
                self._default_pstyle = style_id
        self._defaults_rpr = styles_root.find("w:docDefaults/w:rPrDefault/w:rPr", NS)
        background = document.element.find("w:background", NS)
        self._page_background = (
            _hex_luminance(background.get(_w("color"))) if background is not None else None
        )
        self._theme = _Theme(document)
        self._page_emu = self._page_size_emu()
        self._style_cache: dict[tuple[str, str | None, str], etree._Element | None] = {}
        self._forced_reason: str | None = None
        self._cells: list[_CellContext] = []
        self._dark_decoration = False
        self.hidden = HiddenTextCollector()
        self.unverified_white_chars = 0

    def _page_size_emu(self) -> tuple[int, int]:
        pgsz = self._document.element.body.find("w:sectPr/w:pgSz", NS)
        width = _int_val(pgsz, "w") if pgsz is not None else None
        height = _int_val(pgsz, "h") if pgsz is not None else None
        return ((width or DEFAULT_PAGE_TWIPS[0]) * EMU_PER_TWIP,
                (height or DEFAULT_PAGE_TWIPS[1]) * EMU_PER_TWIP)

    # ---- property resolution ------------------------------------------------------------

    def _style_prop(self, style_id: str | None, path: str,
                    conditional: str | None = None) -> etree._Element | None:
        """``path`` in the style or (for ``conditional``) its w:tblStylePr of that type,
        following basedOn chains."""
        if not style_id:
            return None
        key = (style_id, conditional, path)
        if key in self._style_cache:
            return self._style_cache[key]
        found = None
        seen: set[str] = set()
        current: str | None = style_id
        while current and current not in seen and len(seen) < _MAX_STYLE_DEPTH:
            seen.add(current)
            style = self._styles.get(current)
            if style is None:
                break
            scope: etree._Element | None = style
            if conditional is not None:
                scope = next((c for c in style.iterfind("w:tblStylePr", NS)
                              if c.get(_w("type")) == conditional), None)
            found = scope.find(path, NS) if scope is not None else None
            if found is not None:
                break
            based_on = style.find("w:basedOn", NS)
            current = based_on.get(_w("val")) if based_on is not None else None
        self._style_cache[key] = found
        return found

    def _table_style_prop(self, cell: _CellContext, path: str) -> etree._Element | None:
        """``path`` from the cell's table style: the highest-precedence conditional
        formatting that applies to the cell, else the style's own."""
        found = None
        for cond in cell.conditional_types():
            prop = (self._style_prop(cell.style_id, path) if cond == "wholeTable"
                    else self._style_prop(cell.style_id, path, cond))
            if prop is not None:
                found = prop
        return found

    def _pstyle_id(self, paragraph: etree._Element) -> str | None:
        pstyle = paragraph.find("w:pPr/w:pStyle", NS)
        if pstyle is not None:
            return pstyle.get(_w("val"))
        return self._default_pstyle

    def _run_prop(self, run: etree._Element, paragraph: etree._Element, tag: str
                  ) -> etree._Element | None:
        rpr = run.find("w:rPr", NS)
        if rpr is not None:
            direct = rpr.find(tag, NS)
            if direct is not None:
                return direct
            rstyle = rpr.find("w:rStyle", NS)
            if rstyle is not None:
                styled = self._style_prop(rstyle.get(_w("val")), f"w:rPr/{tag}")
                if styled is not None:
                    return styled
        styled = self._style_prop(self._pstyle_id(paragraph), f"w:rPr/{tag}")
        if styled is not None:
            return styled
        if self._cells:
            styled = self._table_style_prop(self._cells[-1], f"w:rPr/{tag}")
            if styled is not None:
                return styled
        if self._defaults_rpr is not None:
            return self._defaults_rpr.find(tag, NS)
        return None

    # ---- colours --------------------------------------------------------------------------

    def _run_color_luminance(self, color: etree._Element) -> float | None:
        """w:color -> luminance; None for "auto" or an unresolvable theme colour."""
        value = color.get(_w("val"))
        if value is not None and value.strip().lower() == "auto":
            return None
        rgb = _parse_hex(value)
        if rgb is not None:
            return _luminance(rgb)
        theme = color.get(_w("themeColor"))
        rgb = self._theme.rgb(theme)
        if rgb is not None:
            return _luminance(_theme_adjust(rgb, color.get(_w("themeShade")),
                                            color.get(_w("themeTint"))))
        return _THEME_LUMINANCE.get(theme or "")

    def _shading(self, shd: etree._Element | None) -> tuple[float | None, bool]:
        """(luminance, known) of a w:shd fill; (None, True) when there is no fill."""
        if shd is None:
            return None, True
        if shd.get(_w("val")) == "solid":
            rgb = _parse_hex(shd.get(_w("color")))
        else:
            rgb = _parse_hex(shd.get(_w("fill")))
        if rgb is not None:
            return _luminance(rgb), True
        theme = shd.get(_w("themeFill"))
        if theme:
            rgb = self._theme.rgb(theme)
            if rgb is not None:
                rgb = _theme_adjust(rgb, shd.get(_w("themeFillShade")),
                                    shd.get(_w("themeFillTint")))
                return _luminance(rgb), True
            if theme in _THEME_LUMINANCE:
                return _THEME_LUMINANCE[theme], True
            return None, False
        fill = (shd.get(_w("fill")) or "").strip().lower()
        if fill in ("", "auto", "none"):
            return None, True
        return None, False  # a fill we cannot read

    def _drawingml_color(self, element: etree._Element | None) -> tuple[Rgb | None, bool]:
        """(rgb, known) for an a:srgbClr / a:schemeClr / a:sysClr / a:prstClr element."""
        if element is None or not isinstance(element.tag, str):
            return None, False
        tag = etree.QName(element).localname
        rgb: Rgb | None
        if tag == "srgbClr":
            rgb = _parse_hex(element.get("val"))
        elif tag == "sysClr":
            rgb = _parse_hex(element.get("lastClr"))
        elif tag == "schemeClr":
            rgb = self._theme.rgb(element.get("val"))
        elif tag == "prstClr":
            rgb = _PRESET_COLORS.get(element.get("val", "").lower())
        else:
            rgb = None
        if rgb is None:
            return None, False
        return _drawingml_modifiers(rgb, element), True

    def _fill_luminance(self, fill: etree._Element) -> tuple[float | None, bool]:
        """(luminance, known) of a DrawingML fill element (a:solidFill, a:gradFill, ...)."""
        tag = etree.QName(fill).localname
        if tag == "noFill":
            return None, True
        if tag == "solidFill":
            rgb, known = self._drawingml_color(next(iter(fill), None))
            return (_luminance(rgb) if rgb is not None else None), known
        if tag == "gradFill":
            stops = [self._drawingml_color(next(iter(stop), None))
                     for stop in fill.iterfind("a:gsLst/a:gs", NS)]
            if not stops or any(not known for _, known in stops):
                return None, False
            return min(_luminance(rgb) for rgb, _ in stops if rgb is not None), True
        if tag == "pattFill":
            fg = fill.find("a:fgClr", NS)
            rgb, known = self._drawingml_color(next(iter(fg), None) if fg is not None else None)
            return (_luminance(rgb) if rgb is not None else None), known
        return None, False  # blipFill (a picture) or something newer

    def _shape_fill(self, wsp: etree._Element) -> tuple[float | None, bool]:
        """(luminance, known) of a wps:wsp shape's fill: spPr first, else the style's
        fillRef; (None, True) when the shape is transparent."""
        sppr = wsp.find("wps:spPr", NS)
        if sppr is not None:
            for child in sppr:
                if isinstance(child.tag, str) and etree.QName(child).localname.endswith("Fill"):
                    return self._fill_luminance(child)
        fill_ref = wsp.find("wps:style/a:fillRef", NS)
        if fill_ref is None:
            return None, True
        if (_int_attr(fill_ref, "idx") or 0) == 0:
            return None, True
        rgb, known = self._drawingml_color(next(iter(fill_ref), None))
        return (_luminance(rgb) if rgb is not None else None), known

    def _cell_context_for(self, depth: int) -> _CellContext | None:
        """The context of the ``depth``-th enclosing cell (0 = innermost)."""
        index = len(self._cells) - 1 - depth
        return self._cells[index] if index >= 0 else None

    def _backgrounds(self, run: etree._Element, paragraph: etree._Element) -> _Backdrop:
        """Known background luminances behind the run, innermost first."""
        found: list[float | None] = []
        unknown = False

        def add(result: tuple[float | None, bool]) -> None:
            nonlocal unknown
            lum, known = result
            if not known:
                unknown = True
            found.append(lum)

        highlight = self._run_prop(run, paragraph, "w:highlight")
        if highlight is not None:
            found.append(_hex_luminance(_HIGHLIGHT_HEX.get(highlight.get(_w("val"), ""))))
        add(self._shading(self._run_prop(run, paragraph, "w:shd")))
        para_shd = paragraph.find("w:pPr/w:shd", NS)
        if para_shd is None:
            para_shd = self._style_prop(self._pstyle_id(paragraph), "w:pPr/w:shd")
        add(self._shading(para_shd))
        cell_depth = 0
        for ancestor in paragraph.iterancestors():
            tag = ancestor.tag
            if tag == _w("tc"):
                shd = ancestor.find("w:tcPr/w:shd", NS)
                context = self._cell_context_for(cell_depth)
                cell_depth += 1
                if shd is None and context is not None:
                    shd = self._table_style_prop(context, "w:tcPr/w:shd")
                add(self._shading(shd))
            elif tag == _w("tbl"):
                shd = ancestor.find("w:tblPr/w:shd", NS)
                context = self._cell_context_for(cell_depth - 1)
                if shd is None and context is not None:
                    shd = self._style_prop(context.style_id, "w:tblPr/w:shd")
                add(self._shading(shd))
            elif tag == f"{{{WPS_NS}}}wsp":
                add(self._shape_fill(ancestor))
            elif tag.startswith(f"{{{V_NS}}}") and ancestor.get("filled") not in ("f", "false"):
                color = ancestor.get("fillcolor")
                if color:
                    lum = _hex_luminance(color)
                    if lum is None:
                        rgb = _PRESET_COLORS.get(color.strip().lower())
                        lum = _luminance(rgb) if rgb is not None else None
                    add((lum, lum is not None))
        found.append(self._page_background)
        return _Backdrop(tuple(lum for lum in found if lum is not None), unknown)

    def _hidden_reason(self, run: etree._Element, paragraph: etree._Element) -> str | None:
        if self._forced_reason is not None:
            return self._forced_reason
        if _on(self._run_prop(run, paragraph, "w:vanish")) or _on(
                self._run_prop(run, paragraph, "w:specVanish")):
            return "vanish"
        size = _int_val(self._run_prop(run, paragraph, "w:sz"))
        if size is not None and size < MIN_HALF_POINTS:
            return "tiny_font"
        scale = _int_val(self._run_prop(run, paragraph, "w:w"))
        if scale is not None and scale < MIN_CHAR_SCALE:
            return "squashed"
        position = _int_val(self._run_prop(run, paragraph, "w:position"))
        if position is not None and abs(position) > MAX_RUN_OFFSET_HALF_POINTS:
            return "offset"
        color = self._run_prop(run, paragraph, "w:color")
        if color is None:
            return None
        luminance = self._run_color_luminance(color)
        if luminance is None:
            return None  # "auto": Word picks a colour that contrasts with the background
        backdrop = self._backgrounds(run, paragraph)
        if backdrop.dark:
            if abs(luminance - backdrop.luminances[0]) < LOW_CONTRAST:
                return "low_contrast"
            return None
        if backdrop.unknown or self._dark_decoration:
            # Something we cannot see through may be behind the text: keep it.
            return UNVERIFIED_WHITE if luminance > WHITE_LUMINANCE else None
        if luminance > WHITE_LUMINANCE:
            return "white_text"
        background = backdrop.luminances[0] if backdrop.luminances else 1.0
        if abs(luminance - background) < LOW_CONTRAST:
            return "low_contrast"
        return None

    # ---- text boxes and decorative shapes ---------------------------------------------------

    def _box_reason(self, box: etree._Element) -> str | None:
        """"collapsed_box" / "off_page" for a text box too small or too far away to show."""
        page_w, page_h = self._page_emu
        for ancestor in box.iterancestors():
            tag = ancestor.tag
            if tag == _w("r"):
                return None
            if tag in (_WP_ANCHOR, _WP_INLINE):
                extent = ancestor.find("wp:extent", NS)
                cx, cy = _int_attr(extent, "cx"), _int_attr(extent, "cy")
                if cx is not None and cy is not None and (cx < MIN_BOX_EMU or cy < MIN_BOX_EMU):
                    return "collapsed_box"
                if tag == _WP_ANCHOR:
                    x = _offset(ancestor, "wp:positionH")
                    y = _offset(ancestor, "wp:positionV")
                    if (x + (cx or 0) < -page_w or x > 2 * page_w
                            or y + (cy or 0) < -page_h or y > 2 * page_h):
                        return "off_page"
                return None
            if tag in _VML_SHAPES:
                style = _vml_style(ancestor)
                width, height = style.get("width"), style.get("height")
                if width is not None and height is not None and (width < 1 or height < 1):
                    return "collapsed_box"
                left = style.get("margin-left", style.get("left", 0.0))
                top = style.get("margin-top", style.get("top", 0.0))
                page_w_pt, page_h_pt = page_w / EMU_PER_POINT, page_h / EMU_PER_POINT
                if (left + (width or 0) < -page_w_pt or left > 2 * page_w_pt
                        or top + (height or 0) < -page_h_pt or top > 2 * page_h_pt):
                    return "off_page"
                return None
        return None

    def _has_dark_decoration(self, element: etree._Element) -> bool:
        """A sizeable shape without text, anchored behind the text or floating over it,
        whose fill is dark or cannot be read."""
        for wsp in element.iter(f"{{{WPS_NS}}}wsp"):
            if self._skipped(wsp) or wsp.find("wps:txbx", NS) is not None:
                continue
            anchor = next(wsp.iterancestors(_WP_ANCHOR), None)
            if anchor is None:
                continue
            behind = (anchor.get("behindDoc") in ("1", "true")
                      or anchor.find("wp:wrapNone", NS) is not None)
            if not behind:
                continue
            extent = anchor.find("wp:extent", NS)
            area = (_int_attr(extent, "cx") or 0) * (_int_attr(extent, "cy") or 0)
            if area < MIN_DECORATIVE_SHAPE_EMU2:
                continue
            lum, known = self._shape_fill(wsp)
            if not known or (lum is not None and lum < DARK_BACKGROUND):
                return True
        for shape in element.iter(*_VML_SHAPES):
            if self._skipped(shape) or shape.find("v:textbox", NS) is not None:
                continue
            if shape.get("filled") in ("f", "false"):
                continue
            style = _vml_style(shape)
            area_pt2 = style.get("width", 0.0) * style.get("height", 0.0)
            if area_pt2 * EMU_PER_POINT * EMU_PER_POINT < MIN_DECORATIVE_SHAPE_EMU2:
                continue
            lum = _hex_luminance(shape.get("fillcolor"))
            if lum is None or lum < DARK_BACKGROUND:
                return True
        return False

    # ---- traversal ----------------------------------------------------------------------

    @staticmethod
    def _skipped(element: etree._Element) -> bool:
        return any(ancestor.tag in _SKIP_ANCESTORS for ancestor in element.iterancestors())

    def _paragraph(self, paragraph: etree._Element) -> list[str]:
        """[paragraph text, *lines of text boxes anchored in it]."""
        parts: list[str] = []
        nested: list[str] = []
        for run in paragraph.iter(_w("r")):
            if next(run.iterancestors(_w("p")), None) is not paragraph or self._skipped(run):
                continue
            text = _run_text(run)
            if text.strip():
                reason = self._hidden_reason(run, paragraph)
                if reason == UNVERIFIED_WHITE:
                    self.unverified_white_chars += sum(1 for ch in text if not ch.isspace())
                    self.hidden.end_run()
                elif reason is not None:
                    self.hidden.add(text, reason)
                    text = ""
                else:
                    self.hidden.end_run()
            parts.append(text)
            for box in run.iter(_w("txbxContent")):
                if next(box.iterancestors(_w("r")), None) is run and not self._skipped(box):
                    self.hidden.end_run()
                    nested.extend(self._text_box(box))
        self.hidden.end_run()
        return ["".join(parts), *nested]

    def _text_box(self, box: etree._Element) -> list[str]:
        reason = self._box_reason(box)
        if reason is None or self._forced_reason is not None:
            return self._block(box)
        self._forced_reason = reason
        try:
            return self._block(box)
        finally:
            self._forced_reason = None
            self.hidden.end_run()

    @staticmethod
    def _children(element: etree._Element, tag: str) -> Iterator[etree._Element]:
        """``tag`` children, looking through row/cell-level content controls."""
        for child in element:
            if child.tag == tag:
                yield child
            elif child.tag == _w("sdt"):
                content = child.find("w:sdtContent", NS)
                if content is not None:
                    yield from _DocxWalker._children(content, tag)
            elif child.tag == _w("customXml"):
                yield from _DocxWalker._children(child, tag)

    def _table(self, table: etree._Element) -> list[str]:
        tblpr = table.find("w:tblPr", NS)
        style = tblpr.find("w:tblStyle", NS) if tblpr is not None else None
        style_id = style.get(_w("val")) if style is not None else None
        look = _table_look(tblpr)
        rows = [row for row in self._children(table, _w("tr"))
                if row.find("w:trPr/w:del", NS) is None]  # tracked row deletions
        out: list[str] = []
        for r, row in enumerate(rows):
            cells = list(self._children(row, _w("tc")))
            lines_per_cell: list[list[str]] = []
            for c, cell in enumerate(cells):
                self._cells.append(_CellContext(style_id, look, r, len(rows), c, len(cells)))
                try:
                    lines_per_cell.append([t.strip() for t in self._block(cell) if t.strip()])
                finally:
                    self._cells.pop()
            if len(rows) == 1 or any(len(lines) > 1 for lines in lines_per_cell):
                for lines in lines_per_cell:  # layout table: one line per paragraph
                    out.extend(lines)
            else:
                joined = " | ".join(lines[0] for lines in lines_per_cell if lines)
                if joined:
                    out.append(joined)
        return out

    def _block(self, container: etree._Element) -> list[str]:
        out: list[str] = []
        for child in container:
            tag = child.tag
            if tag == _w("p"):
                out.extend(self._paragraph(child))
            elif tag == _w("tbl"):
                out.extend(self._table(child))
            elif tag == _w("sdt"):
                content = child.find("w:sdtContent", NS)
                if content is not None:
                    out.extend(self._block(content))
            elif tag in _CONTAINERS:
                out.extend(self._block(child))
            elif tag == f"{{{MC_NS}}}AlternateContent":
                choice = child.find("mc:Choice", NS)
                if choice is not None:
                    out.extend(self._block(choice))
        return out

    def _part_lines(self, element: etree._Element, displayed: bool) -> list[str]:
        self._forced_reason = None if displayed else "undisplayed_header"
        self._dark_decoration = self._has_dark_decoration(element)
        try:
            return self._block(element)
        finally:
            self._forced_reason = None
            self._dark_decoration = False
            self.hidden.end_run()

    # ---- parts --------------------------------------------------------------------------

    def _related_element(self, rid: str, reltype: str) -> etree._Element | None:
        rel = self._document.part.rels.get(rid)
        if rel is None or rel.is_external or rel.reltype != reltype:
            return None  # dangling or mislabelled reference: ignore it
        part = rel.target_part
        element = getattr(part, "element", None)
        return element if element is not None else parse_xml(part.blob)

    def _headers_and_footers(self) -> tuple[list[tuple[etree._Element, bool]],
                                            list[tuple[etree._Element, bool]]]:
        """(element, displayed) for every header and footer part, in section order.

        First-page parts show only in sections with w:titlePg, even-page parts only with
        w:evenAndOddHeaders; a section without a reference inherits the previous one's.
        """
        even_and_odd = _on(self._document.settings.element.find("w:evenAndOddHeaders", NS))
        body = self._document.element.body
        section_props = [sp for sp in body.iter(_w("sectPr"))
                         if sp.getparent() is body or sp.getparent().tag == _w("pPr")]
        current: dict[tuple[str, str], str] = {}
        shown: dict[str, dict[str, bool]] = {"header": {}, "footer": {}}
        for sect in section_props:
            title_page = _on(sect.find("w:titlePg", NS))
            for kind in ("header", "footer"):
                for ref in sect.iterfind(f"w:{kind}Reference", NS):
                    rid = ref.get(f"{{{R_NS}}}id")
                    if rid:
                        current[(kind, ref.get(_w("type"), "default"))] = rid
                for ref_type in ("default", "first", "even"):
                    rid = current.get((kind, ref_type))
                    if rid is None:
                        continue
                    displayed = (ref_type == "default" or (ref_type == "first" and title_page)
                                 or (ref_type == "even" and even_and_odd))
                    shown[kind][rid] = shown[kind].get(rid, False) or displayed
        result: list[list[tuple[etree._Element, bool]]] = []
        for kind in ("header", "footer"):
            parts: list[tuple[etree._Element, bool]] = []
            seen: set[int] = set()
            for rid, displayed in shown[kind].items():
                element = self._related_element(rid, RT.HEADER if kind == "header"
                                                else RT.FOOTER)
                if element is not None and id(element) not in seen:
                    seen.add(id(element))
                    parts.append((element, displayed))
            result.append(parts)
        return result[0], result[1]

    def _notes(self) -> list[etree._Element]:
        notes: list[etree._Element] = []
        for rel in self._document.part.rels.values():
            if rel.is_external or rel.reltype not in (RT.FOOTNOTES, RT.ENDNOTES):
                continue
            part = rel.target_part
            root = getattr(part, "element", None)
            if root is None:
                root = parse_xml(part.blob)
            for note in root:
                if note.tag in (_w("footnote"), _w("endnote")) and (
                        note.get(_w("type")) not in _NOTE_SKIP_TYPES):
                    notes.append(note)
        return notes

    def extract(self) -> str:
        headers, footers = self._headers_and_footers()
        lines: list[str] = []
        for element, displayed in headers:
            lines.extend(self._part_lines(element, displayed))
        lines.extend(self._part_lines(self._document.element.body, True))
        for note in self._notes():
            lines.extend(self._part_lines(note, True))
        for element, displayed in footers:
            lines.extend(self._part_lines(element, displayed))
        return "\n".join(line for line in lines if line.strip())


def _offset(anchor: etree._Element, path: str) -> int:
    """wp:positionH/V posOffset in EMU; 0 for alignment-based positions (on the page)."""
    return _text_int(anchor.find(f"{path}/wp:posOffset", NS))


def _text_int(element: etree._Element | None) -> int:
    if element is None or not element.text:
        return 0
    try:
        return int(element.text.strip())
    except ValueError:
        return 0


def extract_docx(data: bytes) -> ExtractionResult:
    try:
        document = docx.Document(io.BytesIO(data))
        walker = _DocxWalker(document)
        text = walker.extract()
    except Exception as exc:  # zip, OPC, XML and python-docx errors of every kind
        logger.info("DOCX could not be read (%s)", type(exc).__name__)
        return unreadable("docx", f"could not read DOCX file ({type(exc).__name__})")
    flag = walker.hidden.flag(REASON_LABELS)
    if walker.unverified_white_chars:
        note = (f"{walker.unverified_white_chars} characters of white text kept: a shape or "
                "fill behind them could not be resolved, so they may be on a dark background")
        if flag is None:
            flag = Flag(code=FlagCode.HIDDEN_TEXT, severity="info", detail=note)
        else:
            flag = flag.model_copy(update={"detail": f"{flag.detail}; {note}"})
    return ExtractionResult(
        kind="docx",
        visible_text=text,
        hidden_snippets=walker.hidden.snippets,
        hidden_char_count=walker.hidden.char_count,
        flags=[flag] if flag is not None else [],
    )
