"""PDF text extraction with hidden-text detection and a scanned-PDF fallback.

Approach (verified in research; see docs/design.md):
- Reading order: pdfminer.six ``extract_pages(io.BytesIO(data), laparams=LAParams(),
  maxpages=max_pages)`` — the only extractor that kept multi-column resumes in order.
  (pdfminer does not accept raw bytes; wrap in BytesIO.) ``all_texts=True`` is set so text
  inside form XObjects is laid out (and checked) like any other text.
- Tables: pdfminer groups vertically aligned cells into one text box per column, so a
  skills matrix would read column by column. A post-layout pass joins text boxes into a
  grid when at least ``GRID_MIN_COLUMNS`` of them have disjoint x-ranges, overlapping
  y-ranges and line baselines that coincide (within ``GRID_BASELINE_TOLERANCE_PT``) for
  at least ``GRID_ALIGNED_SHARE`` of their lines, or when exactly two such boxes are
  separated or framed by drawn rulings. A grid is emitted row by row with the cells
  joined by " | "; everything else keeps pdfminer's order, so a two-column resume whose
  sidebar happens to align with the main column still reads column-wise.
- Hidden text, per pdfminer word (chars grouped within LTTextLine):
  * ``off_page``: bbox outside the page's visible area. pdfminer coordinates are relative
    to the MediaBox while pypdfium2 renders the CropBox: map with
    px = (x - (cx0 - mx0)) * scale, py = ((cy1 - my0) - y) * scale, and treat anything
    outside the CropBox as off-page. (Implemented generally: pdfminer coordinates are
    mapped back to PDF user space and then through PDFium's own page-to-device transform,
    which also covers /Rotate.)
  * ``tiny_font``: char size < ``TINY_FONT_PT`` (4pt; legitimate footers are 6pt+).
  * ``squashed``: the glyph advances are under ``SQUASHED_ADVANCE_RATIO`` of the font
    size (horizontal scaling ``Tz`` of a few percent) or consecutive glyph origins step
    by under ``SQUASHED_STEP_RATIO`` of the advance (a negative ``Tc`` piling the glyphs
    on top of each other). Either way the word is an unreadable sliver.
  * ``transparent``: PDFium reports a fill alpha under ``TRANSPARENT_ALPHA`` for a
    filling render mode (``FPDFPageObj_GetFillColor``).
  * ``clipped``: the text object's clip path (``FPDFPageObj_GetClipPath``; PDFium only
    reports clips that restrict the object) leaves most of the word's glyphs outside it.
  * ``no_ink``: render the page with pypdfium2 at ``render_scale`` (greyscale, once with
    and once without text anti-aliasing) and take the luminance range of the word's
    bbox; a range under ``INK_CONTRAST_THRESHOLD`` in both renders means nothing visible
    is drawn there (white-on-white, covered text). Taking the larger range keeps thin
    light glyphs that anti-aliasing washes out (a grey Times "|"). When PDFium knows the
    word's fill colour, the word is also ``no_ink`` when that colour is within the
    threshold of the median of the one-pixel frame around the bbox: the text matches its
    surroundings even if other ink (a line underneath, a rule through it) lies inside the
    box. Black text under an opaque image is the accepted gap of this test.
  * ``invisible_mode``: pypdfium2's raw API reports text render mode 3 (invisible) for
    the character (``pypdfium2.raw.FPDFTextObj_GetTextRenderMode``) and the page is not
    an image-backed scan (image coverage < 0.5). This catches invisible text drawn over
    existing visible ink, which the ink check misses. OCR layers over scans are mode 3
    too and are legitimate. (PDFium characters are matched to pdfminer characters by
    character and origin. A page counts as an image-backed scan only when its text layer
    is also mostly invisible, so a decorative full-page background image cannot be used
    to launder an invisible overlay. Clip-only text, mode 7, is left to the ink check:
    gradient-filled headings legitimately paint through a text clip.)
  Hidden words are removed from ``visible_text`` and reported in ``hidden_snippets``
  (a few short samples, <= 200 chars each) and ``hidden_char_count``; add a
  ``hidden_text`` flag with severity "review" when hidden_char_count >= 20 ("info" below).
  A lone separator glyph ("|", "•", "-") judged ``no_ink`` is dropped from the text but
  not counted, so decorative separators alone can never force a review.
- Scanned/image-only: if visible chars < 100 * pages (and page image coverage is high
  or text is near-empty), set ``needs_vision=True`` and rasterise up to
  ``vision_max_pages`` pages at 150 dpi greyscale to PNG (``page_images``) plus a
  ``scanned_resume`` info flag. Never return the original PDF bytes: Gemini reads a PDF's
  embedded text layer, which would re-expose hidden text. The same fallback is used when
  PDFium can render the file but pdfminer cannot parse its text.
- Encrypted with a user password, or unparsable: ``error`` set, ``unreadable_file`` flag
  (review). Owner-password-only PDFs extract normally.
- PDFium is NOT thread-safe: every pypdfium2 call must hold ``PDFIUM_LOCK`` (the
  extractor also runs parsing in worker processes in production).
"""

from __future__ import annotations

import ctypes
import io
import logging
import math
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LAParams,
    LTChar,
    LTContainer,
    LTFigure,
    LTImage,
    LTLine,
    LTPage,
    LTRect,
    LTTextContainer,
    LTTextLine,
)
from pdfminer.pdfdocument import PDFPasswordIncorrect
from PIL import Image

from applicant_ranking.domain import ExtractionResult, Flag, FlagCode
from applicant_ranking.resumes.extract_text import HiddenTextCollector, unreadable

logger = logging.getLogger(__name__)

PDFIUM_LOCK = threading.Lock()
INK_CONTRAST_THRESHOLD = 40
TINY_FONT_PT = 4.0
TRANSPARENT_ALPHA = 26  # of 255: a fill under 10% opaque paints nothing readable
SQUASHED_ADVANCE_RATIO = 0.08  # mean glyph advance / font size; a normal "i" is about 0.22
SQUASHED_STEP_RATIO = 0.5  # origin step between glyphs / glyph advance; kerning stays > 0.8
FILLING_RENDER_MODES = frozenset({0, 2, 4, 6})  # modes that paint the glyph interior

SCAN_CHARS_PER_PAGE = 100  # fewer visible characters per page than this -> maybe a scan
NEAR_EMPTY_CHARS_PER_PAGE = 20
SCAN_IMAGE_COVERAGE = 0.5
VISION_DPI = 150
INVISIBLE_RENDER_MODES = frozenset({3})  # FPDF_TEXTRENDERMODE_INVISIBLE
ORIGIN_TOLERANCE_PT = 1.0
OFF_PAGE_VISIBLE_SHARE = 0.5  # less of the word's box than this on the page -> off_page
CLIP_VISIBLE_SHARE = 0.5  # fewer glyph centres than this inside the clip -> clipped
MAX_INK_RENDER_PIXELS = 8_000_000  # cap per page for the hidden-text render (letter at 2x: 1.9M)
MAX_VISION_RENDER_PIXELS = 4_000_000

GRID_MIN_COLUMNS = 3
GRID_BASELINE_TOLERANCE_PT = 2.0
GRID_ALIGNED_SHARE = 0.8
GRID_OVERLAP_SHARE = 0.8
RULING_MAX_THICKNESS_PT = 2.0
SEPARATOR_GLYPHS = frozenset("|•·-–—.,:;")  # noqa: RUF001  (every dash is intended)

REASON_LABELS = {
    "off_page": "outside the visible page",
    "tiny_font": "font under 4pt",
    "squashed": "glyphs squashed or piled up",
    "transparent": "fully transparent fill",
    "clipped": "clipped away",
    "invisible_mode": "invisible text render mode",
    "no_ink": "no visible ink (e.g. white or covered text)",
}

Matrix = tuple[float, float, float, float, float, float]
Bbox = tuple[float, float, float, float]


# ----------------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------------


def _apply(m: Matrix, x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = m
    return a * x + c * y + e, b * x + d * y + f


def _compose(first: Matrix, then: Matrix) -> Matrix:
    """Matrix that applies ``first`` and then ``then``."""
    a1, b1, c1, d1, e1, f1 = first
    a2, b2, c2, d2, e2, f2 = then
    return (a1 * a2 + b1 * c2, a1 * b2 + b1 * d2,
            c1 * a2 + d1 * c2, c1 * b2 + d1 * d2,
            e1 * a2 + f1 * c2 + e2, e1 * b2 + f1 * d2 + f2)


def _invert(m: Matrix) -> Matrix:
    a, b, c, d, e, f = m
    det = a * d - b * c
    if det == 0:
        raise ValueError("singular matrix")
    ia, ib, ic, id_ = d / det, -b / det, -c / det, a / det
    return ia, ib, ic, id_, -(e * ia + f * ic), -(e * ib + f * id_)


def _pdfminer_ctm(rotation: int, mediabox: Bbox) -> Matrix:
    """User space -> pdfminer page coordinates (mirrors PDFPageInterpreter.process_page)."""
    x0, y0, x1, y1 = mediabox
    if rotation == 90:
        return (0, -1, 1, 0, -y0, x1)
    if rotation == 180:
        return (-1, 0, 0, -1, x1, y1)
    if rotation == 270:
        return (0, 1, -1, 0, y1, -x0)
    return (1, 0, 0, 1, -x0, -y0)


def _bbox_pixels(m: Matrix, bbox: Bbox) -> Bbox:
    x0, y0, x1, y1 = bbox
    points = [_apply(m, x, y) for x, y in ((x0, y0), (x0, y1), (x1, y0), (x1, y1))]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


class _OriginIndex[T]:
    """Spatial hash of (character, origin) -> value for tolerant lookups."""

    def __init__(self, tolerance: float = ORIGIN_TOLERANCE_PT) -> None:
        self._tol = tolerance
        self._cell = 2 * tolerance
        self._grid: dict[tuple[str, int, int], list[tuple[float, float, T]]] = {}
        self.count = 0

    def add(self, ch: str, x: float, y: float, value: T) -> None:
        key = (ch, int(x // self._cell), int(y // self._cell))
        self._grid.setdefault(key, []).append((x, y, value))
        self.count += 1

    def get(self, ch: str, x: float, y: float) -> T | None:
        if not self.count:
            return None
        gx, gy = int(x // self._cell), int(y // self._cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for px, py, value in self._grid.get((ch, gx + dx, gy + dy), ()):
                    if abs(px - x) <= self._tol and abs(py - y) <= self._tol:
                        return value
        return None

    def contains(self, ch: str, x: float, y: float) -> bool:
        return self.get(ch, x, y) is not None


# ----------------------------------------------------------------------------------------
# PDFium pass: renders, page transforms, render modes, fills and clips (under PDFIUM_LOCK)
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Ink:
    """How PDFium paints one text object."""

    mode: int
    grey: int  # fill colour as the greyscale render sees it, 0-255
    alpha: int  # fill alpha, 0-255
    clip: Bbox | None  # clip path bbox in user space, when it restricts the object

    @property
    def fills(self) -> bool:
        return self.mode in FILLING_RENDER_MODES


@dataclass
class _PageRender:
    image: Image.Image  # greyscale render of the visible area (CropBox)
    sharp: Image.Image  # the same without text anti-aliasing
    to_pixels: Matrix  # pdfminer page coordinates -> pixel coordinates in ``image``
    to_user: Matrix  # pdfminer page coordinates -> PDF user space (PDFium text space)
    invisible: _OriginIndex[bool] = field(default_factory=_OriginIndex)
    inks: _OriginIndex[_Ink] = field(default_factory=_OriginIndex)
    visible_mode_chars: int = 0


def _user_to_display(page: pdfium.PdfPage) -> Matrix:
    """PDF user space -> display points (top-left origin), from PDFium's own transform."""
    width, height = page.get_size()
    k = 64.0  # device pixels per point for the probe; keeps integer rounding negligible
    size_x, size_y = max(1, round(width * k)), max(1, round(height * k))

    def probe(x: float, y: float) -> tuple[float, float]:
        dx, dy = ctypes.c_int(), ctypes.c_int()
        ok = pdfium_c.FPDF_PageToDevice(page, 0, 0, size_x, size_y, 0, x, y,
                                         ctypes.byref(dx), ctypes.byref(dy))
        if not ok:
            raise pdfium.PdfiumError("FPDF_PageToDevice failed")
        return dx.value / k, dy.value / k

    unit = 100.0
    ox, oy = probe(0.0, 0.0)
    xx, xy = probe(unit, 0.0)
    yx, yy = probe(0.0, unit)
    return ((xx - ox) / unit, (xy - oy) / unit, (yx - ox) / unit, (yy - oy) / unit, ox, oy)


def _scale_for(page: pdfium.PdfPage, wanted: float, max_pixels: int) -> float:
    width, height = page.get_size()
    area = max(width * height, 1.0)
    return min(wanted, math.sqrt(max_pixels / area))


def _fill_grey_alpha(obj: pdfium_c.FPDF_PAGEOBJECT) -> tuple[int, int] | None:
    r, g, b, a = (ctypes.c_uint() for _ in range(4))
    if not pdfium_c.FPDFPageObj_GetFillColor(obj, ctypes.byref(r), ctypes.byref(g),
                                             ctypes.byref(b), ctypes.byref(a)):
        return None
    grey = (b.value * 11 + g.value * 59 + r.value * 30) // 100  # PDFium's grey conversion
    return grey, a.value


def _clip_bbox(obj: pdfium_c.FPDF_PAGEOBJECT) -> Bbox | None:
    clip = pdfium_c.FPDFPageObj_GetClipPath(obj)
    if not clip:
        return None
    xs: list[float] = []
    ys: list[float] = []
    x, y = ctypes.c_float(), ctypes.c_float()
    for path_index in range(max(0, pdfium_c.FPDFClipPath_CountPaths(clip))):
        for seg_index in range(max(0, pdfium_c.FPDFClipPath_CountPathSegments(clip, path_index))):
            segment = pdfium_c.FPDFClipPath_GetPathSegment(clip, path_index, seg_index)
            if segment and pdfium_c.FPDFPathSegment_GetPoint(segment, ctypes.byref(x),
                                                              ctypes.byref(y)):
                xs.append(x.value)
                ys.append(y.value)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _collect_render_modes(page: pdfium.PdfPage, render: _PageRender) -> None:
    textpage = page.get_textpage()
    try:
        x, y = ctypes.c_double(), ctypes.c_double()
        per_object: dict[int, tuple[tuple[int, int] | None, Bbox | None]] = {}
        for index in range(textpage.count_chars()):
            obj = pdfium_c.FPDFText_GetTextObject(textpage, index)
            if not obj:
                continue  # generated by PDFium (spaces/line breaks), not in the file
            code = pdfium_c.FPDFText_GetUnicode(textpage, index)
            if not 0 < code < 0x110000:
                continue
            ch = chr(code)
            if ch.isspace():
                continue
            mode = pdfium_c.FPDFTextObj_GetTextRenderMode(obj)
            if not pdfium_c.FPDFText_GetCharOrigin(textpage, index, ctypes.byref(x),
                                                   ctypes.byref(y)):
                continue
            if mode in INVISIBLE_RENDER_MODES:
                render.invisible.add(ch, x.value, y.value, True)
            else:
                render.visible_mode_chars += 1
            key = ctypes.cast(obj, ctypes.c_void_p).value or 0
            if key not in per_object:
                per_object[key] = (_fill_grey_alpha(obj), _clip_bbox(obj))
            fill, clip = per_object[key]
            if fill is not None:
                render.inks.add(ch, x.value, y.value, _Ink(mode, fill[0], fill[1], clip))
    finally:
        textpage.close()


def _render_page(page: pdfium.PdfPage, render_scale: float) -> _PageRender:
    rotation = page.get_rotation()
    mediabox = page.get_mediabox()
    scale = _scale_for(page, render_scale, MAX_INK_RENDER_PIXELS)
    images: list[Image.Image] = []
    for flags in (0, pdfium_c.FPDF_RENDER_NO_SMOOTHTEXT):
        bitmap = page.render(scale=scale, grayscale=True, extra_flags=flags)
        try:
            images.append(bitmap.to_pil().convert("L"))  # copy: detach from PDFium's buffer
        finally:
            bitmap.close()
    to_user = _invert(_pdfminer_ctm(rotation, mediabox))
    display = _user_to_display(page)
    to_pixels = _compose(_compose(to_user, display), (scale, 0, 0, scale, 0, 0))
    render = _PageRender(image=images[0], sharp=images[1], to_pixels=to_pixels, to_user=to_user)
    _collect_render_modes(page, render)
    return render


def _render_pages(data: bytes, max_pages: int, render_scale: float
                  ) -> tuple[int, list[_PageRender]]:
    with PDFIUM_LOCK:
        doc = pdfium.PdfDocument(data)
        try:
            total = len(doc)
            renders: list[_PageRender] = []
            for index in range(min(total, max_pages)):
                page = doc[index]
                try:
                    renders.append(_render_page(page, render_scale))
                finally:
                    page.close()
            return total, renders
        finally:
            doc.close()


def rasterize_pdf(data: bytes, *, max_pages: int, dpi: int = VISION_DPI) -> list[bytes]:
    """Render pages to greyscale PNG bytes (holding PDFIUM_LOCK)."""
    images: list[Image.Image] = []
    with PDFIUM_LOCK:
        doc = pdfium.PdfDocument(data)
        try:
            for index in range(min(len(doc), max_pages)):
                page = doc[index]
                try:
                    scale = _scale_for(page, dpi / 72.0, MAX_VISION_RENDER_PIXELS)
                    bitmap = page.render(scale=scale, grayscale=True)
                    try:
                        images.append(bitmap.to_pil().convert("L"))
                    finally:
                        bitmap.close()
                finally:
                    page.close()
        finally:
            doc.close()
    pngs = []
    for image in images:
        out = io.BytesIO()
        image.save(out, format="PNG")
        pngs.append(out.getvalue())
    return pngs


# ----------------------------------------------------------------------------------------
# pdfminer pass: reading order and per-word classification
# ----------------------------------------------------------------------------------------


def _iter_text_boxes(container: LTContainer) -> Iterator[LTTextContainer]:
    for item in container:
        if isinstance(item, LTTextContainer) and not isinstance(item, LTTextLine):
            yield item
        elif isinstance(item, LTTextLine):
            yield item  # a bare line outside a box: treat it as a one-line box
        elif isinstance(item, LTFigure):
            yield from _iter_text_boxes(item)


def _iter_lines(box: LTTextContainer) -> Iterator[LTTextLine]:
    if isinstance(box, LTTextLine):
        yield box
        return
    for line in box:
        if isinstance(line, LTTextLine):
            yield line


def _iter_words(line: LTTextLine) -> Iterator[list[LTChar]]:
    word: list[LTChar] = []
    for item in line:
        if isinstance(item, LTChar) and not item.get_text().isspace():
            word.append(item)
        elif word:
            yield word
            word = []
    if word:
        yield word


def _iter_rulings(container: LTContainer) -> Iterator[Bbox]:
    """Thin drawn lines and rectangles (table rulings)."""
    for item in container:
        if isinstance(item, (LTLine, LTRect)):
            if item.height < RULING_MAX_THICKNESS_PT or item.width < RULING_MAX_THICKNESS_PT:
                yield item.bbox
        elif isinstance(item, LTFigure):
            yield from _iter_rulings(item)


def _image_coverage(page: LTPage) -> float:
    width, height = page.width, page.height
    if width <= 0 or height <= 0:
        return 0.0
    covered = 0.0

    def walk(container: LTContainer) -> None:
        nonlocal covered
        for item in container:
            if isinstance(item, LTImage):
                w = max(0.0, min(item.x1, width) - max(item.x0, 0.0))
                h = max(0.0, min(item.y1, height) - max(item.y0, 0.0))
                covered += w * h
            elif isinstance(item, LTContainer):
                walk(item)

    walk(page)
    return min(1.0, covered / (width * height))


@dataclass
class _PageContext:
    layout: LTPage
    render: _PageRender | None
    scan_page: bool


def _word_bbox(chars: list[LTChar]) -> Bbox:
    return (min(c.x0 for c in chars), min(c.y0 for c in chars),
            max(c.x1 for c in chars), max(c.y1 for c in chars))


def _visible_share(bbox: Bbox, width: float, height: float
                   ) -> tuple[float, tuple[int, int, int, int] | None]:
    """Share of ``bbox`` inside [0, width] x [0, height], and the clipped integer box."""
    x0, y0, x1, y1 = bbox
    cx0, cy0, cx1, cy1 = max(x0, 0.0), max(y0, 0.0), min(x1, width), min(y1, height)
    area = (x1 - x0) * (y1 - y0)
    if area <= 0:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        inside = 0 <= mx <= width and 0 <= my <= height
        share = 1.0 if inside else 0.0
    else:
        share = max(0.0, cx1 - cx0) * max(0.0, cy1 - cy0) / area
    if share <= 0:
        return 0.0, None
    ix0, iy0 = math.floor(cx0), math.floor(cy0)
    ix1, iy1 = max(math.ceil(cx1), ix0 + 1), max(math.ceil(cy1), iy0 + 1)
    return share, (ix0, iy0, ix1, iy1)


def _is_upright(char: LTChar) -> bool:
    """Horizontal, unrotated text (italic skew allowed). LTChar.upright is unreliable for
    90-degree rotations computed in floating point (cos 90 = 6e-17 counts as upright)."""
    a, b, c, d = char.matrix[:4]
    return a > 0 and d > 0 and abs(b) <= 0.05 * a and abs(c) <= 0.4 * d


def _glyph_size(char: LTChar) -> float:
    """Font size as drawn. LTChar.size is the box height, which for rotated text is the
    glyph's advance (a sideways "i" would look like 2pt text)."""
    return char.size if _is_upright(char) else max(char.width, char.height)


def _squashed(chars: list[LTChar]) -> bool:
    """Glyphs scaled to a sliver (Tz) or piled onto each other (negative Tc)."""
    letters = [c for c in chars if _is_upright(c) and c.adv > 0 and c.height > 0
               and c.get_text()[:1].isalnum()]
    if not letters:
        return False
    if sum(c.adv / c.height for c in letters) / len(letters) < SQUASHED_ADVANCE_RATIO:
        return True
    if len(letters) < 2:
        return False
    # Forward progress between glyph origins; a backwards step is no progress at all.
    steps = sum(max(b.matrix[4] - a.matrix[4], 0.0) for a, b in pairwise(letters))
    advances = sum(a.adv for a in letters[:-1])
    return advances > 0 and steps / advances < SQUASHED_STEP_RATIO


def _frame_median(image: Image.Image, crop: tuple[int, int, int, int]) -> int:
    """Median luminance of the one-pixel frame just outside ``crop`` (clamped to the
    image): what surrounds the word."""
    x0, y0, x1, y1 = crop
    width, height = image.size
    fx0, fy0 = max(x0 - 1, 0), max(y0 - 1, 0)
    fx1, fy1 = min(x1 + 1, width), min(y1 + 1, height)
    pixels = bytearray()
    if fy0 < y0:
        pixels += image.crop((fx0, fy0, fx1, y0)).tobytes()
    if y1 < fy1:
        pixels += image.crop((fx0, y1, fx1, fy1)).tobytes()
    if fx0 < x0:
        pixels += image.crop((fx0, y0, x0, y1)).tobytes()
    if x1 < fx1:
        pixels += image.crop((x1, y0, fx1, y1)).tobytes()
    if not pixels:
        return 255
    return sorted(pixels)[len(pixels) // 2]


def _ink_range(render: _PageRender, crop: tuple[int, int, int, int]) -> int:
    ranges = []
    for image in (render.image, render.sharp):
        low, high = image.crop(crop).getextrema()
        ranges.append(high - low)
    return max(ranges)


def _matched_inks(chars: list[LTChar], render: _PageRender) -> list[_Ink]:
    inks: list[_Ink] = []
    for c in chars:
        ux, uy = _apply(render.to_user, c.matrix[4], c.matrix[5])
        ink = render.inks.get(c.get_text()[:1], ux, uy)
        if ink is not None:
            inks.append(ink)
    return inks


def _clipped(chars: list[LTChar], render: _PageRender, clip: Bbox) -> bool:
    cx0, cy0, cx1, cy1 = clip
    inside = 0
    for c in chars:
        ux, uy = _apply(render.to_user, (c.x0 + c.x1) / 2, (c.y0 + c.y1) / 2)
        if cx0 - 0.5 <= ux <= cx1 + 0.5 and cy0 - 0.5 <= uy <= cy1 + 0.5:
            inside += 1
    return inside < CLIP_VISIBLE_SHARE * len(chars)


def _classify(chars: list[LTChar], ctx: _PageContext) -> str | None:
    """Reason the word is hidden, or None when it is visible."""
    bbox = _word_bbox(chars)
    render = ctx.render
    if render is not None:
        pixel_bbox = _bbox_pixels(render.to_pixels, bbox)
        share, crop = _visible_share(pixel_bbox, *render.image.size)
    else:
        share, crop = _visible_share(bbox, ctx.layout.width, ctx.layout.height)
    if share < OFF_PAGE_VISIBLE_SHARE:
        return "off_page"
    if max(_glyph_size(c) for c in chars) < TINY_FONT_PT:
        return "tiny_font"
    if _squashed(chars):
        return "squashed"
    if render is None or crop is None:
        return None
    if not ctx.scan_page and render.invisible.count:
        matched = sum(1 for c in chars if render.invisible.contains(
            c.get_text()[:1], *_apply(render.to_user, c.matrix[4], c.matrix[5])))
        if matched * 2 >= len(chars):
            return "invisible_mode"
    inks = _matched_inks(chars, render)
    ink = inks[0] if inks and len(inks) * 2 >= len(chars) else None
    if ink is not None:
        if ink.fills and ink.alpha < TRANSPARENT_ALPHA:
            return "transparent"
        if ink.clip is not None and _clipped(chars, render, ink.clip):
            return "clipped"
    if _ink_range(render, crop) < INK_CONTRAST_THRESHOLD:
        return "no_ink"
    if ink is not None and ink.fills and abs(
            ink.grey - _frame_median(render.image, crop)) < INK_CONTRAST_THRESHOLD:
        return "no_ink"
    return None


# ----------------------------------------------------------------------------------------
# Grids (tables laid out as one text box per column)
# ----------------------------------------------------------------------------------------


@dataclass
class _Column:
    box: LTTextContainer
    lines: list[LTTextLine]
    baselines: list[float]

    @property
    def bbox(self) -> Bbox:
        return self.box.bbox


def _aligned_share(a: _Column, b: _Column) -> float:
    if not a.baselines:
        return 0.0
    hits = sum(1 for y in a.baselines
               if any(abs(y - other) <= GRID_BASELINE_TOLERANCE_PT for other in b.baselines))
    return hits / len(a.baselines)


def _grid_compatible(a: _Column, b: _Column) -> bool:
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    if not (ax1 <= bx0 + 1.0 or bx1 <= ax0 + 1.0):
        return False  # x-ranges overlap
    overlap = min(ay1, by1) - max(ay0, by0)
    if overlap < GRID_OVERLAP_SHARE * min(ay1 - ay0, by1 - by0):
        return False
    return (_aligned_share(a, b) >= GRID_ALIGNED_SHARE
            and _aligned_share(b, a) >= GRID_ALIGNED_SHARE)


def _ruled(group: Sequence[_Column], rulings: Sequence[Bbox]) -> bool:
    """Two columns separated by a vertical ruling or framed by two horizontal ones."""
    left, right = sorted(group, key=lambda c: c.bbox[0])
    gy0 = min(c.bbox[1] for c in group) - GRID_BASELINE_TOLERANCE_PT
    gy1 = max(c.bbox[3] for c in group) + GRID_BASELINE_TOLERANCE_PT
    gx0, gx1 = left.bbox[0] + 2.0, right.bbox[2] - 2.0
    horizontals = 0
    for rx0, ry0, rx1, ry1 in rulings:
        vertical = rx1 - rx0 < RULING_MAX_THICKNESS_PT
        if vertical and left.bbox[2] <= rx0 <= right.bbox[0] and ry0 <= gy1 and ry1 >= gy0:
            return True
        if not vertical and rx0 <= gx0 and rx1 >= gx1 and gy0 <= ry0 <= gy1:
            horizontals += 1
    return horizontals >= 2


def _same_columns(a: Sequence[_Column], b: Sequence[_Column]) -> bool:
    """Two rows of cells whose columns overlap in x, pairwise."""
    if len(a) != len(b):
        return False
    for ca, cb in zip(sorted(a, key=lambda c: c.bbox[0]), sorted(b, key=lambda c: c.bbox[0]),
                      strict=True):
        if min(ca.bbox[2], cb.bbox[2]) - max(ca.bbox[0], cb.bbox[0]) <= 0:
            return False
    return True


def _grid_groups(columns: list[_Column], rulings: Sequence[Bbox]) -> dict[int, list[_Column]]:
    """id(box) -> the grid (all its cells) a box belongs to, for boxes that form a table.

    pdfminer boxes a table either as one box per column (multi-line boxes, which form a
    grid on their own) or as one box per cell (single-line boxes, where one aligned row is
    not enough: a title, a right-aligned date and a sidebar word can share a baseline, so a
    table needs at least two rows with the same columns).
    """
    groups: list[list[_Column]] = []
    for column in sorted(columns, key=lambda c: c.bbox[0]):
        for group in groups:
            if all(_grid_compatible(column, member) for member in group):
                group.append(column)
                break
        else:
            groups.append([column])
    tables: list[list[_Column]] = []
    rows: list[list[_Column]] = []
    for group in groups:
        if not (len(group) >= GRID_MIN_COLUMNS or (len(group) == 2 and _ruled(group, rulings))):
            continue
        if any(len(column.lines) > 1 for column in group):
            tables.append(group)
        else:
            rows.append(group)
    stacked: list[list[list[_Column]]] = []
    for row in sorted(rows, key=lambda g: -max(c.bbox[3] for c in g)):
        for table in stacked:
            if _same_columns(table[-1], row):
                table.append(row)
                break
        else:
            stacked.append([row])
    tables.extend([cell for row in table for cell in row] for table in stacked if len(table) >= 2)
    return {id(column.box): table for table in tables for column in table}


def _grid_rows(group: Sequence[_Column]) -> list[list[LTTextLine]]:
    """Lines of the grid clustered by baseline, top row first, cells left to right."""
    entries = sorted(((y, line) for column in group
                      for y, line in zip(column.baselines, column.lines, strict=True)),
                     key=lambda e: -e[0])
    rows: list[list[LTTextLine]] = []
    current_y: float | None = None
    for y, line in entries:
        if current_y is None or abs(y - current_y) > GRID_BASELINE_TOLERANCE_PT:
            rows.append([])
            current_y = y
        rows[-1].append(line)
    return [sorted(row, key=lambda line: line.x0) for row in rows]


# ----------------------------------------------------------------------------------------
# Text pass
# ----------------------------------------------------------------------------------------


@dataclass
class _TextPass:
    text: str
    pages_read: int
    visible_chars: int
    rotated_chars: int  # visible characters that are not upright (sideways pages)
    mean_image_coverage: float
    hidden: HiddenTextCollector


@dataclass
class _Tally:
    visible_chars: int = 0
    rotated_chars: int = 0


def _line_text(line: LTTextLine, ctx: _PageContext, hidden: HiddenTextCollector,
               tally: _Tally) -> str:
    """Visible words of one line; hidden words go to the collector."""
    words: list[str] = []
    for chars in _iter_words(line):
        word = "".join(c.get_text() for c in chars)
        reason = _classify(chars, ctx)
        if reason is None:
            hidden.end_run()
            words.append(word)
            tally.visible_chars += len(word)
            tally.rotated_chars += sum(len(c.get_text()) for c in chars if not _is_upright(c))
        elif reason == "no_ink" and len(word) == 1 and word in SEPARATOR_GLYPHS:
            continue  # a washed-out separator: dropped, but not evidence of hidden text
        else:
            hidden.add(word, reason)
    return " ".join(words)


def _extract_text(data: bytes, renders: list[_PageRender], max_pages: int) -> _TextPass:
    hidden = HiddenTextCollector()
    page_texts: list[str] = []
    tally = _Tally()
    coverages: list[float] = []
    laparams = LAParams(all_texts=True)
    for index, layout in enumerate(extract_pages(io.BytesIO(data), laparams=laparams,
                                                 maxpages=max_pages)):
        render = renders[index] if index < len(renders) else None
        coverage = _image_coverage(layout)
        coverages.append(coverage)
        scan_page = (render is not None and coverage >= SCAN_IMAGE_COVERAGE
                     and render.invisible.count >= max(1, render.visible_mode_chars))
        ctx = _PageContext(layout=layout, render=render, scan_page=scan_page)
        columns = []
        for box in _iter_text_boxes(layout):
            lines = list(_iter_lines(box))
            columns.append(_Column(box, lines, [round(line.y0 * 2) / 2 for line in lines]))
        grids = _grid_groups(columns, list(_iter_rulings(layout)))
        emitted: set[int] = set()
        blocks: list[str] = []
        for column in columns:
            group = grids.get(id(column.box))
            if group is not None:
                if id(group[0].box) in emitted:
                    continue
                emitted.add(id(group[0].box))
                rows = []
                for row in _grid_rows(group):
                    cells = [text for text in (_line_text(line, ctx, hidden, tally)
                                               for line in row) if text]
                    if cells:
                        rows.append(" | ".join(cells))
                hidden.end_run()
                if rows:
                    blocks.append("\n".join(rows))
                continue
            lines = [text for text in (_line_text(line, ctx, hidden, tally)
                                       for line in column.lines) if text]
            hidden.end_run()
            if lines:
                blocks.append("\n".join(lines))
        page_texts.append("\n\n".join(blocks))
    pages_read = len(page_texts)
    return _TextPass(
        text="\n\n".join(t for t in page_texts if t),
        pages_read=pages_read,
        visible_chars=tally.visible_chars,
        rotated_chars=tally.rotated_chars,
        mean_image_coverage=sum(coverages) / pages_read if pages_read else 0.0,
        hidden=hidden,
    )


# ----------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------


def _is_password_error(exc: pdfium.PdfiumError) -> bool:
    return getattr(exc, "err_code", None) == pdfium_c.FPDF_ERR_PASSWORD


def _vision_result(data: bytes, *, pages: int, vision_pages: int, detail: str,
                   text: str = "", flags: list[Flag] | None = None,
                   hidden: HiddenTextCollector | None = None) -> ExtractionResult:
    images = rasterize_pdf(data, max_pages=vision_pages)
    all_flags = list(flags or [])
    all_flags.append(Flag(code=FlagCode.SCANNED_RESUME, severity="info",
                          detail=f"{detail}; {len(images)} page(s) rendered for transcription"))
    return ExtractionResult(
        kind="pdf", visible_text=text, pages=pages, needs_vision=True, page_images=images,
        hidden_snippets=hidden.snippets if hidden else [],
        hidden_char_count=hidden.char_count if hidden else 0,
        flags=all_flags,
    )


def extract_pdf(data: bytes, *, max_pages: int = 10, vision_max_pages: int = 5,
                render_scale: float = 2.0) -> ExtractionResult:
    try:
        total_pages, renders = _render_pages(data, max_pages, render_scale)
    except pdfium.PdfiumError as exc:
        if _is_password_error(exc):
            return unreadable("pdf", "PDF is password-protected")
        return unreadable("pdf", "PDF could not be parsed")
    if total_pages == 0:
        return unreadable("pdf", "PDF has no pages")

    flags: list[Flag] = []
    if total_pages > max_pages:
        flags.append(Flag(code=FlagCode.TRUNCATED, severity="info",
                          detail=f"only the first {max_pages} of {total_pages} PDF pages "
                                 "were read"))
    vision_pages = min(vision_max_pages, max_pages)

    try:
        text_pass = _extract_text(data, renders, max_pages)
    except PDFPasswordIncorrect:
        return unreadable("pdf", "PDF is password-protected")
    except Exception as exc:  # pdfminer raises many types on malformed files
        logger.warning("pdfminer could not parse a PDF that PDFium opened (%s); "
                       "falling back to page images", type(exc).__name__)
        return _vision_result(data, pages=total_pages, vision_pages=vision_pages,
                              detail="the PDF text layer could not be parsed", flags=flags)

    hidden_flag = text_pass.hidden.flag(REASON_LABELS)
    if hidden_flag is not None:
        flags.append(hidden_flag)

    pages_read = max(text_pass.pages_read, 1)
    few_chars = text_pass.visible_chars < SCAN_CHARS_PER_PAGE * pages_read
    near_empty = text_pass.visible_chars < NEAR_EMPTY_CHARS_PER_PAGE * pages_read
    if few_chars and (text_pass.mean_image_coverage >= SCAN_IMAGE_COVERAGE or near_empty):
        return _vision_result(
            data, pages=total_pages, vision_pages=vision_pages, text=text_pass.text,
            flags=flags, hidden=text_pass.hidden,
            detail=f"image-only PDF ({text_pass.visible_chars} visible characters on "
                   f"{text_pass.pages_read} page(s))",
        )
    if text_pass.rotated_chars * 2 > text_pass.visible_chars:
        # pdfminer lays sideways text out one character per line; read the pages instead
        return _vision_result(
            data, pages=total_pages, vision_pages=vision_pages, text=text_pass.text,
            flags=flags, hidden=text_pass.hidden,
            detail="most of the PDF text is rotated and cannot be read in order",
        )
    return ExtractionResult(
        kind="pdf",
        visible_text=text_pass.text,
        hidden_snippets=text_pass.hidden.snippets,
        hidden_char_count=text_pass.hidden.char_count,
        pages=total_pages,
        flags=flags,
    )
