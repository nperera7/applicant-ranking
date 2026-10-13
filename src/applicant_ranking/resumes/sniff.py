"""Identify a resume file's type from its bytes. Never trust the filename, the Lever
``ext`` or the HTTP Content-Type: a DOCX renamed .pdf must be read as DOCX.

Rules (verified on synthetic samples):
- ``%PDF-`` within the first 1024 bytes -> pdf (real PDFs may have junk before it).
- ``PK\\x03\\x04`` zip containing ``word/document.xml`` -> docx; ODF ``mimetype`` entry
  ``application/vnd.oasis.opendocument.text`` -> odt. Zip-bomb guards: total
  uncompressed size <= 50 MB (images included), XML parts (``*.xml``, ``*.rels``) <= 8 MB
  together (an lxml tree takes about 25x the XML's size in RAM, so 45 MB of ``<w:p/>``
  would cost over a gigabyte in the extraction worker) and <= 2000 entries, else unknown.
  A zip that is neither -> unknown.
- OLE magic ``D0 CF 11 E0 A1 B1 1A E1`` with a ``WordDocument`` stream -> doc
  (an ``EncryptedPackage`` stream means a password-protected docx -> unknown).
- ``{\\rtf`` -> rtf.
- Image magics: PNG, JPEG ``FF D8 FF``, RIFF....WEBP, ``ftyp`` heic/heix/mif1, GIF8,
  TIFF (II*\\0 / MM\\0*) -> image.
- Otherwise txt if there are no NUL bytes and < 1% control bytes (excluding \\t \\r \\n \\f);
  else unknown. Text that starts with a UTF-16/UTF-32 byte-order mark (and so contains
  NUL bytes) is judged on its decoded characters instead.
"""

from __future__ import annotations

import codecs
import io
import logging
import zipfile

import olefile

from applicant_ranking.domain import FileKind

logger = logging.getLogger(__name__)

MAX_ZIP_UNCOMPRESSED = 50 * 1024 * 1024
MAX_ZIP_XML_UNCOMPRESSED = 8 * 1024 * 1024
MAX_ZIP_ENTRIES = 2000
_XML_SUFFIXES = (".xml", ".rels")

PDF_MAGIC = b"%PDF-"
PDF_SEARCH_WINDOW = 1024
ZIP_MAGIC = b"PK\x03\x04"
OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
ODT_MIMETYPE = b"application/vnd.oasis.opendocument.text"
MAX_CONTROL_SHARE = 0.01

_HEIF_BRANDS = frozenset({b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"avif"})
# Control bytes that disqualify text: C0 controls other than \t \n \f \r, plus DEL.
_CONTROL_BYTES = bytes(b for b in [*range(0x20), 0x7F] if b not in (0x09, 0x0A, 0x0C, 0x0D))
_CONTROL_CHARS = dict.fromkeys(_CONTROL_BYTES)  # str.translate table deleting them
_WIDE_BOMS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)


def sniff(data: bytes) -> FileKind:
    head = data[:16]
    if PDF_MAGIC in data[:PDF_SEARCH_WINDOW]:
        return "pdf"
    if head.startswith(ZIP_MAGIC):
        return _sniff_zip(data)
    if head.startswith(OLE_MAGIC):
        return _sniff_ole(data)
    if data[:64].removeprefix(codecs.BOM_UTF8).lstrip(b" \t\r\n").startswith(b"{\\rtf"):
        return "rtf"
    if _is_image(head):
        return "image"
    if _looks_like_text(data):
        return "txt"
    return "unknown"


def _sniff_zip(data: bytes) -> FileKind:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_ZIP_ENTRIES:
                logger.info("zip rejected: %d entries", len(infos))
                return "unknown"
            if sum(info.file_size for info in infos) > MAX_ZIP_UNCOMPRESSED:
                logger.info("zip rejected: uncompressed size over %d bytes",
                            MAX_ZIP_UNCOMPRESSED)
                return "unknown"
            xml_size = sum(info.file_size for info in infos
                           if info.filename.lower().endswith(_XML_SUFFIXES))
            if xml_size > MAX_ZIP_XML_UNCOMPRESSED:
                logger.info("zip rejected: XML parts total %d bytes (limit %d)", xml_size,
                            MAX_ZIP_XML_UNCOMPRESSED)
                return "unknown"
            names = {info.filename for info in infos}
            if "word/document.xml" in names:
                return "docx"
            if "mimetype" in names and "content.xml" in names:
                with archive.open("mimetype") as handle:
                    mimetype = handle.read(200).strip()
                if mimetype == ODT_MIMETYPE:
                    return "odt"
    except Exception:  # malformed or hostile archives of every kind
        return "unknown"
    return "unknown"


def _sniff_ole(data: bytes) -> FileKind:
    try:
        with olefile.OleFileIO(io.BytesIO(data)) as ole:
            if ole.exists("EncryptedPackage"):
                return "unknown"  # password-protected OOXML
            if ole.exists("WordDocument"):
                return "doc"
    except Exception:  # olefile raises many exception types on malformed input
        return "unknown"
    return "unknown"


def _is_image(head: bytes) -> bool:
    return (
        head.startswith(b"\x89PNG\r\n\x1a\n")
        or head.startswith(b"\xff\xd8\xff")
        or (head[:4] == b"RIFF" and head[8:12] == b"WEBP")
        or (head[4:8] == b"ftyp" and head[8:12] in _HEIF_BRANDS)
        or head.startswith(b"GIF8")
        or head[:4] in (b"II*\x00", b"MM\x00*")
    )


def _looks_like_text(data: bytes) -> bool:
    if not data:
        return False
    for bom, encoding in _WIDE_BOMS:
        if data.startswith(bom):
            try:
                text = data[len(bom):].decode(encoding)
            except UnicodeDecodeError:
                return False
            control = len(text) - len(text.translate(_CONTROL_CHARS))
            return control < MAX_CONTROL_SHARE * max(len(text), 1)
    if b"\x00" in data:
        return False
    control = len(data) - len(data.translate(None, _CONTROL_BYTES))
    return control < MAX_CONTROL_SHARE * len(data)
