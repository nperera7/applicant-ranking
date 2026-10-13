from __future__ import annotations

import codecs
import io
import zipfile

import pytest
from PIL import Image

from applicant_ranking.resumes.sniff import sniff


def image_bytes(fmt: str, mode: str = "RGB") -> bytes:
    out = io.BytesIO()
    Image.new(mode, (8, 8)).save(out, format=fmt)
    return out.getvalue()


def zip_with(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buf.getvalue()


def test_pdf(pdfs):
    assert sniff(pdfs.one_column) == "pdf"


def test_pdf_with_junk_before_header(pdfs):
    assert sniff(pdfs.junk_prefix) == "pdf"


def test_pdf_header_beyond_first_kilobyte_is_not_pdf():
    assert sniff(b"x" * 1100 + b"%PDF-1.7\n") != "pdf"


def test_docx_renamed_to_pdf_is_sniffed_by_content(docxs, tmp_path):
    path = tmp_path / "resume.pdf"
    path.write_bytes(docxs.full)

    assert sniff(path.read_bytes()) == "docx"


def test_odt(build_odt):
    assert sniff(build_odt("<text:p>hello</text:p>")) == "odt"


@pytest.mark.parametrize("entries", [
    {"readme.txt": b"just a zip"},
    {"mimetype": b"application/vnd.oasis.opendocument.spreadsheet", "content.xml": b"<x/>"},
    {"mimetype": b"application/vnd.oasis.opendocument.text"},  # no content.xml
])
def test_other_zips_are_unknown(entries):
    assert sniff(zip_with(entries)) == "unknown"


def test_zip_bomb_by_size_is_unknown(zip_bomb_size):
    assert len(zip_bomb_size) < 1024 * 1024  # small on disk
    assert sniff(zip_bomb_size) == "unknown"


def test_zip_bomb_by_entry_count_is_unknown(zip_bomb_entries):
    assert sniff(zip_bomb_entries) == "unknown"


def test_xml_bomb_docx_is_unknown(xml_bomb_docx):
    assert len(xml_bomb_docx) < 200 * 1024  # tiny on disk
    assert sniff(xml_bomb_docx) == "unknown"


def test_docx_with_a_large_image_but_small_xml_is_docx(docx_with_large_image):
    assert sniff(docx_with_large_image) == "docx"


def test_odt_with_oversized_content_xml_is_unknown():
    body = "<text:p>x</text:p>" * 600_000  # about 11 MB of XML
    content = ('<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:'
               'office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
               f'<office:body><office:text>{body}</office:text></office:body>'
               '</office:document-content>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        archive.writestr("content.xml", content)

    assert sniff(buf.getvalue()) == "unknown"


def test_corrupt_zip_is_unknown():
    assert sniff(b"PK\x03\x04" + b"\x00" * 200) == "unknown"


def test_doc(build_doc):
    assert sniff(build_doc([("Hello", True)])) == "doc"


def test_textutil_doc(synthetic_doc_file):
    assert sniff(synthetic_doc_file.read_bytes()) == "doc"


def test_password_protected_docx_is_unknown(build_ole):
    data = build_ole({"EncryptionInfo": b"\x04\x00\x04\x00", "EncryptedPackage": b"\x00" * 64})
    assert sniff(data) == "unknown"


@pytest.mark.parametrize("data", [
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 600,
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
])
def test_broken_or_non_word_ole_is_unknown(data):
    assert sniff(data) == "unknown"


def test_non_word_ole_is_unknown(build_ole):
    assert sniff(build_ole({"Workbook": b"\x00" * 10})) == "unknown"


@pytest.mark.parametrize("data", [b"{\\rtf1\\ansi hello}", b"\xef\xbb\xbf\r\n {\\rtf1 x}"])
def test_rtf(data):
    assert sniff(data) == "rtf"


@pytest.mark.parametrize(("fmt", "mode"), [("PNG", "RGB"), ("JPEG", "RGB"), ("GIF", "P"),
                                           ("TIFF", "RGB"), ("WEBP", "RGB")])
def test_images(fmt, mode):
    assert sniff(image_bytes(fmt, mode)) == "image"


@pytest.mark.parametrize("brand", [b"heic", b"heix", b"mif1"])
def test_heif_brands(brand):
    assert sniff(b"\x00\x00\x00\x18ftyp" + brand + b"\x00" * 20) == "image"


def test_big_endian_tiff():
    assert sniff(b"MM\x00*" + b"\x00" * 20) == "image"


@pytest.mark.parametrize("data", [
    b"Plain resume text\nPython, SQL\n",
    "R\xe9sum\xe9 caf\xe9".encode("cp1252"),
    "Опыт".encode("cp1251"),
    b"Sam\tExample\r\nPage\x0cbreak",
    codecs.BOM_UTF8 + b"utf-8 with bom",
    "utf-16 with bom: R\xe9sum\xe9".encode("utf-16"),
    codecs.BOM_UTF16_BE + "big endian".encode("utf-16-be"),
    codecs.BOM_UTF32_LE + "utf-32".encode("utf-32-le"),
])
def test_text(data):
    assert sniff(data) == "txt"


def test_control_byte_threshold():
    base = b"a" * 199
    assert sniff(base + b"\x01") == "txt"  # 0.5% control bytes
    assert sniff(b"a" * 98 + b"\x01\x02") == "unknown"  # 2%


@pytest.mark.parametrize("data", [b"", b"text with a NUL\x00 byte", b"\x00\x01\x02\x03binary",
                                  codecs.BOM_UTF16_LE + b"\x00"])
def test_unknown(data):
    assert sniff(data) == "unknown"
