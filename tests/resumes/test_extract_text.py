from __future__ import annotations

import codecs
import io
import zipfile

import pytest

from applicant_ranking.domain import FlagCode
from applicant_ranking.resumes.extract_text import (
    MAX_SNIPPET_CHARS,
    MAX_SNIPPETS,
    HiddenTextCollector,
    clean_unicode,
    decode_text,
    extract_odt,
    extract_txt,
    hidden_text_flag,
    unreadable,
)

LATIN_RESUME = (
    "Sam Example\nR\xe9sum\xe9 – na\xefve caf\xe9 owner turned data analyst\n"
    "Experience: S\xe3o Paulo office, built reporting with Python and SQL for 5 years.\n"
)
CYRILLIC_RESUME = (
    "Опытный инженер "
    "Python, Москва\n"
    "Разработка конв"
    "ейеров данных и "
    "отчётности для "
    "клиентов.\n"
)


@pytest.mark.parametrize(("data", "expected"), [
    (LATIN_RESUME.encode("utf-8"), LATIN_RESUME),
    (codecs.BOM_UTF8 + LATIN_RESUME.encode("utf-8"), LATIN_RESUME),
    (LATIN_RESUME.encode("utf-16"), LATIN_RESUME),  # BOM + native order
    (codecs.BOM_UTF16_BE + LATIN_RESUME.encode("utf-16-be"), LATIN_RESUME),
    (codecs.BOM_UTF32_LE + LATIN_RESUME.encode("utf-32-le"), LATIN_RESUME),
    (LATIN_RESUME.encode("cp1252"), LATIN_RESUME),
    (CYRILLIC_RESUME.encode("cp1251"), CYRILLIC_RESUME),
])
def test_decode_text(data, expected):
    assert decode_text(data) == expected


def test_short_cp1252_text_is_not_misread_as_central_european():
    assert decode_text("na\xefve caf\xe9 r\xe9sum\xe9".encode("cp1252")) == (
        "na\xefve caf\xe9 r\xe9sum\xe9")


def test_bytes_undefined_in_cp1252_use_the_detector():
    text = decode_text(b"Engineer \x81\x8d\x8f\x90\x9d end")
    assert text.startswith("Engineer ") and text.endswith(" end")


def test_extract_txt():
    result = extract_txt(codecs.BOM_UTF8 + b"Plain resume\nPython")

    assert result.kind == "txt"
    assert result.visible_text == "Plain resume\nPython"
    assert result.flags == []
    assert result.error is None


def test_extract_odt_collects_paragraphs_in_order(build_odt):
    body = (
        '<text:h text:outline-level="1">Morgan Example</text:h>'
        '<text:p>Senior <text:span>engineer</text:span>,<text:s text:c="3"/>Python'
        '<text:tab/>SQL<text:line-break/>Second line</text:p>'
        '<text:p>Before frame<draw:frame><draw:text-box><text:p>FRAMETEXT inside</text:p>'
        '</draw:text-box></draw:frame> after frame</text:p>'
        '<table:table><table:table-row><table:table-cell><text:p>CELL one</text:p>'
        '</table:table-cell></table:table-row></table:table>'
        '<text:p>Visible<office:annotation><text:p>COMMENTTEXT</text:p></office:annotation>'
        '<!-- XMLCOMMENT --> end</text:p>'
    )

    result = extract_odt(build_odt(body))

    assert result.kind == "odt"
    assert result.error is None
    assert result.visible_text.splitlines() == [
        "Morgan Example",
        "Senior engineer,   Python\tSQL",
        "Second line",
        "Before frame after frame",
        "FRAMETEXT inside",
        "CELL one",
        "Visible end",
    ]
    assert result.flags == []  # the hidden-text check runs for ODT as well
    assert result.hidden_char_count == 0


# --- ODT hidden text -------------------------------------------------------------------------

PAYLOAD = "Kubernetes Terraform Golang Rust Scala Snowflake Databricks PhD Stanford MIT"
ODT_STYLES = (
    '<style:style style:name="Hidden" style:family="text">'
    '<style:text-properties text:display="none"/></style:style>'
    '<style:style style:name="White" style:family="text">'
    '<style:text-properties fo:color="#ffffff"/></style:style>'
    '<style:style style:name="Tiny" style:family="text">'
    '<style:text-properties fo:font-size="1pt"/></style:style>'
    '<style:style style:name="Squash" style:family="text">'
    '<style:text-properties style:text-scale="1%"/></style:style>'
    '<style:style style:name="Grey" style:family="text">'
    '<style:text-properties fo:color="#808080"/></style:style>'
    '<style:style style:name="LightOnLight" style:family="text">'
    '<style:text-properties fo:color="#deeaf6" fo:background-color="#deeaf6"/></style:style>'
    '<style:style style:name="NavyCell" style:family="table-cell">'
    '<style:table-cell-properties fo:background-color="#1f3864"/></style:style>'
    '<style:style style:name="WhitePara" style:family="paragraph">'
    '<style:text-properties fo:color="#ffffff"/></style:style>'
    '<style:style style:name="HalfOfTiny" style:family="text" style:parent-style-name="Tiny">'
    '<style:text-properties fo:font-size="150%"/></style:style>'
)


def _hidden_odt(build_odt, body: str, **kwargs) -> tuple[list[str], object]:
    result = build_odt(f"<text:p>Morgan Example - Data Analyst</text:p>{body}",
                       automatic_styles=ODT_STYLES, **kwargs)
    extracted = extract_odt(result)
    return extracted.visible_text.splitlines(), extracted


def _span(style: str) -> str:
    return f'<text:p><text:span text:style-name="{style}">{PAYLOAD}</text:span></text:p>'


@pytest.mark.parametrize(("body", "label"), [
    (_span("Hidden"), "display: none"),
    (_span("White"), "white text"),
    (_span("Tiny"), "font under 3pt"),
    (_span("Squash"), "width under 20%"),
    (_span("LightOnLight"), "matching its background"),
    (f'<text:p text:style-name="WhitePara">{PAYLOAD}</text:p>', "white text"),
    (_span("HalfOfTiny"), "font under 3pt"),  # 150% of 1pt
    (f'<text:p>Field: <text:hidden-text text:condition="ooow:1" text:is-hidden="true">'
     f'{PAYLOAD}</text:hidden-text> end.</text:p>', "hidden text or hidden paragraph field"),
    (f'<text:p><text:hidden-paragraph text:condition="ooow:1" text:is-hidden="true"/>{PAYLOAD}'
     '</text:p>', "hidden text or hidden paragraph field"),
])
def test_odt_hidden_text_is_removed_and_reported(build_odt, body, label):
    lines, result = _hidden_odt(build_odt, body)

    assert "Kubernetes" not in result.visible_text
    assert lines[0] == "Morgan Example - Data Analyst"
    assert result.hidden_snippets == [PAYLOAD]
    assert result.hidden_char_count == len(PAYLOAD.replace(" ", ""))
    flag = {f.code: f for f in result.flags}[FlagCode.HIDDEN_TEXT]
    assert flag.severity == "review" and label in flag.detail


def test_odt_hidden_text_string_value_attribute_is_never_read(build_odt):
    body = (f'<text:p>Field: <text:hidden-text text:condition="ooow:1" '
            f'text:string-value="{PAYLOAD}" text:is-hidden="true"/> end.</text:p>')
    _, result = _hidden_odt(build_odt, body)

    assert "Kubernetes" not in result.visible_text
    assert result.hidden_char_count == 0 and result.flags == []


def test_odt_white_text_on_a_dark_cell_is_visible(build_odt):
    body = ('<table:table><table:table-row><table:table-cell table:style-name="NavyCell">'
            '<text:p><text:span text:style-name="White">SIDEBAR Python SQL</text:span></text:p>'
            '</table:table-cell></table:table-row></table:table>'
            '<text:p><text:span text:style-name="Grey">Grey meta line, Mar 2020</text:span>'
            '</text:p>')
    lines, result = _hidden_odt(build_odt, body)

    assert lines[1:] == ["SIDEBAR Python SQL", "Grey meta line, Mar 2020"]
    assert result.flags == []


def test_odt_styles_from_styles_xml_are_resolved(build_odt):
    styles = ('<style:style style:name="Ghost" style:family="paragraph">'
              '<style:text-properties text:display="none"/></style:style>')
    body = f'<text:p text:style-name="Ghost">{PAYLOAD}</text:p>'
    _, result = _hidden_odt(build_odt, body, styles_xml=styles)

    assert "Kubernetes" not in result.visible_text
    assert result.hidden_snippets == [PAYLOAD]


def test_odt_small_hidden_fragment_is_info_only(build_odt):
    body = '<text:p>Visible <text:span text:style-name="Hidden">tiny note</text:span> text</text:p>'
    lines, result = _hidden_odt(build_odt, body)

    assert lines[1] == "Visible  text"
    assert result.hidden_char_count == len("tinynote")
    assert {f.code: f for f in result.flags}[FlagCode.HIDDEN_TEXT].severity == "info"


def test_odt_oversized_part_is_unreadable(build_odt, monkeypatch):
    from applicant_ranking.resumes import extract_text as module

    monkeypatch.setattr(module, "ODT_MAX_PART_BYTES", 100)
    result = extract_odt(build_odt("<text:p>" + "x" * 200 + "</text:p>"))

    assert result.error is not None
    assert {f.code for f in result.flags} == {FlagCode.UNREADABLE_FILE}


def test_odt_external_entities_are_not_resolved(tmp_path, build_odt):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRETVALUE")
    doctype = f'<!DOCTYPE office:document-content [<!ENTITY xxe SYSTEM "file://{secret}">]>'

    result = extract_odt(build_odt("<text:p>Name &xxe; end</text:p>", doctype=doctype))

    assert "TOPSECRETVALUE" not in result.visible_text


@pytest.mark.parametrize("data", [b"PK\x03\x04 broken", b""])
def test_broken_odt_is_unreadable(data):
    result = extract_odt(data)

    assert result.error is not None
    assert {f.code for f in result.flags} == {FlagCode.UNREADABLE_FILE}


def test_odt_without_content_is_unreadable():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("mimetype", "application/vnd.oasis.opendocument.text")

    assert extract_odt(buf.getvalue()).error is not None


def test_collector_limits_snippets_but_counts_everything():
    collector = HiddenTextCollector()
    for i in range(MAX_SNIPPETS + 3):
        collector.add(f"hidden{i}", "white")
        collector.end_run()
    collector.add("x" * 500, "tiny")

    snippets = collector.snippets

    assert len(snippets) == MAX_SNIPPETS
    assert all(len(s) <= MAX_SNIPPET_CHARS for s in snippets)
    assert collector.char_count == sum(len(f"hidden{i}") for i in range(MAX_SNIPPETS + 3)) + 500
    assert collector.reasons == {"white": collector.char_count - 500, "tiny": 500}


def test_collector_merges_a_run_and_truncates_it():
    collector = HiddenTextCollector()
    collector.add("first", "a")
    collector.add("  ", "a")  # whitespace is neither counted nor a break
    collector.add("second\nline", "b")
    collector.add("y" * 300, "b")

    snippet = collector.snippets[0]

    assert snippet.startswith("first second line yyy")
    assert len(snippet) == MAX_SNIPPET_CHARS
    assert snippet.endswith("...")
    assert collector.char_count == len("firstsecondline") + 300


@pytest.mark.parametrize(("count", "severity"), [(0, None), (1, "info"), (19, "info"),
                                                 (20, "review"), (500, "review")])
def test_hidden_text_flag_threshold(count, severity):
    flag = hidden_text_flag(count, detail="d")

    assert (flag.severity if flag else None) == severity
    if flag:
        assert flag.code == FlagCode.HIDDEN_TEXT


def test_unreadable_result():
    result = unreadable("pdf", "boom")

    assert result.kind == "pdf"
    assert result.error == "boom"
    assert [(f.code, f.severity) for f in result.flags] == [(FlagCode.UNREADABLE_FILE, "review")]


@pytest.mark.parametrize(("text", "expected"), [
    ("plain", "plain"),
    ("pair 😀 ok", "pair \U0001F600 ok"),
    ("lone \ud83d end", "lone � end"),
    ("reversed \ude00\ud83d", "reversed ��"),
])
def test_clean_unicode(text, expected):
    assert clean_unicode(text) == expected
