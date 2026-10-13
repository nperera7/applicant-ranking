from __future__ import annotations

import pytest

from applicant_ranking.domain import FlagCode
from applicant_ranking.resumes.extract_doc import (
    DocFormatError,
    extract_doc,
    text_from_streams,
)


def flags_by_code(result):
    return {flag.code: flag for flag in result.flags}


def test_textutil_word97_file(synthetic_doc_file):
    result = extract_doc(synthetic_doc_file.read_bytes())

    assert result.kind == "doc"
    assert result.error is None
    lines = result.visible_text.splitlines()
    assert lines[0] == "Riley Synthetic"
    assert "Data Engineer, Example Corp (2019 - present)" in lines
    assert "Python, SQL, Airflow" in lines
    flag = flags_by_code(result)[FlagCode.HIDDEN_TEXT_CHECK_UNAVAILABLE]
    assert flag.severity == "info"


def test_compressed_and_unicode_pieces_with_header_story(build_doc):
    data = build_doc(
        [("Compressed piece with caf\xe9. ", True), ("Unicode piece: Krak\xf3w Ł\xf3dź", False)],
        ccp_hdd_text="Header story text\r",
    )

    result = extract_doc(data)

    assert result.error is None
    assert "Compressed piece with caf\xe9. Unicode piece: Krak\xf3w Ł\xf3dź" in (
        result.visible_text)
    assert "Header story text" in result.visible_text
    assert FlagCode.HIDDEN_TEXT_CHECK_UNAVAILABLE in flags_by_code(result)


def test_field_codes_are_dropped_and_results_kept(build_word_streams):
    word, table = build_word_streams([
        ("Profile \x13 HYPERLINK \"https://example.com/INSTR\" \x14LinkText\x15 end\r", True),
        ("Nested \x13 IF \x13 PAGE \x142\x15 = 2 \x14Yes\x15 done\r", True),
    ])

    text = text_from_streams(word, table)

    assert "INSTR" not in text and "HYPERLINK" not in text
    assert "Profile LinkText end" in text
    assert "Nested Yes done" in text


def test_control_characters_are_mapped(build_word_streams):
    raw = "Cell A\x07Cell B\x07\x07Line\x0bbreak\x0cpage non\x1ebreaking soft\x1fhyphen\x01\r"
    word, table = build_word_streams([(raw, True)])

    text = text_from_streams(word, table)

    assert text == "Cell A\tCell B\t\tLine\nbreak\npage non-breaking softhyphen\n"


def test_zero_table_stream_selected_by_flag(build_doc):
    result = extract_doc(build_doc([("Uses the 0Table stream\r", True)], table1=False))

    assert result.visible_text.strip() == "Uses the 0Table stream"


def test_encrypted_doc_is_unreadable(build_doc):
    result = extract_doc(build_doc([("secret", True)], encrypted=True))

    assert result.error == "Word document is password-protected"
    assert flags_by_code(result)[FlagCode.UNREADABLE_FILE].severity == "review"


@pytest.mark.parametrize("kwargs", [{"magic": 0x1234}, {"nfib": 0x65}])
def test_bad_fib_is_unreadable(build_doc, kwargs):
    result = extract_doc(build_doc([("text", True)], **kwargs))

    assert result.error is not None
    assert flags_by_code(result)[FlagCode.UNREADABLE_FILE].severity == "review"


def test_missing_table_stream_is_unreadable(build_ole, build_word_streams):
    word, _ = build_word_streams([("text", True)])

    result = extract_doc(build_ole({"WordDocument": word}))

    assert result.error is not None
    assert FlagCode.UNREADABLE_FILE in flags_by_code(result)


@pytest.mark.parametrize("table", [b"", b"\x02\xff\xff\xff\x7f", b"\x05garbage", b"\x01\xff\xff"])
def test_malformed_piece_table_raises_format_error(build_word_streams, table):
    word, _ = build_word_streams([("text", True)])

    with pytest.raises(DocFormatError):
        text_from_streams(word, table)


def test_piece_table_cannot_amplify_output(build_word_streams, monkeypatch):
    from applicant_ranking.resumes import extract_doc as module

    monkeypatch.setattr(module, "MAX_DOC_CHARS", 50)
    word, table = build_word_streams([("x" * 40, True), ("y" * 40, True), ("z" * 40, True)])

    assert len(text_from_streams(word, table)) <= 50


# --- hidden character formatting in .doc files ---------------------------------------------------

import struct  # noqa: E402

VISIBLE = "Riley Synthetic, data engineer. Built pipelines on GCP. "
PAYLOAD = "HIDDENMARK Kubernetes Terraform Spark Airflow Scala"
TAIL = " More visible experience follows.\r"
VANISH = struct.pack("<HB", 0x083C, 1)
SPEC_VANISH = struct.pack("<HB", 0x0818, 1)
WHITE_CV = struct.pack("<H", 0x6870) + bytes([0xFF, 0xFF, 0xFF, 0x00])
WHITE_ICO = struct.pack("<HB", 0x2A42, 8)
TINY = struct.pack("<HH", 0x4A43, 4)
SQUASHED = struct.pack("<HH", 0x4852, 5)
SHADING = struct.pack("<HH", 0x4866, 0x0000)
BOLD = struct.pack("<HB", 0x0835, 1)


def _doc_with_run(build_doc, grpprl: bytes) -> bytes:
    start, end = len(VISIBLE), len(VISIBLE) + len(PAYLOAD)
    return build_doc([(VISIBLE + PAYLOAD + TAIL, True)], chpx=[(start, end, grpprl)])


@pytest.mark.parametrize(("grpprl", "label"), [
    (VANISH, "vanish"),
    (SPEC_VANISH, "vanish"),
    (WHITE_CV, "white text"),
    (WHITE_ICO, "white text"),
    (TINY, "font under 3pt"),
    (SQUASHED, "width under 20%"),
    (BOLD + WHITE_CV, "white text"),  # other modifiers in the same run are skipped over
])
def test_hidden_character_runs_are_removed_and_reported(build_doc, grpprl, label):
    result = extract_doc(_doc_with_run(build_doc, grpprl))

    assert result.error is None
    assert "HIDDENMARK" not in result.visible_text
    assert result.visible_text.split() == (VISIBLE + TAIL).split()
    assert result.hidden_snippets == [PAYLOAD]
    assert result.hidden_char_count == len(PAYLOAD.replace(" ", ""))
    flag = flags_by_code(result)[FlagCode.HIDDEN_TEXT]
    assert flag.severity == "review" and label in flag.detail
    assert "direct character formatting" in flags_by_code(result)[
        FlagCode.HIDDEN_TEXT_CHECK_UNAVAILABLE].detail


@pytest.mark.parametrize("grpprl", [WHITE_CV + SHADING, BOLD, struct.pack("<HH", 0x4A43, 20)])
def test_shaded_white_bold_and_normal_runs_stay_visible(build_doc, grpprl):
    result = extract_doc(_doc_with_run(build_doc, grpprl))

    assert PAYLOAD in result.visible_text
    assert result.hidden_char_count == 0
    assert FlagCode.HIDDEN_TEXT not in flags_by_code(result)


def test_small_hidden_run_is_info_only(build_doc):
    text = VISIBLE + "note" + TAIL
    data = build_doc([(text, True)], chpx=[(len(VISIBLE), len(VISIBLE) + 4, VANISH)])

    result = extract_doc(data)

    assert "note" not in result.visible_text
    assert result.hidden_char_count == 4
    assert flags_by_code(result)[FlagCode.HIDDEN_TEXT].severity == "info"


def test_garbage_is_unreadable():
    result = extract_doc(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 100)

    assert result.kind == "doc"
    assert result.error is not None
