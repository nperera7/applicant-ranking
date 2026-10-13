from __future__ import annotations

import hashlib

import pytest

from applicant_ranking.resumes.redact import redact
from applicant_ranking.resumes.signals import (
    find_duplicates,
    fingerprint,
    jaccard,
    jd_overlap,
    shingles,
)

BODY = """\
EXPERIENCE
Senior Data Engineer, Acme Analytics (2021 - present)
- Built production data pipelines on GCP using BigQuery and Dataflow for 12 clients.
- Wrote Python services and SQL transformations processing 2 TB per day.
"""


def _long(seed: str, words: int = 120) -> str:
    return " ".join(f"{seed}{i}" for i in range(words))


# --- fingerprint ---------------------------------------------------------------------------


def test_fingerprint_is_sha256_of_the_normalised_text() -> None:
    assert fingerprint("Hello,   World!\n") == hashlib.sha256(b"hello world").hexdigest()


def test_fingerprint_ignores_case_punctuation_and_whitespace() -> None:
    assert fingerprint("Built ETL -- in Python!") == fingerprint("built etl in python")
    assert fingerprint("Built ETL in Python") != fingerprint("Built ETL in Java")


def test_fingerprint_collides_for_resumes_differing_only_in_redacted_identity() -> None:
    a = "Jordan Lee\njordan@example.com | (312) 555-0100 | Chicago, IL\n" + BODY
    b = "Priya Raman\npriya.r@mail.net | +1 415 555 0199 | Austin, Texas\n" + BODY
    ra = redact(a, candidate_names=["Jordan Lee"]).text
    rb = redact(b, candidate_names=["Priya Raman"]).text
    assert ra == rb
    assert fingerprint(ra) == fingerprint(rb)
    assert fingerprint(a) != fingerprint(b)


def test_fingerprint_keeps_non_latin_letters() -> None:
    # Different Cyrillic resumes with the same digits must not collide.
    assert fingerprint("Инженер данных 2019 2021") != fingerprint("Аналитик продаж 2019 2021")
    assert fingerprint("Résumé") == fingerprint("resume")


# --- shingles and jaccard -----------------------------------------------------------------


def test_shingles() -> None:
    assert shingles("a b c d e f", 5) == {"a b c d e", "b c d e f"}
    assert shingles("A, b; C!", 2) == {"a b", "b c"}
    assert shingles("only four words here", 5) == set()
    assert shingles("", 1) == set()


def test_shingles_rejects_bad_n() -> None:
    with pytest.raises(ValueError):
        shingles("a b", 0)


def test_jaccard() -> None:
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard({"a"}, {"a"}) == 1.0
    assert jaccard(set(), {"a"}) == 0.0
    assert jaccard(set(), set()) == 0.0


# --- find_duplicates --------------------------------------------------------------------------


def test_exact_duplicates_are_symmetric_and_need_no_minimum_length() -> None:
    texts = {"o1": "Short resume text.", "o2": "short   RESUME text", "o3": "Different."}
    assert find_duplicates(texts) == {"o1": ["o2"], "o2": ["o1"]}


def test_exact_group_of_three_in_input_order() -> None:
    texts = {"z": BODY, "a": "unique", "m": BODY, "b": BODY}
    assert find_duplicates(texts) == {"z": ["m", "b"], "m": ["z", "b"], "b": ["z", "m"]}


def test_empty_texts_are_never_duplicates() -> None:
    assert find_duplicates({"o1": "", "o2": "  ", "o3": "[...]"}) == {}


def test_min_exact_words_option() -> None:
    texts = {"o1": "Resume attached", "o2": "resume attached"}
    assert find_duplicates(texts, min_exact_words=5) == {}


def test_near_duplicate_with_one_word_changed() -> None:
    base = _long("w", 200)
    changed = base.replace("w100", "changed")
    assert find_duplicates({"a": base, "b": changed}) == {"a": ["b"], "b": ["a"]}


def test_near_duplicates_need_min_words() -> None:
    base = _long("w", 60)
    changed = base.replace("w30", "changed")
    assert find_duplicates({"a": base, "b": changed}) == {}
    assert find_duplicates({"a": base, "b": changed}, min_words=50, near_threshold=0.7) == {
        "a": ["b"], "b": ["a"]}


def test_near_threshold_is_inclusive() -> None:
    base = _long("w", 200)
    changed = base.replace("w100", "x")
    j = jaccard(shingles(base), shingles(changed))
    assert 0.9 < j < 1.0
    pair = {"a": ["b"], "b": ["a"]}
    assert find_duplicates({"a": base, "b": changed}, near_threshold=j) == pair
    assert find_duplicates({"a": base, "b": changed}, near_threshold=j + 1e-9) == {}


def test_unrelated_long_resumes_are_not_duplicates() -> None:
    assert find_duplicates({"a": _long("alpha"), "b": _long("beta")}) == {}


def test_size_filter_skips_very_different_lengths() -> None:
    short = _long("w", 100)
    longer = short + " " + _long("extra", 400)
    assert find_duplicates({"a": short, "b": longer}) == {}


def test_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        find_duplicates({}, near_threshold=0)


# --- jd_overlap ------------------------------------------------------------------------------

JD = ("We are looking for a senior data engineer to design build and operate production "
      "data pipelines on Google Cloud using BigQuery Dataflow and Pub Sub for our clients")


def test_jd_fully_copied() -> None:
    assert jd_overlap("Summary\n" + JD + "\nSkills: Python", JD) == 1.0


def test_jd_not_copied() -> None:
    assert jd_overlap(BODY, JD) == 0.0


def test_jd_partly_copied() -> None:
    words = JD.split()
    half = " ".join(words[: len(words) // 2])
    overlap = jd_overlap(half, JD)
    assert 0.3 < overlap < 0.6


def test_jd_without_shingles_is_zero() -> None:
    assert jd_overlap("anything", "") == 0.0
    assert jd_overlap("short jd here", "short jd here") == 0.0  # fewer than 8 words


def test_jd_html_is_stripped() -> None:
    html_jd = "<ul><li>" + JD.replace(" and ", " &amp; and ", 1) + "</li></ul>"
    assert jd_overlap(JD.replace(" and ", " & and ", 1), html_jd) == 1.0
