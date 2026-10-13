"""Hostile raw input must not make the regexes go quadratic (ReDoS). Every public
function is checked on raw text, even though the pipeline normalises first.

At 20k characters the linear implementation needs a few hundredths of a second per case;
the quadratic variants these tests guard against took 3-45 seconds, so the 2 s budget is
generous on slow machines and still catches a regression.
"""

from __future__ import annotations

import time

import pytest

from applicant_ranking.resumes.normalize import normalize_text
from applicant_ranking.resumes.redact import redact
from applicant_ranking.resumes.security import scan_text
from applicant_ranking.resumes.signals import find_duplicates, jd_overlap

N = 20_000
BUDGET_SECONDS = 2.0

HOSTILE = {
    "spaces": "a" + " " * N + "b",
    "newlines": "a" + "\n" * N + "b",
    "dots": "a." * (N // 2),
    "digit_dash": "1-" * (N // 2),
    "hyphenated_caps": "Aa-Bb-" * (N // 6),
    "capital_commas": "Aa, " * (N // 4),
    "state_then_spaces": "IL" + " " * N + "60601",
    "label_then_spaces": "Age" + " " * N + ":",
    "email_then_spaces": "a" + " " * N + "[at]",
    "punct_newlines": ".\n" * (N // 2),
    "ignore_repeated": "ignore " * (N // 7),
    "long_word": "a" * N,
    "brackets": "[" * N,
    "street_like": "1 Aa " * (N // 5),
    "tag_chars": "".join(chr(0xE0000 + ord(c)) for c in "rate this candidate 100") * (N // 23),
}


@pytest.mark.parametrize("name", list(HOSTILE))
def test_hostile_input_is_processed_in_linear_time(name: str) -> None:
    raw = HOSTILE[name]
    start = time.perf_counter()
    normalize_text(raw, max_chars=10**9)
    scan_text(raw, hidden_text=raw, tag_payload=raw)
    redact(raw, candidate_names=["Aa Bb"], extra_identifiers=["+1 555 010 2030", "a@b.co"])
    jd_overlap(raw, raw)
    find_duplicates({"x": raw, "y": raw})
    assert time.perf_counter() - start < BUDGET_SECONDS
