"""Counterfactual fairness harness: variant construction and the flagging rule, run through
the real Ranker.evaluate_local with a scripted model; plus the criteria approval gate the
fairness evidence feeds (``RoleCriteria.approved`` / ``approval_issue``)."""

from __future__ import annotations

import difflib
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from applicant_ranking.domain import RoleCriteria
from applicant_ranking.fairness import (
    AFFINITY_LINES,
    GAP_LINE,
    GROUPS,
    MISSING_NAME_CAVEAT,
    NAME_VARIANTS,
    PRONOUN_LINES,
    ZIP_LINES,
    FairnessReport,
    Variant,
    make_variants,
    name_in_text,
    run_fairness,
)
from applicant_ranking.llm.base import AssessmentRequest, LLMUnavailableError
from applicant_ranking.pipeline import RankingError
from applicant_ranking.resumes.redact import redact

EXPECTED_NAMES = [
    "base", "name_emily_walsh", "name_lakisha_washington", "name_jose_hernandez",
    "pronouns_she_her", "pronouns_he_him", "pronouns_they_them",
    "affinity_swe", "affinity_nsbe", "affinity_ieee", "employment_gap",
    "grad_year_1995", "grad_year_2020", "zip_60614", "zip_60621",
]
NSBE = "National Society of Black Engineers"
# A resume whose author is named in the header (in capitals) and again in the body.
QUINLAN = "Quinlan Testworth-Abara"
QUINLAN_TEXT = (
    "QUINLAN TESTWORTH-ABARA\n"
    "EXPERIENCE\n"
    "- Quinlan designed the ingestion pipelines (2018 - 2021)\n"
    "- Abara and the team shipped a Kafka platform (2015 - 2018)\n"
    "References: the Testworth family business\n"
)
QUINLAN_WORDS = ("Quinlan", "Testworth", "Abara")


@pytest.fixture
def kit(pipeline_kit: SimpleNamespace) -> SimpleNamespace:
    return pipeline_kit


def changes(base: str, variant: str) -> list[tuple[str, list[str], list[str]]]:
    a, b = base.split("\n"), variant.split("\n")
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return [(tag, a[i1:i2], b[j1:j2]) for tag, i1, i2, j1, j2 in matcher.get_opcodes()
            if tag != "equal"]


def by_name(variants: list[Variant]) -> dict[str, Variant]:
    return {v.name: v for v in variants}


# ------------------------------------------------------------------------- make_variants


def test_every_group_is_present(sample_resume_text: str) -> None:
    variants = make_variants(sample_resume_text)
    assert [v.name for v in variants] == EXPECTED_NAMES
    assert {v.group for v in variants} == set(GROUPS)
    base = variants[0]
    assert (base.name, base.group) == ("base", "base")
    assert base.text == "Jordan Lee\n" + sample_resume_text
    assert base.candidate_names == ["Jordan Lee"]


def test_each_variant_changes_only_its_own_text(sample_resume_text: str) -> None:
    variants = by_name(make_variants(sample_resume_text))
    base = variants["base"].text
    lines = base.split("\n")

    for label, name in NAME_VARIANTS:
        v = variants[label]
        # Redacted with both names, so a leftover piece of the base name never gets through.
        assert v.group == "name" and v.candidate_names == [name, "Jordan Lee"]
        assert v.text == base.replace("Jordan Lee", name)
        assert changes(base, v.text) == [("replace", ["Jordan Lee"], [name])]
    for label, line in PRONOUN_LINES:
        v = variants[label]
        assert changes(base, v.text) == [("insert", [], [line])]
        assert v.text.split("\n")[1] == line
    pronoun_words = [set(variants[label].text.split()) ^ set(base.split())
                     for label, _ in PRONOUN_LINES]
    assert {"She", "her"} <= pronoun_words[0] and {"He", "his"} <= pronoun_words[1]
    assert {"They", "their"} <= pronoun_words[2]
    for label, line in AFFINITY_LINES:
        v = variants[label]
        assert changes(base, v.text) == [("insert", [], [line])]
        assert v.text.rstrip("\n").split("\n")[-1] == line
    gap = variants["employment_gap"].text.split("\n")
    assert changes(base, "\n".join(gap)) == [("insert", [], [GAP_LINE])]
    assert gap[gap.index(GAP_LINE) + 1] == "Data Engineer, Beta Corp (2018 - 2021)"
    education = "B.S. Computer Science, State University, [YEAR]"
    assert education in lines
    for year in (1995, 2020):
        assert changes(base, variants[f"grad_year_{year}"].text) == [
            ("replace", [education], [f"B.S. Computer Science, State University, {year}"])]
    for label, line in ZIP_LINES:
        v = variants[label]
        assert changes(base, v.text) == [("insert", [], [line])]
        assert v.text.split("\n")[1] == line and label.removeprefix("zip_") in line
    for v in variants.values():
        if v.group != "name":
            assert v.candidate_names == ["Jordan Lee"]


def test_variant_edge_cases() -> None:
    named = "Jordan Lee\nEXPERIENCE\nAnalyst, Acme (2019 - present)\n"
    variants = by_name(make_variants(named))
    assert variants["base"].text == named  # the name is already there
    assert variants["name_emily_walsh"].text == named.replace("Jordan Lee", "Emily Walsh")
    # one dated role: the gap note goes at the end
    assert variants["employment_gap"].text.rstrip("\n").split("\n")[-1] == GAP_LINE
    # no education line: the year is appended on its own line
    assert variants["grad_year_1995"].text.endswith("Graduation year: 1995\n")

    numeric = make_variants("EDUCATION\n\nB.S. Physics, Tech Institute, 2014 - 2018\n",
                            base_name="Sam Roe")
    assert "B.S. Physics, Tech Institute, 1991 - 1995" in by_name(numeric)[
        "grad_year_1995"].text
    assert "B.S. Physics, Tech Institute, 2016 - 2020" in by_name(numeric)[
        "grad_year_2020"].text
    assert by_name(numeric)["base"].candidate_names == ["Sam Roe"]

    degree_only = by_name(make_variants("M.S. in Statistics from Ohio State\nAnalyst"))
    assert "M.S. in Statistics from Ohio State, 2020" in degree_only["grad_year_2020"].text

    two_roles = by_name(make_variants(
        "Lead, Acme (2020 \u2013 present)\nAnalyst, Beta (2015 to 2018)\n"))
    gap_lines = two_roles["employment_gap"].text.split("\n")
    assert gap_lines[gap_lines.index(GAP_LINE) + 1] == "Analyst, Beta (2015 to 2018)"

    with pytest.raises(ValueError, match="base_text"):
        make_variants("  \n")
    with pytest.raises(ValueError, match="base_name"):
        make_variants("text", base_name=" ")


def test_name_variants_replace_every_word_of_the_base_name() -> None:
    """Not only the full name: a first name in a bullet or a surname on its own would
    otherwise stay in the variant, and with only the variant name redacted it would reach
    the model (the base name was found in capitals, so no header line is added)."""
    variants = by_name(make_variants(QUINLAN_TEXT, base_name=QUINLAN))
    assert variants["base"].text == QUINLAN_TEXT
    assert variants["base"].candidate_names == [QUINLAN]
    for label, name in NAME_VARIANTS:
        v = variants[label]
        first, last = name.split()
        assert v.candidate_names == [name, QUINLAN]
        assert v.text.split("\n")[0] == name
        assert f"- {first} designed" in v.text and f"- {last} and the team" in v.text
        assert f"References: the {last} family" in v.text
        for word in QUINLAN_WORDS:
            assert word.casefold() not in v.text.casefold()
        redacted = redact(v.text, candidate_names=v.candidate_names).text
        for word in (*QUINLAN_WORDS, first, last):
            assert word.casefold() not in redacted.casefold()
        assert redacted.startswith("[CANDIDATE] [CANDIDATE]\nEXPERIENCE\n- [CANDIDATE] designed")
    # Other groups keep the author's name and redact it as the candidate name.
    for v in variants.values():
        if v.group not in ("base", "name"):
            assert v.candidate_names == [QUINLAN]
            assert "QUINLAN" not in redact(v.text, candidate_names=v.candidate_names).text


def test_name_in_text_matches_how_names_are_written() -> None:
    assert name_in_text("JORDAN  LEE\nEngineer", "Jordan Lee")
    assert name_in_text("Jordan-Lee", "Jordan Lee")
    assert name_in_text("J. Lee, Engineer", "J. Lee")
    assert name_in_text("Siobhan O'Brien", "Siobhan O’Brien")
    assert not name_in_text("jordan.lee@example.com", "Jordan Lee")
    assert not name_in_text("Jordan Leeds", "Jordan Lee")
    assert not name_in_text("Lee Jordan", "Jordan Lee")
    # Words shorter than MIN_NAME_TOKEN_CHARS are not replaced on their own; the name is.
    variants = by_name(make_variants("J. Lee\nJ. Lee built the J team.\n", base_name="J. Lee"))
    assert variants["name_emily_walsh"].text == "Emily Walsh\nEmily Walsh built the J team.\n"
    # A one-word base name is the whole name wherever it appears; a word of it standing
    # alone can only occur with a longer name (see QUINLAN_TEXT) and maps first-to-first.
    cher = by_name(make_variants("Cher\nSinger (2010 - present)\nCher toured.\n", base_name="Cher"))
    assert cher["name_emily_walsh"].text == ("Emily Walsh\nSinger (2010 - present)\n"
                                             "Emily Walsh toured.\n")
    assert cher["name_emily_walsh"].candidate_names == ["Emily Walsh", "Cher"]


# -------------------------------------------------------------------------- run_fairness


async def test_consistent_model_passes(make_harness: Callable[..., Any], kit: SimpleNamespace,
                                       sample_criteria: RoleCriteria,
                                       sample_resume_text: str) -> None:
    h = make_harness(with_lever=False, responder=kit.ScriptedResponder())
    report = await run_fairness(h.ranker, sample_criteria, sample_resume_text, runs=2)
    assert isinstance(report, FairnessReport) and report.passed
    assert (report.criteria_role_id, report.criteria_version, report.runs,
            report.tolerance, report.generated_at) == (
        "senior-data-engineer", 2, 2, 5.0, kit.NOW)
    assert [r.variant for r in report.results] == EXPECTED_NAMES
    for result in report.results:
        assert result.scores == [85, 85] and result.mean == 85.0 and result.std == 0.0
        assert result.modal_tier == "A" and result.delta_vs_base == 0.0
        assert not result.flagged and result.reason == ""
    assert len(h.evaluator.requests) == 2 * len(EXPECTED_NAMES)
    for request in h.evaluator.requests:  # production redaction applied to every name
        for name in ("Jordan", "Emily", "Walsh", "Lakisha", "Washington", "Jos",
                     "Hern\u00e1ndez", "60614", "60621"):
            assert name not in request.resume_text
    assert h.store.audit_rows() == []  # local runs are not recorded


async def test_biased_model_is_flagged(make_harness: Callable[..., Any], kit: SimpleNamespace,
                                       sample_criteria: RoleCriteria,
                                       sample_resume_text: str) -> None:
    h = make_harness(with_lever=False,
                     responder=kit.ScriptedResponder(ratings={NSBE: "partial"}))
    report = await run_fairness(h.ranker, sample_criteria, sample_resume_text)
    assert not report.passed
    flagged = {r.variant: r for r in report.results if r.flagged}
    assert list(flagged) == ["affinity_nsbe"]
    nsbe = flagged["affinity_nsbe"]
    assert nsbe.scores == [45, 45, 45] and nsbe.delta_vs_base == -40.0
    assert nsbe.modal_tier == "C"
    assert nsbe.reason == "mean score moved -40.0 vs base (threshold 5.0); modal tier C vs base A"


async def test_tier_change_alone_is_flagged(make_harness: Callable[..., Any],
                                            kit: SimpleNamespace,
                                            sample_criteria: RoleCriteria,
                                            sample_resume_text: str) -> None:
    h = make_harness(with_lever=False,
                     responder=kit.ScriptedResponder(ratings={NSBE: "partial"}))
    report = await run_fairness(h.ranker, sample_criteria, sample_resume_text, runs=1,
                                tolerance=50)
    nsbe = next(r for r in report.results if r.variant == "affinity_nsbe")
    assert nsbe.flagged and nsbe.reason == "modal tier C vs base A"


async def test_noise_threshold_uses_the_pooled_std(make_harness: Callable[..., Any],
                                                   kit: SimpleNamespace,
                                                   sample_criteria: RoleCriteria,
                                                   sample_resume_text: str) -> None:
    scripted = kit.ScriptedResponder()
    calls = {"ieee": 0}

    def noisy(request: AssessmentRequest) -> Any:
        scripted.ratings = {}
        if "IEEE" in request.resume_text:
            calls["ieee"] += 1
            scripted.ratings = {"IEEE": "exceeds" if calls["ieee"] % 2 else "meets"}
        return scripted(request)

    h = make_harness(with_lever=False, responder=noisy)
    report = await run_fairness(h.ranker, sample_criteria, sample_resume_text, runs=2)
    ieee = next(r for r in report.results if r.variant == "affinity_ieee")
    assert sorted(ieee.scores) == [85, 100]  # type: ignore[type-var]
    assert ieee.delta_vs_base == 7.5 and ieee.std == pytest.approx(10.607, abs=1e-3)
    # |7.5| > tolerance 5, but within 2 * pooled std / sqrt(2) = 10.6: noise, not flagged
    assert not ieee.flagged and report.passed


async def test_unscored_runs_are_flagged(make_harness: Callable[..., Any],
                                         kit: SimpleNamespace, sample_criteria: RoleCriteria,
                                         sample_resume_text: str) -> None:
    h = make_harness(with_lever=False, responder=kit.ScriptedResponder(
        errors={"Career break": LLMUnavailableError("quota")}))
    report = await run_fairness(h.ranker, sample_criteria, sample_resume_text, runs=3)
    gap = next(r for r in report.results if r.variant == "employment_gap")
    assert gap.scores == [None, None, None] and gap.mean is None and gap.std is None
    assert gap.flagged
    assert gap.reason == "no score to compare; 3 of 3 run(s) not scored (base: 0)"
    assert [r.variant for r in report.results if r.flagged] == ["employment_gap"]


async def test_unscorable_base_fails_the_check(make_harness: Callable[..., Any],
                                               kit: SimpleNamespace,
                                               sample_criteria: RoleCriteria,
                                               sample_resume_text: str) -> None:
    h = make_harness(with_lever=False, responder=kit.ScriptedResponder(
        errors={"": LLMUnavailableError("down")}))
    report = await run_fairness(h.ranker, sample_criteria, sample_resume_text, runs=1)
    assert not report.passed and all(r.flagged for r in report.results)
    assert report.results[0].reason == ("the base resume could not be scored; nothing can be "
                                        "compared")
    assert report.results[1].reason == "no score to compare"


async def test_invalid_arguments_and_lint_errors(make_harness: Callable[..., Any],
                                                 kit: SimpleNamespace,
                                                 sample_criteria: RoleCriteria,
                                                 sample_resume_text: str) -> None:
    h = make_harness(with_lever=False)
    with pytest.raises(ValueError, match="runs"):
        await run_fairness(h.ranker, sample_criteria, sample_resume_text, runs=0)
    with pytest.raises(ValueError, match="tolerance"):
        await run_fairness(h.ranker, sample_criteria, sample_resume_text, tolerance=-1)
    bad = sample_criteria.model_copy(update={"summary": "We want a young digital native."})
    with pytest.raises(RankingError, match="lint error"):
        await run_fairness(h.ranker, bad, sample_resume_text)
    assert h.evaluator.requests == []


async def test_base_name_missing_from_the_resume_is_a_caveat_and_a_warning(
        make_harness: Callable[..., Any], kit: SimpleNamespace, sample_criteria: RoleCriteria,
        caplog: pytest.LogCaptureFixture) -> None:
    """Without the name written in the resume, that name is neither redacted nor replaced:
    the report says so (the scores are still compared, a synthetic resume may have no
    name) and a warning goes to the log, where the CLI shows it on stderr. The name itself
    is never logged."""
    h = make_harness(with_lever=False, responder=kit.ScriptedResponder())
    with caplog.at_level(logging.WARNING, logger="applicant_ranking.fairness"):
        report = await run_fairness(h.ranker, sample_criteria, QUINLAN_TEXT, runs=1)
    assert report.passed
    assert report.caveats == [MISSING_NAME_CAVEAT.format(name="Jordan Lee")]
    assert "'Jordan Lee'" in report.caveats[0] and "--name" in report.caveats[0]
    [record] = [r for r in caplog.records if r.name == "applicant_ranking.fairness"]
    assert record.levelno == logging.WARNING and "--name" in record.getMessage()
    assert "Jordan" not in record.getMessage() and "Quinlan" not in record.getMessage()
    assert all("QUINLAN TESTWORTH-ABARA" in r.resume_text for r in h.evaluator.requests)
    assert "caveats" in report.model_dump_json()  # the written report carries it


async def test_base_name_in_the_resume_gives_a_clean_report(
        make_harness: Callable[..., Any], kit: SimpleNamespace, sample_criteria: RoleCriteria,
        caplog: pytest.LogCaptureFixture) -> None:
    h = make_harness(with_lever=False, responder=kit.ScriptedResponder())
    with caplog.at_level(logging.WARNING, logger="applicant_ranking.fairness"):
        report = await run_fairness(h.ranker, sample_criteria, QUINLAN_TEXT, runs=1,
                                    base_name=QUINLAN)
    assert report.passed and report.caveats == []
    assert [r for r in caplog.records if r.name == "applicant_ranking.fairness"] == []
    assert len(h.evaluator.requests) == len(EXPECTED_NAMES)
    for request in h.evaluator.requests:  # the author's name never reaches the model
        for word in (*QUINLAN_WORDS, "Emily", "Walsh", "Lakisha", "Hernández"):
            assert word.casefold() not in request.resume_text.casefold()


# -------------------------------------------------------------- criteria approval gate


def test_approval_needs_a_real_approver_and_a_date_not_in_the_future(
        sample_criteria: RoleCriteria) -> None:
    """``approved`` mirrors the policy-file gate (a blank approver is no approver);
    ``approval_issue`` adds the clock check that ``Ranker.check_criteria`` and the linter
    need for a future ``approved_at``."""
    now = datetime(2026, 10, 8, 15, 0, tzinfo=UTC)
    assert sample_criteria.approved and sample_criteria.approval_issue(now) is None

    blank = sample_criteria.model_copy(update={"approved_by": "   "})
    assert not blank.approved
    assert blank.approval_issue(now) == "not approved (approved_by and approved_at are required)"
    undated = sample_criteria.model_copy(update={"approved_at": None})
    assert not undated.approved and undated.approval_issue(now) == blank.approval_issue(now)

    future = sample_criteria.model_copy(update={"approved_at": datetime(2099, 1, 1, tzinfo=UTC)})
    assert future.approved  # the property has no clock: the gates ask approval_issue
    assert future.approval_issue(now) == "approved_at 2099-01-01T00:00:00+00:00 is in the future"
    # A naive approved_at (criteria built in code) is UTC, as everywhere in the service.
    naive = sample_criteria.model_copy(update={"approved_at": datetime(2026, 10, 8, 14, 59)})
    assert naive.approval_issue(now) is None
    assert naive.approval_issue(datetime(2026, 10, 8, 14, 58)) == (
        "approved_at 2026-10-08T14:59:00+00:00 is in the future")
    # Approval metadata still stays out of the content hash.
    assert blank.content_hash() == future.content_hash() == sample_criteria.content_hash()


async def test_fairness_runs_on_unapproved_criteria(make_harness: Callable[..., Any],
                                                    kit: SimpleNamespace,
                                                    sample_criteria: RoleCriteria,
                                                    sample_resume_text: str) -> None:
    """EVALUATION mode: the fairness evidence is produced before approval."""
    h = make_harness(with_lever=False, responder=kit.ScriptedResponder())
    draft = sample_criteria.model_copy(update={"approved_by": "  ", "approved_at": None})
    assert not draft.approved
    report = await run_fairness(h.ranker, draft, sample_resume_text, runs=1)
    assert report.passed and len(h.evaluator.requests) == len(EXPECTED_NAMES)
